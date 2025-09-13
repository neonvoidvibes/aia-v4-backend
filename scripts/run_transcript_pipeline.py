#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path
import logging

# Load env early from backend .env (and project root .env as fallback)
try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve()
    backend_env = here.parents[1] / ".env"
    root_env = here.parents[2] / ".env"
    for envp in [backend_env, root_env]:
        if envp.exists():
            load_dotenv(dotenv_path=str(envp), override=False)
except Exception:
    pass

sys.path.append('.')

from utils.multi_agent_summarizer.pipeline import summarize_transcript, run_pipeline_steps
from utils.embedding_handler import EmbeddingHandler
import boto3
from botocore.exceptions import ClientError


def _preflight():
    # LLM (Groq) for agents
    if not os.getenv("GROQ_API_KEY"):
        print("[warn] GROQ_API_KEY not set. Agents will fallback to minimal outputs.", file=sys.stderr)
    # Embeddings (OpenAI) for Pinecone upsert
    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY not set. Embedding upserts will fail.", file=sys.stderr)
    # Pinecone
    if not os.getenv("PINECONE_API_KEY"):
        print("[warn] PINECONE_API_KEY not set. Upserts will be skipped.", file=sys.stderr)


def _parse_s3_uri(s3_uri: str):
    """Return (bucket, key/prefix) from an s3:// URI."""
    if not s3_uri.startswith("s3://"):
        return None, None
    _, _, bucket_and_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    return bucket, key


def _list_s3_objects(bucket: str, prefix: str):
    """List S3 objects under prefix. Returns list of keys (immediate files only)."""
    s3c = boto3.client("s3")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    paginator = s3c.get_paginator('list_objects_v2')
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Filter to immediate children only (no subfolders)
            rel = key[len(prefix):]
            if not rel or "/" in rel:
                continue
            keys.append(key)
    return keys


def _read_s3_text(s3_uri: str) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    s3c = boto3.client("s3")
    obj = s3c.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def _move_s3_to_saved(s3_uri: str) -> str:
    """Move an S3 object to transcripts/saved/ alongside its current folder. Returns new s3 uri."""
    bucket, key = _parse_s3_uri(s3_uri)
    if not bucket or not key:
        return s3_uri
    # Skip if already in saved/
    if "/transcripts/saved/" in key:
        return s3_uri
    dirname = os.path.dirname(key)  # .../events/0000/transcripts
    basename = os.path.basename(key)
    dest_prefix = dirname + "/saved"
    dest_key = dest_prefix + "/" + basename
    s3c = boto3.client("s3")
    try:
        s3c.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': key}, Key=dest_key)
        s3c.delete_object(Bucket=bucket, Key=key)
        return f"s3://{bucket}/{dest_key}"
    except ClientError as e:
        print(f"[warn] Failed to move to saved/: {e}", file=sys.stderr)
        return s3_uri


def _process_single_text(agent: str, event: str, source_id: str, text: str, filename: str, dump_dir: str, do_upsert: bool):
    """Runs the pipeline for one text blob and optionally upserts. Returns dict with results."""
    from utils.multi_agent_summarizer.pipeline import extract_datetime_from_filename
    meeting_datetime = extract_datetime_from_filename(filename) if filename else None
    steps = run_pipeline_steps(text, meeting_datetime=meeting_datetime)
    final_md = steps.get("full_md", "")

    default_dump_dir = None
    written_files = []
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename or "transcript"))[0]
        seg_path = os.path.join(dump_dir, f"{base}__segments.json")
        with open(seg_path, "w", encoding="utf-8") as f:
            json.dump(steps["segments"], f, ensure_ascii=False, indent=2)
        written_files.append(seg_path)

        agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"),
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning"),
            ("reality_check_md", "reality_check"),
        ]
        for md_key, file_name in agent_outputs:
            if steps.get(md_key):
                out_path = os.path.join(dump_dir, f"{base}__{file_name}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(steps[md_key])
                written_files.append(out_path)

        full_content_parts = []
        full_agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"),
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning"),
        ]
        for md_key, file_name in full_agent_outputs:
            if steps.get(md_key):
                full_content_parts.append(steps[md_key])
        full_content = "\n\n=======\n\n".join(full_content_parts)
        full_path = os.path.join(dump_dir, f"{base}__full.md")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        written_files.append(full_path)

        steps_json_path = os.path.join(dump_dir, f"{base}__pipeline_steps.json")
        with open(steps_json_path, "w", encoding="utf-8") as f:
            steps_summary = {
                "agent_outputs": list(steps.keys()),
                "segments_count": len(steps.get("segments", [])),
                "pipeline_version": "business_first_v1",
            }
            json.dump(steps_summary, f, ensure_ascii=False, indent=2)
        written_files.append(steps_json_path)

    upserted = False
    if do_upsert:
        upsert_agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"),
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning"),
        ]
        full_content_parts = [steps[mk] for mk, _ in upsert_agent_outputs if steps.get(mk)]
        full_content = "\n\n=======\n\n".join(full_content_parts) or "# No Summary Generated\n"

        summary_filename = "full.md"
        if filename and filename.startswith("transcript_"):
            summary_filename = filename.replace("transcript_", "summary_", 1)
        elif filename:
            summary_filename = f"summary_{filename}"

        EmbeddingHandler(index_name="river", namespace=f"{agent}").embed_and_upsert(
            content=full_content,
            metadata={
                "agent_name": agent,
                "event_id": event,
                "transcript": event,
                "source": "transcript_full",
                "source_type": "summary",
                "content_category": "meeting_summary",
                "analysis_type": "multi_agent",
                "temporal_relevance": "time_sensitive",
                "meeting_date": meeting_datetime.split()[0] if meeting_datetime else None,
                "source_identifier": source_id,
                "file_name": summary_filename,
                "doc_id": f"{source_id}:summary",
                "pipeline_version": "business_first_v2",
            },
        )
        upserted = True

    return {
        "written": written_files,
        "upserted": upserted,
        "dump_dir": dump_dir,
        "filename": filename,
        "source_id": source_id,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--event", default="0000")
    p.add_argument("--s3-key", required=True)
    p.add_argument("--text-path", help="optional local file override")
    p.add_argument("--dump-dir", help="optional directory to dump intermediate JSON outputs")
    p.add_argument("--no-upsert", action="store_true", help="do not upsert to Pinecone")
    args = p.parse_args()

    _preflight()

    results = []

    # Case 1: local text path override (single file, no move)
    if args.text_path:
        with open(args.text_path, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(args.text_path)
        dump_dir = args.dump_dir or (os.path.dirname(os.path.abspath(args.text_path)) if os.path.isfile(args.text_path) else None)
        res = _process_single_text(
            agent=args.agent,
            event=args.event,
            source_id=args.text_path,
            text=text,
            filename=filename,
            dump_dir=dump_dir,
            do_upsert=not args.no_upsert,
        )
        results.append({**res, "moved_to": None})
    else:
        # s3-key path or prefix
        if args.s3_key.startswith("s3://") and args.s3_key.rstrip().endswith("/"):
            # Treat as folder: list and process transcript_* files
            bucket, prefix = _parse_s3_uri(args.s3_key)
            all_keys = _list_s3_objects(bucket, prefix)
            # Only files whose basename starts with transcript_
            target_keys = [k for k in all_keys if os.path.basename(k).startswith("transcript_")]
            for key in sorted(target_keys):
                s3_uri = f"s3://{bucket}/{key}"
                text = _read_s3_text(s3_uri)
                filename = os.path.basename(key)
                res = _process_single_text(
                    agent=args.agent,
                    event=args.event,
                    source_id=s3_uri,
                    text=text,
                    filename=filename,
                    dump_dir=args.dump_dir or None,
                    do_upsert=not args.no_upsert,
                )
                moved_to = None
                if not args.no_upsert:
                    moved_to = _move_s3_to_saved(s3_uri)
                results.append({**res, "moved_to": moved_to})
        else:
            # Single S3 object path
            # Reuse backend helper (supports s3:// and local paths)
            from api_server import _read_transcript_text_for_ma
            text = _read_transcript_text_for_ma(args.s3_key)
            filename = os.path.basename(args.s3_key)
            res = _process_single_text(
                agent=args.agent,
                event=args.event,
                source_id=args.s3_key,
                text=text,
                filename=filename,
                dump_dir=args.dump_dir or None,
                do_upsert=not args.no_upsert,
            )
            moved_to = None
            if not args.no_upsert and args.s3_key.startswith("s3://"):
                moved_to = _move_s3_to_saved(args.s3_key)
            results.append({**res, "moved_to": moved_to})

    print(json.dumps({
        "ok": True,
        "count": len(results),
        "results": results,
    }))


if __name__ == "__main__":
    main()
