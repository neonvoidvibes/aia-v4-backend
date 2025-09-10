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
from utils.multi_agent_summarizer.markdown import full_summary_to_markdown
from utils.embedding_handler import EmbeddingHandler


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

    if args.text_path:
        with open(args.text_path, "r", encoding="utf-8") as f:
            text = f.read()
        source_id = args.text_path
    else:
        # Reuse backend helper (supports s3:// and local paths)
        from api_server import _read_transcript_text_for_ma
        text = _read_transcript_text_for_ma(args.s3_key)
        source_id = args.s3_key

    steps = run_pipeline_steps(text)
    full = steps["full"]

    # Determine default dump directory when a local text file is provided
    default_dump_dir = None
    written_files = []
    if args.text_path and os.path.isfile(args.text_path):
        default_dump_dir = os.path.dirname(os.path.abspath(args.text_path))

    dump_dir = args.dump_dir or default_dump_dir
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.text_path or "transcript"))[0]
        # segments json
        seg_path = os.path.join(dump_dir, f"{base}__segments.json")
        with open(seg_path, "w", encoding="utf-8") as f:
            json.dump(steps["segments"], f, ensure_ascii=False, indent=2)
        written_files.append(seg_path)
        # Write markdown artifacts
        for k in ["story_md","mirror_md","lens_md","portal_md","layer3_md","layer4_md","full_md"]:
            if steps.get(k):
                out_path = os.path.join(dump_dir, f"{base}__{k.replace('_md','')}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(steps[k])
                written_files.append(out_path)
        # Full JSON
        full_json_path = os.path.join(dump_dir, f"{base}__full.json")
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        written_files.append(full_json_path)

    if not args.no_upsert:
        # Upsert final summary as Markdown to a single namespace (agent)
        md = steps.get("full_md") or full_summary_to_markdown(full)
        # Save local markdown alongside source if possible
        if dump_dir:
            md_path = os.path.join(dump_dir, f"{base}__transcript_summary.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
            written_files.append(md_path)
        EmbeddingHandler(index_name="river", namespace=f"{args.agent}").embed_and_upsert(
            content=md,
            metadata={
                "agent_name": args.agent,
                "event_id": args.event,
                "transcript": args.event,
                "source": "transcript_summary",
                "source_type": "transcript",
                "source_identifier": source_id,
                "file_name": "transcript_summary.md",
                "doc_id": f"{source_id}:summary",
            },
        )
    print(json.dumps({
        "ok": True,
        "upserted": (not args.no_upsert),
        "dump_dir": dump_dir,
        "written": written_files
    }))


if __name__ == "__main__":
    main()
