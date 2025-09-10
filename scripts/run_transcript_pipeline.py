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

    # Extract filename for datetime extraction
    filename = None
    if args.text_path:
        filename = os.path.basename(args.text_path)
    elif args.s3_key:
        filename = os.path.basename(args.s3_key)
    
    # Extract meeting datetime from filename if available
    from utils.multi_agent_summarizer.pipeline import extract_datetime_from_filename
    meeting_datetime = extract_datetime_from_filename(filename) if filename else None
    
    steps = run_pipeline_steps(text, meeting_datetime=meeting_datetime)
    # New pipeline doesn't have "full" - use full_md directly
    final_md = steps.get("full_md", "")

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
        # Write individual agent outputs (no executive summaries, next actions now in business reality)
        agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"), 
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning"),
            ("reality_check_md", "reality_check")
        ]
        
        for md_key, file_name in agent_outputs:
            if steps.get(md_key):
                out_path = os.path.join(dump_dir, f"{base}__{file_name}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(steps[md_key])
                written_files.append(out_path)
        
        # Create concatenated full.md with all agent outputs (skip reality check, next actions now in business reality)
        full_content_parts = []
        full_agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"), 
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning")
        ]
        
        for md_key, file_name in full_agent_outputs:
            if steps.get(md_key):
                full_content_parts.append(steps[md_key])
        
        full_content = "\n\n=======\n\n".join(full_content_parts)
        full_path = os.path.join(dump_dir, f"{base}__full.md")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        written_files.append(full_path)
        
        # Steps metadata JSON
        steps_json_path = os.path.join(dump_dir, f"{base}__pipeline_steps.json")
        with open(steps_json_path, "w", encoding="utf-8") as f:
            # Create summary dict without the large markdown content
            steps_summary = {
                "agent_outputs": list(steps.keys()),
                "segments_count": len(steps.get("segments", [])),
                "pipeline_version": "business_first_v1"
            }
            json.dump(steps_summary, f, ensure_ascii=False, indent=2)
        written_files.append(steps_json_path)

    if not args.no_upsert:
        # Create full content for upserting (all agent outputs except reality check, next actions now in business reality)
        full_content_parts = []
        upsert_agent_outputs = [
            ("context_md", "context"),
            ("business_reality_md", "business_reality"), 
            ("org_dynamics_md", "org_dynamics"),
            ("strategic_md", "strategic_implications"),
            ("wisdom_learning_md", "wisdom_learning")
        ]
        
        for md_key, file_name in upsert_agent_outputs:
            if steps.get(md_key):
                full_content_parts.append(steps[md_key])
        
        full_content = "\n\n=======\n\n".join(full_content_parts) or "# No Summary Generated\n"
        
        # Generate summary filename by replacing 'transcript_' with 'summary_'
        summary_filename = "full.md"  # default fallback
        if filename and filename.startswith("transcript_"):
            summary_filename = filename.replace("transcript_", "summary_", 1)
        elif filename:
            # If filename doesn't start with transcript_, prepend summary_
            summary_filename = f"summary_{filename}"
        
        EmbeddingHandler(index_name="river", namespace=f"{args.agent}").embed_and_upsert(
            content=full_content,
            metadata={
                "agent_name": args.agent,
                "event_id": args.event,
                "transcript": args.event,
                "source": "transcript_full",
                "source_type": "summary",  # Changed from "transcript" to "summary"
                "source_identifier": source_id,
                "file_name": summary_filename,
                "doc_id": f"{source_id}:summary",  # Changed from ":full" to ":summary"
                "pipeline_version": "business_first_v2",
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
