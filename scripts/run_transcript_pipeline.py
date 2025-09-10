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

    # Optional dump of intermediates
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for k in ["segments", "mirror", "lens", "portal", "layer3", "layer4", "full"]:
            with open(os.path.join(args.dump_dir, f"{k}.json"), "w", encoding="utf-8") as f:
                json.dump(steps[k], f, ensure_ascii=False, indent=2)

    if not args.no_upsert:
        # Upsert final summary as Markdown to a single namespace (agent)
        md = full_summary_to_markdown(full)
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

    print(json.dumps({"ok": True, "upserted": (not args.no_upsert), "dump_dir": args.dump_dir or None}))


if __name__ == "__main__":
    main()
