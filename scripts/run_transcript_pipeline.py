#!/usr/bin/env python3
import argparse
import json
import sys

sys.path.append('.')

from utils.multi_agent_summarizer.pipeline import summarize_transcript
from utils.embedding_handler import EmbeddingHandler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--event", default="0000")
    p.add_argument("--s3-key", required=True)
    p.add_argument("--text-path", help="optional local file override")
    args = p.parse_args()

    if args.text_path:
        with open(args.text_path, "r", encoding="utf-8") as f:
            text = f.read()
        source_id = args.text_path
    else:
        # Reuse backend helper
        from api_server import _read_transcript_text_for_ma
        text = _read_transcript_text_for_ma(args.s3_key)
        source_id = args.s3_key

    full = summarize_transcript(args.agent, args.event, text, source_id)

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        EmbeddingHandler(index_name="river", namespace=f"{args.agent}.{layer}").embed_and_upsert(
            content=json.dumps(full[layer], ensure_ascii=False),
            metadata={
                "agent_name": args.agent,
                "event_id": args.event,
                "transcript": args.event,
                "source": "transcript_summary",
                "source_type": "transcript",
                "source_identifier": source_id,
                "file_name": f"{layer}.json",
                "doc_id": f"{source_id}:{layer}",
            },
        )

    print(json.dumps({"ok": True, "layers": ["layer1", "layer2", "layer3", "layer4"]}))


if __name__ == "__main__":
    main()

