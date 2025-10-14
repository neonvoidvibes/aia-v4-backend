import json

def sse_headers():
    # Prevent proxy buffering/transform
    return {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Transfer-Encoding": "chunked",
    }

def write_sse(data: dict) -> str:
    return f"data: {json.dumps(data, separators=(',',':'))}\n\n"
