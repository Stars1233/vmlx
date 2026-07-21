#!/usr/bin/env python3
import base64
import json
import sys
import time
from pathlib import Path

import requests

video = Path(sys.argv[1])
out = Path(sys.argv[2])
is_video = video.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}
mime = "video/mp4" if is_video else "image/png"
data_url = f"data:{mime};base64," + base64.b64encode(video.read_bytes()).decode()
media_item = (
    {"type": "input_video", "video_url": data_url}
    if is_video
    else {"type": "input_image", "image_url": data_url}
)
payload = {
    "model": "jangq-ai/gemma-4-12B-it-qat-JANG_4M",
    "input": [{
        "role": "user",
        "content": [
            media_item,
            {"type": "input_text", "text": "The media contains a small heading and a much larger lower alphanumeric string inside a rectangle. Read the pixels and reply with only the larger lower string. Do not return the heading."},
        ],
    }],
    "stream": True,
    "temperature": 0,
    "enable_thinking": False,
    "max_output_tokens": 128,
    "image_token_budget": 1120,
}
started = time.monotonic()
events = []
with requests.post("http://127.0.0.1:8141/v1/responses", json=payload, stream=True, timeout=300) as response:
    status = response.status_code
    response.raise_for_status()
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        body = raw[6:]
        if body == "[DONE]":
            events.append({"t": round(time.monotonic() - started, 6), "type": "[DONE]"})
            continue
        event = json.loads(body)
        events.append({"t": round(time.monotonic() - started, 6), "event": event})

content_events = [x for x in events if x.get("event", {}).get("type") == "response.output_text.delta"]
reasoning_events = [x for x in events if "reasoning" in str(x.get("event", {}).get("type", ""))]
done_events = [x for x in events if x.get("event", {}).get("type") == "response.output_text.done"]
completed_events = [x for x in events if x.get("event", {}).get("type") == "response.completed"]
content = "".join(str(x["event"].get("delta", "")) for x in content_events)
usage = completed_events[-1]["event"].get("response", {}).get("usage", {}) if completed_events else {}
last_content_t = content_events[-1]["t"] if content_events else None
terminal_t = completed_events[-1]["t"] if completed_events else None
summary = {
    "status": status,
    "fixture": str(video),
    "fixture_sha256": __import__("hashlib").sha256(video.read_bytes()).hexdigest(),
    "content": content,
    "content_delta_count": len(content_events),
    "reasoning_event_count": len(reasoning_events),
    "done_event_count": len(done_events),
    "completed_event_count": len(completed_events),
    "last_content_to_completed_seconds": round(terminal_t - last_content_t, 6) if terminal_t is not None and last_content_t is not None else None,
    "usage": usage,
    "checks": {
        "status_200": status == 200,
        "exact_content": content == "BANANA8426",
        "progressive_content": len(content_events) >= 2,
        "thinking_off_separate": len(reasoning_events) == 0,
        "single_done": len(done_events) == 1,
        "single_completed": len(completed_events) == 1,
    },
}
out.write_text(json.dumps({"summary": summary, "events": events}, indent=2))
print(json.dumps(summary, indent=2))
if not all(summary["checks"].values()):
    raise SystemExit(2)
