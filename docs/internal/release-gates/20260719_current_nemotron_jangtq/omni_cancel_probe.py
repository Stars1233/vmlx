#!/usr/bin/env python3
"""Disconnect from a live Omni Responses stream after real model deltas."""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlsplit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024")
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--delta-events", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    mime = mimetypes.guess_type(args.media.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(args.media.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    payload = {
        "model": args.model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": args.prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        "stream": True,
        "max_output_tokens": args.max_tokens,
        "enable_thinking": True,
        "temperature": 0,
    }

    parsed = urlsplit(args.base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/responses",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    response = connection.getresponse()
    print(json.dumps({
        "status": response.status,
        "reason": response.reason,
        "target_delta_events": args.delta_events,
        "media_sha256_source": args.media.name,
    }), flush=True)
    if response.status != 200:
        print(response.read().decode("utf-8", "replace"))
        return 1

    delta_events = 0
    event_type = ""
    while True:
        raw = response.fp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        elapsed = time.perf_counter() - started
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            if event_type in {
                "response.reasoning_summary_text.delta",
                "response.output_text.delta",
            }:
                delta_events += 1
                print(f"{elapsed:.3f}\t{event_type}\t{line[6:220]}", flush=True)
                if delta_events >= args.delta_events:
                    print(json.dumps({
                        "disconnected_after_delta_events": delta_events,
                        "elapsed_seconds": round(elapsed, 3),
                    }), flush=True)
                    response.close()
                    connection.close()
                    return 0
        if event_type in {"response.completed", "response.failed"}:
            print(json.dumps({
                "unexpected_terminal_before_disconnect": event_type,
                "delta_events": delta_events,
            }), flush=True)
            return 2

    print(json.dumps({"stream_ended_early": True, "delta_events": delta_events}))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
