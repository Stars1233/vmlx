#!/usr/bin/env python3
"""Capture timestamped Nemotron Omni SSE without an explicit output cap."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", choices=("responses", "chat"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    if args.api == "responses":
        endpoint = f"{args.base_url}/v1/responses"
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
        }
    else:
        endpoint = f"{args.base_url}/v1/chat/completions"
        payload = {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    print(json.dumps({
        "api": args.api,
        "endpoint": endpoint,
        "model": args.model,
        "image": str(args.image),
        "omitted_fields": ["max_tokens", "max_completion_tokens", "max_output_tokens"],
        "prompt": args.prompt,
    }, sort_keys=True), flush=True)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        print(f"{time.perf_counter() - started:.3f}\tHTTP {response.status}", flush=True)
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                print(f"{time.perf_counter() - started:.3f}\t{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
