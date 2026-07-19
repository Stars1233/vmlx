#!/usr/bin/env python3
"""Capture timestamped text-only Chat or Responses SSE without buffering."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    thinking = None if args.thinking == "auto" else args.thinking == "on"
    if args.api == "responses":
        endpoint = f"{args.base_url.rstrip('/')}/v1/responses"
        payload = {
            "model": args.model,
            "input": args.prompt,
            "stream": True,
            "max_output_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    else:
        endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "stream": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream_options": {"include_usage": True},
        }
    if thinking is not None:
        payload["enable_thinking"] = thinking

    print(json.dumps({
        "api": args.api,
        "endpoint": endpoint,
        "model": args.model,
        "prompt": args.prompt,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }, sort_keys=True), flush=True)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        print(
            f"{time.perf_counter() - started:.3f}\tHTTP {response.status} "
            f"{response.headers.get_content_type()}",
            flush=True,
        )
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line:
                print(f"{time.perf_counter() - started:.3f}\t{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
