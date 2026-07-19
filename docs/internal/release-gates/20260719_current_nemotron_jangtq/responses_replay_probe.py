#!/usr/bin/env python3
"""Capture a deterministic Responses stream from a supplied history JSON."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--skip-prefix-cache", action="store_true")
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "input": json.loads(args.input_json.read_text()),
        "stream": True,
        "max_output_tokens": 256,
        "temperature": 0,
        "seed": 20260719,
        "skip_prefix_cache": args.skip_prefix_cache,
    }
    print(json.dumps({
        "skip_prefix_cache": args.skip_prefix_cache,
        "seed": payload["seed"],
        "input_json": str(args.input_json),
    }, sort_keys=True), flush=True)
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        print(f"{time.perf_counter() - started:.3f}\tHTTP {response.status}", flush=True)
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line:
                print(f"{time.perf_counter() - started:.3f}\t{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
