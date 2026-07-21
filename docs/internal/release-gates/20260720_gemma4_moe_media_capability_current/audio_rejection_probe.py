#!/usr/bin/env python3
"""Capture the expected API rejection for a bundle without an audio tower."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()

    encoded = base64.b64encode(args.audio.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Transcribe this audio."},
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": "wav"},
                },
            ],
        }],
        "stream": True,
        "max_output_tokens": 64,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(json.dumps({"status": response.status, "body": body}))
            return 1
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(json.dumps({"status": error.code, "body": json.loads(body)}))
        return 0 if error.code == 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
