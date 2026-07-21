#!/usr/bin/env python3
"""Probe a running vMLX Responses tool loop with timed SSE evidence."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


TOOLS = [
    {
        "type": "function",
        "name": "file_info",
        "description": "Get file metadata",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


def stream_response(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            events.append(
                {
                    "ms": round((time.perf_counter() - started) * 1000, 3),
                    "data": data,
                }
            )
    completed = [
        event["data"]["response"]
        for event in events
        if event["data"].get("type") == "response.completed"
    ]
    if len(completed) != 1:
        raise RuntimeError(f"expected one response.completed, got {len(completed)}")
    counts = Counter(event["data"].get("type") for event in events)
    reasoning_deltas = [
        event["data"].get("delta", "")
        for event in events
        if event["data"].get("type") == "response.reasoning_summary_text.delta"
    ]
    content_deltas = [
        event["data"].get("delta", "")
        for event in events
        if event["data"].get("type") == "response.output_text.delta"
    ]
    argument_deltas = [
        event["data"].get("delta", "")
        for event in events
        if event["data"].get("type") == "response.function_call_arguments.delta"
    ]
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "event_counts": dict(sorted(counts.items())),
        "reasoning_delta_count": len(reasoning_deltas),
        "reasoning_text": "".join(reasoning_deltas),
        "content_delta_count": len(content_deltas),
        "content_text": "".join(content_deltas),
        "argument_delta_count": len(argument_deltas),
        "argument_text": "".join(argument_deltas),
        "terminal": completed[0],
        "timed_events": events,
    }


def function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8006")
    parser.add_argument(
        "--model", default="models/Codex-Quant-Probe-OsaurusAgent-9b-JANG_4M"
    )
    parser.add_argument("--repo", default="/Users/eric/mlx/vllm-mlx-release-1.6.13")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/v1/responses"
    common = {
        "model": args.model,
        "stream": True,
        "store": True,
        "tools": TOOLS,
        "tool_choice": "auto",
        "enable_thinking": True,
        "temperature": 0,
        "max_output_tokens": 256,
    }
    round1_body = {
        **common,
        "input": (
            "[QUANT-API-R1-GATE] Call file_info exactly once with path "
            "panel/package.json. Do not answer from memory."
        ),
    }
    round1 = stream_response(endpoint, round1_body, args.timeout)
    calls = function_calls(round1["terminal"])

    manifest = Path(args.repo) / "panel/package.json"
    size_bytes = manifest.stat().st_size
    size_human = f"{size_bytes / 1024:.1f} KB"
    tool_output = (
        "Path: panel/package.json\n"
        "Type: file\n"
        f"Size: {size_human}\n"
        "Permissions: 0644"
    )
    call_id = calls[0].get("call_id") if len(calls) == 1 else ""
    round2_body = {
        **common,
        "previous_response_id": round1["terminal"].get("id"),
        "input": [
            {"type": "function_call_output", "call_id": call_id, "output": tool_output},
            {
                "role": "user",
                "content": (
                    "No more tools. Reply exactly "
                    f"QUANT-API-R2-GATE-DONE SIZE={size_human} and nothing else."
                ),
            },
        ],
    }
    round2 = stream_response(endpoint, round2_body, args.timeout)
    round2_calls = function_calls(round2["terminal"])
    expected = f"QUANT-API-R2-GATE-DONE SIZE={size_human}"

    checks = {
        "round1_exactly_one_call": len(calls) == 1,
        "round1_call_name": len(calls) == 1 and calls[0].get("name") == "file_info",
        "round1_arguments": len(calls) == 1
        and json.loads(calls[0].get("arguments") or "{}")
        == {"path": "panel/package.json"},
        "round1_reasoning_progressive": round1["reasoning_delta_count"] > 1,
        "round1_arguments_progressive": round1["argument_delta_count"] > 1,
        "previous_response_id_used": bool(round2_body["previous_response_id"]),
        "round2_tools_remained_available": bool(round2_body["tools"]),
        "round2_no_repeated_call": not round2_calls,
        "round2_reasoning_progressive": round2["reasoning_delta_count"] > 1,
        "round2_content_progressive": round2["content_delta_count"] > 1,
        "round2_exact_visible_text": round2["content_text"] == expected,
        "round2_terminal_text": round2["terminal"].get("output_text") == expected,
        "round1_one_terminal": round1["event_counts"].get("response.completed") == 1,
        "round2_one_terminal": round2["event_counts"].get("response.completed") == 1,
    }
    result = {
        "status": "pass" if all(checks.values()) else "review",
        "endpoint": endpoint,
        "model": args.model,
        "artifact": {
            "path": str(manifest),
            "size_bytes": size_bytes,
            "size_human": size_human,
        },
        "checks": checks,
        "round1": round1,
        "round2": round2,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
