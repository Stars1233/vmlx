#!/usr/bin/env python3
"""Capture raw streamed math/currency bytes across the four public protocols."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


PROMPT = (
    "Do not call tools. Reply exactly two lines and preserve the literal dollar "
    "and TeX backslashes:\n"
    "M3-RAW-MATH\n"
    "CURRENCY=$43 TEX=\\(47 \\times 19 = 893\\)"
)
MODEL = "JANGQ-AI/MiniMax-M3-Coder-Small"


def elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def stream_sse(base: str, path: str, payload: dict[str, Any], protocol: str) -> dict[str, Any]:
    started = time.monotonic()
    raw_lines: list[str] = []
    deltas: list[dict[str, Any]] = []
    terminals: list[str] = []
    event_name = ""
    with requests.post(
        base + path,
        json=payload,
        stream=True,
        timeout=240,
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            raw_lines.append(raw)
            line = raw.strip()
            if not line:
                event_name = ""
                continue
            if line.startswith("event: "):
                event_name = line[7:]
                continue
            if not line.startswith("data: "):
                continue
            encoded = line[6:]
            if encoded == "[DONE]":
                terminals.append("[DONE]")
                continue
            data = json.loads(encoded)
            kind = str(data.get("type") or event_name)
            text = ""
            if protocol == "chat":
                choice = (data.get("choices") or [{}])[0]
                text = str((choice.get("delta") or {}).get("content") or "")
                if choice.get("finish_reason"):
                    terminals.append(str(choice["finish_reason"]))
            elif protocol == "responses":
                if kind == "response.output_text.delta":
                    text = str(data.get("delta") or "")
                if kind in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                    "response.cancelled",
                }:
                    terminals.append(kind)
            else:
                delta = data.get("delta") or {}
                if kind == "content_block_delta" and delta.get("type") == "text_delta":
                    text = str(delta.get("text") or "")
                if kind in {"message_stop", "error"}:
                    terminals.append(kind)
            if text:
                deltas.append({"at_ms": elapsed_ms(started), "text": text})
    content = "".join(str(row["text"]) for row in deltas)
    return {
        "status": status,
        "elapsed_ms": elapsed_ms(started),
        "raw_lines": raw_lines,
        "content_deltas": deltas,
        "content": content,
        "terminals": terminals,
    }


def stream_ollama(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    raw_lines: list[str] = []
    deltas: list[dict[str, Any]] = []
    terminals: list[str] = []
    with requests.post(
        base + "/api/chat",
        json=payload,
        stream=True,
        timeout=240,
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            raw_lines.append(raw)
            data = json.loads(raw)
            text = str((data.get("message") or {}).get("content") or "")
            if text:
                deltas.append({"at_ms": elapsed_ms(started), "text": text})
            if data.get("done"):
                terminals.append(str(data.get("done_reason") or "done"))
    content = "".join(str(row["text"]) for row in deltas)
    return {
        "status": status,
        "elapsed_ms": elapsed_ms(started),
        "raw_lines": raw_lines,
        "content_deltas": deltas,
        "content": content,
        "terminals": terminals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    chat = stream_sse(
        base,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": 160,
            "enable_thinking": False,
        },
        "chat",
    )
    responses = stream_sse(
        base,
        "/v1/responses",
        {
            "model": MODEL,
            "input": PROMPT,
            "stream": True,
            "temperature": 0,
            "max_output_tokens": 160,
            "enable_thinking": False,
        },
        "responses",
    )
    anthropic = stream_sse(
        base,
        "/v1/messages",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "temperature": 0,
            "max_tokens": 160,
            "thinking": {"type": "disabled"},
        },
        "anthropic",
    )
    ollama = stream_ollama(
        base,
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "think": False,
            "options": {"temperature": 0, "num_predict": 160},
        },
    )

    routes = {
        "chat": chat,
        "responses": responses,
        "anthropic": anthropic,
        "ollama": ollama,
    }
    checks = {
        "all_status_200": all(row["status"] == 200 for row in routes.values()),
        "all_progressive": all(len(row["content_deltas"]) > 1 for row in routes.values()),
        "all_terminal": all(bool(row["terminals"]) for row in routes.values()),
        "all_keep_currency": all("$43" in row["content"] for row in routes.values()),
        "all_keep_math_delimiters": all(
            "\\(" in row["content"] and "\\)" in row["content"]
            for row in routes.values()
        ),
        "all_keep_math_operator": all(
            "\\times" in row["content"] or "\\×" in row["content"]
            for row in routes.values()
        ),
        "no_renderer_html": all(
            "katex" not in row["content"].lower()
            and "math-inline" not in row["content"].lower()
            for row in routes.values()
        ),
        "all_route_content_equal": len({row["content"] for row in routes.values()}) == 1,
    }
    diagnostics = {
        # Keep an exact-copy miss visible without confusing it with transport
        # corruption. M3 currently substitutes the Unicode operator token and
        # emits `\×`; every direct/gateway route must still carry those same
        # raw model bytes.
        "all_exact_requested_tex_command": all(
            "\\times" in row["content"] for row in routes.values()
        )
    }
    output = {
        "base": base,
        "model": MODEL,
        "prompt": PROMPT,
        "routes": routes,
        "checks": checks,
        "diagnostics": diagnostics,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "base": base,
                "checks": checks,
                "diagnostics": diagnostics,
                "routes": {
                    name: {
                        "status": row["status"],
                        "delta_count": len(row["content_deltas"]),
                        "terminals": row["terminals"],
                        "content": row["content"],
                    }
                    for name, row in routes.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
