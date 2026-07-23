#!/usr/bin/env python3
"""One deterministic MiniMax-M3 native-cache request with before/after evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


MODEL = "JANGQ-AI/MiniMax-M3-Coder-Small"
PROBE_ID = "R17-M3-NATIVE-L2-20260723"


def shared_prefix(probe_id: str) -> str:
    rows = [
        (
            f"{probe_id} record {index:03d}: "
            "amber birch cobalt delta ember fjord granite harbor iris juniper "
            "keystone lantern maple northstar."
        )
        for index in range(192)
    ]
    return "\n".join(rows)


def build_prompt(
    layout: str,
    tail: str,
    expected: str,
    probe_id: str,
    stable_pad: str = "",
) -> str:
    prefix = shared_prefix(probe_id)
    pad = f"{stable_pad}\n" if stable_pad else ""
    if layout == "prefix":
        body = (
            f"{probe_id} continuous shared-prefix cache corpus begins.\n"
            f"{prefix}\n"
            f"{pad}"
            f"{probe_id} CHANGED TAIL={tail}.\n"
        )
    else:
        different_lead = "\n".join(
            (
                f"SUFFIX-ONLY-DIFFERENT-LEAD-{tail}-{index:03d}: "
                "violet willow xenon yellow zephyr."
            )
            for index in range(40)
        )
        body = (
            f"{different_lead}\n"
            f"{probe_id} shared corpus appears only after a different lead.\n"
            f"{prefix}\n"
            f"{pad}"
            f"{probe_id} SUFFIX-ONLY TAIL={tail}.\n"
        )
    return (
        f"{body}"
        "Do not call tools. Do not explain the corpus. "
        f"Reply exactly {expected} and nothing else."
    )


def health(base: str) -> dict[str, Any]:
    response = requests.get(base + "/health", timeout=30)
    response.raise_for_status()
    return response.json()


def health_summary(payload: dict[str, Any]) -> dict[str, Any]:
    cache = payload.get("cache") or {}
    return {
        "status": payload.get("status"),
        "model_name": payload.get("model_name"),
        "last_request_time": payload.get("last_request_time"),
        "native_cache": payload.get("native_cache"),
        "turboquant_kv_cache": payload.get("turboquant_kv_cache"),
        "kv_cache_quantization": payload.get("kv_cache_quantization"),
        "scheduler_cache": cache.get("scheduler_cache"),
        "block_disk_cache": cache.get("block_disk_cache"),
        "ssm_companion": cache.get("ssm_companion"),
        "totals": cache.get("totals"),
    }


def counter(summary: dict[str, Any], section: str, name: str) -> int:
    try:
        return int((summary.get(section) or {}).get(name) or 0)
    except (TypeError, ValueError):
        return 0


def delta(before: dict[str, Any], after: dict[str, Any], section: str, name: str) -> int:
    return counter(after, section, name) - counter(before, section, name)


def stream_chat(
    base: str, prompt: str, expected: str, model: str
) -> dict[str, Any]:
    started = time.monotonic()
    content: list[str] = []
    reasoning: list[str] = []
    terminals: list[str] = []
    raw_lines: list[str] = []
    usage: dict[str, Any] = {}
    first_delta_ms: float | None = None
    last_delta_ms: float | None = None
    with requests.post(
        base + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "seed": 17017,
            "max_tokens": 32,
            "enable_thinking": False,
        },
        stream=True,
        timeout=300,
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            raw_lines.append(raw)
            if not raw.startswith("data: "):
                continue
            encoded = raw[6:]
            if encoded == "[DONE]":
                terminals.append("[DONE]")
                continue
            data = json.loads(encoded)
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices") or []:
                part = choice.get("delta") or {}
                reasoning_text = str(
                    part.get("reasoning_content") or part.get("reasoning") or ""
                )
                content_text = str(part.get("content") or "")
                now_ms = round((time.monotonic() - started) * 1000, 3)
                if reasoning_text or content_text:
                    if first_delta_ms is None:
                        first_delta_ms = now_ms
                    last_delta_ms = now_ms
                if reasoning_text:
                    reasoning.append(reasoning_text)
                if content_text:
                    content.append(content_text)
                if choice.get("finish_reason"):
                    terminals.append(str(choice["finish_reason"]))
    visible = "".join(content)
    return {
        "status": status,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "first_delta_ms": first_delta_ms,
        "last_delta_ms": last_delta_ms,
        "content": visible,
        "reasoning": "".join(reasoning),
        "terminals": terminals,
        "usage": usage,
        "exact_expected": visible.strip() == expected,
        "raw_lines": raw_lines,
    }


def settled_health(base: str) -> dict[str, Any]:
    last_serialized = ""
    stable = 0
    latest: dict[str, Any] = {}
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        latest = health_summary(health(base))
        watched = {
            "scheduler_cache": latest.get("scheduler_cache"),
            "block_disk_cache": latest.get("block_disk_cache"),
            "ssm_companion": latest.get("ssm_companion"),
            "totals": latest.get("totals"),
        }
        serialized = json.dumps(watched, sort_keys=True)
        active = counter(latest, "scheduler_cache", "active_requests")
        if active == 0 and serialized == last_serialized:
            stable += 1
            if stable >= 3:
                return latest
        else:
            stable = 0
        last_serialized = serialized
        time.sleep(0.5)
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--layout", choices=["prefix", "suffix-only"], required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--probe-id", default=PROBE_ID)
    parser.add_argument("--stable-pad", default="")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    prompt = build_prompt(
        args.layout,
        args.tail,
        args.expected,
        args.probe_id,
        args.stable_pad,
    )

    before = health_summary(health(base))
    result = stream_chat(base, prompt, args.expected, args.model)
    after = settled_health(base)
    usage_details = (result.get("usage") or {}).get("prompt_tokens_details") or {}
    cache_deltas = {
        "scheduler_hits": delta(before, after, "scheduler_cache", "hits"),
        "scheduler_tokens_saved": delta(
            before, after, "scheduler_cache", "tokens_saved"
        ),
        "cache_hits": delta(before, after, "scheduler_cache", "cache_hits"),
        "cache_misses": delta(before, after, "scheduler_cache", "cache_misses"),
        "disk_hits": delta(before, after, "block_disk_cache", "disk_hits"),
        "disk_misses": delta(before, after, "block_disk_cache", "disk_misses"),
        "disk_writes": delta(before, after, "block_disk_cache", "disk_writes"),
        "disk_evictions": delta(
            before, after, "block_disk_cache", "disk_evictions"
        ),
        "tq_native_hits": delta(
            before, after, "block_disk_cache", "tq_native_hits"
        ),
    }
    output = {
        "model": args.model,
        "probe_id": args.probe_id,
        "stable_pad": args.stable_pad,
        "base": base,
        "layout": args.layout,
        "tail": args.tail,
        "expected": args.expected,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "before": before,
        "result": result,
        "after": after,
        "cache_deltas": cache_deltas,
        "usage_cache": {
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            "cached_tokens": usage_details.get("cached_tokens"),
            "cache_detail": usage_details.get("cache_detail"),
        },
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "layout": args.layout,
        "tail": args.tail,
        "status": result["status"],
        "exact_expected": result["exact_expected"],
        "first_delta_ms": result["first_delta_ms"],
        "elapsed_ms": result["elapsed_ms"],
        "terminals": result["terminals"],
        "usage_cache": output["usage_cache"],
        "cache_deltas": cache_deltas,
        "backend_before": (before.get("scheduler_cache") or {}).get("backend_mode"),
        "backend_after": (after.get("scheduler_cache") or {}).get("backend_mode"),
        "paged_ram_enabled": (after.get("scheduler_cache") or {}).get(
            "paged_ram_enabled"
        ),
        "disk_only": (after.get("scheduler_cache") or {}).get("disk_only"),
        "tq_native_enabled": (after.get("block_disk_cache") or {}).get(
            "tq_native_enabled"
        ),
    }
    print(json.dumps(compact, indent=2))
    ok = (
        result["status"] == 200
        and result["exact_expected"]
        and bool(result["terminals"])
        and not result["reasoning"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
