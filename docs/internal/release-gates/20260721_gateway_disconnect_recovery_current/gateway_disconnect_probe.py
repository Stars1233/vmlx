#!/usr/bin/env python3
"""Exercise real Electron gateway disconnect cleanup across four protocols."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8088"
BACKEND = "http://127.0.0.1:8005"
MODEL = "dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP"
PROTOCOLS = ("chat", "responses", "anthropic", "ollama")


def payload(protocol: str, prompt: str, *, max_tokens: int) -> dict:
    if protocol == "responses":
        return {
            "model": MODEL,
            "input": prompt,
            "stream": True,
            "enable_thinking": False,
            "temperature": 0,
            "max_output_tokens": max_tokens,
        }
    if protocol == "anthropic":
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    if protocol == "ollama":
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "think": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }


def path(protocol: str) -> str:
    return {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
        "ollama": "/api/chat",
    }[protocol]


def visible_delta(protocol: str, line: str, event_name: str | None) -> str:
    if protocol == "ollama":
        obj = json.loads(line)
        return ((obj.get("message") or {}).get("content") or "")
    if not line.startswith("data: "):
        return ""
    raw = line[6:]
    if raw == "[DONE]":
        return ""
    obj = json.loads(raw)
    if protocol == "chat":
        return ((((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")) or "")
    if protocol == "responses":
        typ = obj.get("type") or event_name
        return obj.get("delta", "") if typ == "response.output_text.delta" else ""
    typ = obj.get("type") or event_name
    delta = obj.get("delta") or {}
    if typ == "content_block_delta" and delta.get("type") == "text_delta":
        return delta.get("text", "")
    return ""


def backend_state() -> dict:
    body = requests.get(BACKEND + "/health", timeout=10).json()
    scheduler = body.get("scheduler") or {}
    cache = body.get("cache") or {}
    scheduler_cache = cache.get("scheduler_cache") or {}
    return {
        "status": body.get("status"),
        "model_loaded": body.get("model_loaded"),
        "num_running": scheduler.get("num_running"),
        "num_waiting": scheduler.get("num_waiting"),
        "active_requests": scheduler_cache.get("active_requests"),
    }


def wait_for_idle(timeout: float = 15.0) -> dict:
    start = time.perf_counter()
    samples = []
    while time.perf_counter() - start < timeout:
        state = backend_state()
        state["at_ms"] = round((time.perf_counter() - start) * 1000, 2)
        samples.append(state)
        if state.get("num_running") == 0 and state.get("active_requests") == 0:
            return {"idle": True, "elapsed_ms": state["at_ms"], "samples": samples}
        time.sleep(0.1)
    return {"idle": False, "elapsed_ms": round(timeout * 1000, 2), "samples": samples}


def abort_after_content(protocol: str) -> dict:
    prompt = (
        "Output 500 lines. Each line must contain STREAM followed by its line "
        "number. Begin immediately and do not summarize."
    )
    start = time.perf_counter()
    response = requests.post(
        BASE + path(protocol),
        json=payload(protocol, prompt, max_tokens=1024),
        stream=True,
        timeout=(15, 300),
    )
    deltas = []
    event_name = None
    try:
        for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            line = raw.strip()
            if not line:
                event_name = None
                continue
            if line.startswith("event: "):
                event_name = line[7:]
                continue
            try:
                delta = visible_delta(protocol, line, event_name)
            except json.JSONDecodeError:
                continue
            if delta:
                deltas.append({
                    "at_ms": round((time.perf_counter() - start) * 1000, 2),
                    "text": delta,
                })
                if len(deltas) >= 3:
                    break
    finally:
        response.close()
    return {
        "status": response.status_code,
        "closed_at_ms": round((time.perf_counter() - start) * 1000, 2),
        "content_deltas_before_close": deltas,
        "idle_after_close": wait_for_idle(),
    }


def recovery(protocol: str) -> dict:
    marker = f"GATEWAY-{protocol.upper()}-DISCONNECT-RECOVERY-DONE"
    prompt = f"Reply exactly {marker} and nothing else."
    start = time.perf_counter()
    response = requests.post(
        BASE + path(protocol),
        json=payload(protocol, prompt, max_tokens=96),
        stream=True,
        timeout=(15, 300),
    )
    deltas = []
    event_name = None
    terminal = []
    for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
        if raw is None:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()
        if not line:
            event_name = None
            continue
        if line.startswith("event: "):
            event_name = line[7:]
            continue
        if protocol == "ollama":
            obj = json.loads(line)
            text = ((obj.get("message") or {}).get("content") or "")
            if text:
                deltas.append(text)
            if obj.get("done"):
                terminal.append(obj.get("done_reason") or "done")
            continue
        if not line.startswith("data: "):
            continue
        raw_data = line[6:]
        if raw_data == "[DONE]":
            terminal.append("DONE")
            continue
        obj = json.loads(raw_data)
        text = visible_delta(protocol, line, event_name)
        if text:
            deltas.append(text)
        typ = obj.get("type") or event_name
        if protocol == "chat":
            reason = ((obj.get("choices") or [{}])[0].get("finish_reason"))
            if reason:
                terminal.append(reason)
        elif protocol == "responses" and typ in {
            "response.completed", "response.incomplete", "response.failed"
        }:
            terminal.append(typ)
        elif protocol == "anthropic" and typ in {"message_stop", "error"}:
            terminal.append(typ)
    response.close()
    visible = "".join(deltas).strip()
    return {
        "status": response.status_code,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        "expected": marker,
        "visible": visible,
        "content_delta_count": len(deltas),
        "terminal": terminal,
        "exact": visible == marker,
        "idle_after_recovery": wait_for_idle(),
    }


out = {
    "model": MODEL,
    "gateway": BASE,
    "backend": BACKEND,
    "gateway_health_before": requests.get(BASE + "/health", timeout=10).json(),
    "backend_before": backend_state(),
    "rows": {},
}
target = Path("/tmp/q27-gateway-disconnect-proof.json")
for protocol in PROTOCOLS:
    print(f"{protocol}: disconnect", flush=True)
    aborted = abort_after_content(protocol)
    print(f"{protocol}: recovery", flush=True)
    recovered = recovery(protocol)
    out["rows"][protocol] = {"disconnect": aborted, "recovery": recovered}
    target.write_text(json.dumps(out, indent=2) + "\n")
out["gateway_health_after"] = requests.get(BASE + "/health", timeout=10).json()
out["backend_after"] = backend_state()
out["checks"] = {
    protocol: {
        "stream_started": row["disconnect"]["status"] == 200
        and len(row["disconnect"]["content_deltas_before_close"]) >= 3,
        "cancelled_to_idle": row["disconnect"]["idle_after_close"]["idle"],
        "recovery_200": row["recovery"]["status"] == 200,
        "recovery_exact": row["recovery"]["exact"],
        "recovery_progressive": row["recovery"]["content_delta_count"] > 1,
        "recovery_terminal": bool(row["recovery"]["terminal"]),
        "recovery_idle": row["recovery"]["idle_after_recovery"]["idle"],
    }
    for protocol, row in out["rows"].items()
}
target.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"proof": str(target), "checks": out["checks"]}, indent=2))
