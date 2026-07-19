#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8014"
MODEL = "jangq-ai/MiniMax-M2.7-Small-JANGTQ"
PROMPT = (
    "[M27-REASON-STREAM] Privately calculate 137 times 29 and double-check it. "
    "After the private reasoning, the visible answer must be exactly "
    "M27-REASON-STREAM-DONE and nothing else."
)


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def sse(path, payload, protocol):
    start = time.perf_counter()
    response = requests.post(BASE + path, json=payload, stream=True, timeout=300)
    result = {"status": response.status_code, "content": [], "reasoning": [], "terminals": [], "events": []}
    current_event = None
    active_anthropic_block = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            current_event = None
            continue
        if line.startswith("event: "):
            current_event = line[7:]
            result["events"].append({"at_ms": elapsed_ms(start), "event": current_event})
            continue
        if not line.startswith("data: "):
            continue
        raw_data = line[6:]
        if raw_data == "[DONE]":
            result["terminals"].append({"at_ms": elapsed_ms(start), "type": "DONE"})
            continue
        data = json.loads(raw_data)
        now = elapsed_ms(start)
        typ = data.get("type") or current_event or ""
        if protocol == "chat":
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                result["content"].append({"at_ms": now, "text": delta["content"]})
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning:
                result["reasoning"].append({"at_ms": now, "text": reasoning})
            if choice.get("finish_reason"):
                result["terminals"].append({"at_ms": now, "type": choice["finish_reason"]})
        elif protocol == "responses":
            if typ == "response.output_text.delta" and data.get("delta"):
                result["content"].append({"at_ms": now, "text": data["delta"]})
            if "reason" in typ and typ.endswith(".delta"):
                text = data.get("delta") or data.get("text") or ""
                if isinstance(text, str) and text:
                    result["reasoning"].append({"at_ms": now, "text": text})
            if typ in {"response.completed", "response.incomplete", "response.failed", "response.cancelled"}:
                result["terminals"].append({"at_ms": now, "type": typ})
        elif protocol == "anthropic":
            if typ == "content_block_start":
                active_anthropic_block = (data.get("content_block") or {}).get("type")
            delta = data.get("delta") or {}
            if typ == "content_block_delta":
                dtype = delta.get("type") or active_anthropic_block
                if dtype in {"text_delta", "text"} and delta.get("text"):
                    result["content"].append({"at_ms": now, "text": delta["text"]})
                if dtype in {"thinking_delta", "reasoning_delta", "thinking", "reasoning"}:
                    text = delta.get("thinking") or delta.get("reasoning") or delta.get("text") or ""
                    if text:
                        result["reasoning"].append({"at_ms": now, "text": text})
            if typ in {"message_stop", "error"}:
                result["terminals"].append({"at_ms": now, "type": typ})
    result["elapsed_ms"] = elapsed_ms(start)
    result["visible"] = "".join(item["text"] for item in result["content"])
    result["reasoning_text"] = "".join(item["text"] for item in result["reasoning"])
    result["content_delta_count"] = len(result["content"])
    result["reasoning_delta_count"] = len(result["reasoning"])
    return result


def ollama(payload):
    start = time.perf_counter()
    response = requests.post(BASE + "/api/chat", json=payload, stream=True, timeout=300)
    result = {"status": response.status_code, "content": [], "reasoning": [], "terminals": [], "objects": []}
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        data = json.loads(raw)
        now = elapsed_ms(start)
        message = data.get("message") or {}
        if message.get("content"):
            result["content"].append({"at_ms": now, "text": message["content"]})
        thinking = message.get("thinking") or message.get("reasoning") or ""
        if thinking:
            result["reasoning"].append({"at_ms": now, "text": thinking})
        if data.get("done"):
            result["terminals"].append({"at_ms": now, "type": data.get("done_reason") or "done"})
        result["objects"].append({"at_ms": now, "keys": sorted(data.keys()), "message_keys": sorted(message.keys())})
    result["elapsed_ms"] = elapsed_ms(start)
    result["visible"] = "".join(item["text"] for item in result["content"])
    result["reasoning_text"] = "".join(item["text"] for item in result["reasoning"])
    result["content_delta_count"] = len(result["content"])
    result["reasoning_delta_count"] = len(result["reasoning"])
    return result


common = {"model": MODEL, "temperature": 0.0}
out = {"model": MODEL, "prompt": PROMPT, "streams": {}}
out["streams"]["chat"] = sse(
    "/v1/chat/completions",
    {**common, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 1024, "enable_thinking": True, "stream": True, "stream_options": {"include_usage": True}},
    "chat",
)
out["streams"]["responses"] = sse(
    "/v1/responses",
    {**common, "input": PROMPT, "max_output_tokens": 1024, "enable_thinking": True, "stream": True},
    "responses",
)
out["streams"]["anthropic"] = sse(
    "/v1/messages",
    {**common, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 1024, "enable_thinking": True, "stream": True},
    "anthropic",
)
out["streams"]["ollama"] = ollama(
    {**common, "messages": [{"role": "user", "content": PROMPT}], "options": {"temperature": 0.0, "num_predict": 1024}, "think": True, "stream": True}
)
out["checks"] = {
    "status_200": all(v["status"] == 200 for v in out["streams"].values()),
    "reasoning_nonempty": all(v["reasoning_delta_count"] > 0 and bool(v["reasoning_text"]) for v in out["streams"].values()),
    "visible_nonempty": all(v["content_delta_count"] > 0 and bool(v["visible"]) for v in out["streams"].values()),
    "visible_exact": all(v["visible"].strip() == "M27-REASON-STREAM-DONE" for v in out["streams"].values()),
    "separate_rail_fields": all(
        v["reasoning_delta_count"] > 0
        and v["content_delta_count"] > 0
        and "<think>" not in v["visible"]
        and "</think>" not in v["visible"]
        for v in out["streams"].values()
    ),
    "terminal": all(bool(v["terminals"]) for v in out["streams"].values()),
}
Path("/tmp/m27-reasoning-stream.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"checks": out["checks"], "summary": {k: {"status": v["status"], "reasoning_delta_count": v["reasoning_delta_count"], "content_delta_count": v["content_delta_count"], "reasoning_chars": len(v["reasoning_text"]), "visible": v["visible"], "terminals": v["terminals"], "elapsed_ms": v["elapsed_ms"], "events": sorted({x["event"] for x in v.get("events", [])})} for k, v in out["streams"].items()}}, indent=2, ensure_ascii=False))
