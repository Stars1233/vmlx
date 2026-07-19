#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8014"
MODEL = "jangq-ai/MiniMax-M2.7-Small-JANGTQ"
PROMPT = (
    "[M27-PROTOCOL-STREAM] Reply with exactly these twelve lines and no other text: "
    "M27-ONE, M27-TWO, M27-THREE, M27-FOUR, M27-FIVE, M27-SIX, "
    "M27-SEVEN, M27-EIGHT, M27-NINE, M27-TEN, M27-ELEVEN, M27-TWELVE. "
    "Put one item per line."
)
EXPECTED = "\n".join(
    [
        "M27-ONE", "M27-TWO", "M27-THREE", "M27-FOUR", "M27-FIVE", "M27-SIX",
        "M27-SEVEN", "M27-EIGHT", "M27-NINE", "M27-TEN", "M27-ELEVEN", "M27-TWELVE",
    ]
)


def ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def post_json(path, payload):
    start = time.perf_counter()
    response = requests.post(BASE + path, json=payload, timeout=240)
    elapsed = ms(start)
    body = response.json()
    return {"status": response.status_code, "elapsed_ms": elapsed, "body": body}


def stream_sse(path, payload, protocol):
    start = time.perf_counter()
    response = requests.post(BASE + path, json=payload, stream=True, timeout=240)
    result = {
        "status": response.status_code,
        "events": [],
        "content_deltas": [],
        "reasoning_deltas": [],
        "terminals": [],
        "usage": [],
    }
    event_name = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            event_name = None
            continue
        if line.startswith("event: "):
            event_name = line[7:]
            result["events"].append({"at_ms": ms(start), "event": event_name})
            continue
        if not line.startswith("data: "):
            continue
        data_text = line[6:]
        if data_text == "[DONE]":
            result["terminals"].append({"at_ms": ms(start), "type": "DONE"})
            continue
        data = json.loads(data_text)
        now = ms(start)
        typ = data.get("type") or event_name
        if protocol == "chat":
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content:
                result["content_deltas"].append({"at_ms": now, "text": content})
            if reasoning:
                result["reasoning_deltas"].append({"at_ms": now, "text": reasoning})
            if choice.get("finish_reason"):
                result["terminals"].append({"at_ms": now, "type": choice["finish_reason"]})
            if data.get("usage"):
                result["usage"].append(data["usage"])
        elif protocol == "responses":
            if typ == "response.output_text.delta" and data.get("delta"):
                result["content_deltas"].append({"at_ms": now, "text": data["delta"]})
            if typ in {"response.reasoning_text.delta", "response.reasoning.delta", "response.output_text.reasoning_delta"} and data.get("delta"):
                result["reasoning_deltas"].append({"at_ms": now, "text": data["delta"]})
            if typ in {"response.completed", "response.incomplete", "response.failed", "response.cancelled"}:
                result["terminals"].append({"at_ms": now, "type": typ})
                usage = (data.get("response") or {}).get("usage")
                if usage:
                    result["usage"].append(usage)
        elif protocol == "anthropic":
            delta = data.get("delta") or {}
            if typ == "content_block_delta" and delta.get("type") == "text_delta":
                result["content_deltas"].append({"at_ms": now, "text": delta.get("text", "")})
            if typ == "content_block_delta" and delta.get("type") in {"thinking_delta", "reasoning_delta"}:
                result["reasoning_deltas"].append({"at_ms": now, "text": delta.get("thinking") or delta.get("reasoning") or ""})
            if typ in {"message_stop", "error"}:
                result["terminals"].append({"at_ms": now, "type": typ})
            if typ == "message_delta" and data.get("usage"):
                result["usage"].append(data["usage"])
    result["elapsed_ms"] = ms(start)
    result["visible"] = "".join(item["text"] for item in result["content_deltas"])
    result["reasoning"] = "".join(item["text"] for item in result["reasoning_deltas"])
    result["content_delta_count"] = len(result["content_deltas"])
    result["reasoning_delta_count"] = len(result["reasoning_deltas"])
    result["first_content_ms"] = result["content_deltas"][0]["at_ms"] if result["content_deltas"] else None
    result["last_content_ms"] = result["content_deltas"][-1]["at_ms"] if result["content_deltas"] else None
    return result


def stream_ollama(payload):
    start = time.perf_counter()
    response = requests.post(BASE + "/api/chat", json=payload, stream=True, timeout=240)
    result = {
        "status": response.status_code,
        "objects": [],
        "content_deltas": [],
        "reasoning_deltas": [],
        "terminals": [],
        "usage": [],
    }
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        data = json.loads(raw)
        now = ms(start)
        message = data.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("thinking") or message.get("reasoning") or ""
        if content:
            result["content_deltas"].append({"at_ms": now, "text": content})
        if reasoning:
            result["reasoning_deltas"].append({"at_ms": now, "text": reasoning})
        if data.get("done"):
            result["terminals"].append({"at_ms": now, "type": data.get("done_reason") or "done"})
            result["usage"].append({k: data.get(k) for k in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration")})
        result["objects"].append({"at_ms": now, "keys": sorted(data.keys()), "done": data.get("done")})
    result["elapsed_ms"] = ms(start)
    result["visible"] = "".join(item["text"] for item in result["content_deltas"])
    result["reasoning"] = "".join(item["text"] for item in result["reasoning_deltas"])
    result["content_delta_count"] = len(result["content_deltas"])
    result["reasoning_delta_count"] = len(result["reasoning_deltas"])
    result["first_content_ms"] = result["content_deltas"][0]["at_ms"] if result["content_deltas"] else None
    result["last_content_ms"] = result["content_deltas"][-1]["at_ms"] if result["content_deltas"] else None
    return result


def chat_text(body):
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def responses_text(body):
    if body.get("output_text"):
        return body["output_text"]
    texts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    texts.append(content.get("text", ""))
    return "".join(texts)


def anthropic_text(body):
    return "".join(item.get("text", "") for item in body.get("content") or [] if item.get("type") == "text")


def ollama_text(body):
    return (body.get("message") or {}).get("content") or ""


chat_base = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 192,
    "temperature": 0.0,
    "enable_thinking": False,
}
responses_base = {
    "model": MODEL,
    "input": PROMPT,
    "max_output_tokens": 192,
    "temperature": 0.0,
    "enable_thinking": False,
}
anthropic_base = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 192,
    "temperature": 0.0,
    "enable_thinking": False,
}
ollama_base = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "think": False,
    "options": {"temperature": 0.0, "num_predict": 192},
}

out = {"model": MODEL, "prompt": PROMPT, "expected": EXPECTED, "streams": {}, "nonstreams": {}}
out["streams"]["chat"] = stream_sse("/v1/chat/completions", {**chat_base, "stream": True, "stream_options": {"include_usage": True}}, "chat")
out["streams"]["responses"] = stream_sse("/v1/responses", {**responses_base, "stream": True}, "responses")
out["streams"]["anthropic"] = stream_sse("/v1/messages", {**anthropic_base, "stream": True}, "anthropic")
out["streams"]["ollama"] = stream_ollama({**ollama_base, "stream": True})

ns = post_json("/v1/chat/completions", {**chat_base, "stream": False})
ns["visible"] = chat_text(ns["body"])
out["nonstreams"]["chat"] = ns
ns = post_json("/v1/responses", {**responses_base, "stream": False})
ns["visible"] = responses_text(ns["body"])
out["nonstreams"]["responses"] = ns
ns = post_json("/v1/messages", {**anthropic_base, "stream": False})
ns["visible"] = anthropic_text(ns["body"])
out["nonstreams"]["anthropic"] = ns
ns = post_json("/api/chat", {**ollama_base, "stream": False})
ns["visible"] = ollama_text(ns["body"])
out["nonstreams"]["ollama"] = ns

out["checks"] = {
    "stream_status_200": all(v["status"] == 200 for v in out["streams"].values()),
    "stream_nonempty": all(bool(v["visible"]) for v in out["streams"].values()),
    "stream_progressive": all(v["content_delta_count"] > 1 and v["last_content_ms"] > v["first_content_ms"] for v in out["streams"].values()),
    "stream_reasoning_empty_when_disabled": all(v["reasoning_delta_count"] == 0 and not v["reasoning"] for v in out["streams"].values()),
    "stream_terminal": all(bool(v["terminals"]) for v in out["streams"].values()),
    "nonstream_status_200": all(v["status"] == 200 for v in out["nonstreams"].values()),
    "nonstream_nonempty": all(bool(v["visible"]) for v in out["nonstreams"].values()),
    "stream_nonstream_equal_per_protocol": all(out["streams"][k]["visible"] == out["nonstreams"][k]["visible"] for k in out["streams"]),
    "all_protocol_stream_visible_equal": len({v["visible"] for v in out["streams"].values()}) == 1,
    "exact_expected": all(v["visible"].strip() == EXPECTED for v in out["streams"].values()) and all(v["visible"].strip() == EXPECTED for v in out["nonstreams"].values()),
}

Path("/tmp/m27-protocol-parity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"checks": out["checks"], "stream_summary": {k: {x: v[x] for x in ("status", "content_delta_count", "reasoning_delta_count", "first_content_ms", "last_content_ms", "elapsed_ms", "terminals", "visible")} for k, v in out["streams"].items()}, "nonstream_summary": {k: {"status": v["status"], "elapsed_ms": v["elapsed_ms"], "visible": v["visible"]} for k, v in out["nonstreams"].items()}}, indent=2, ensure_ascii=False))
