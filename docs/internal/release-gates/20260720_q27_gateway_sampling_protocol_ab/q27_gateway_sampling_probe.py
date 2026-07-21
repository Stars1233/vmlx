#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8088"
MODEL = "dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP"
PROMPT = ""
EXPECTED = ""


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
            if typ in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta", "response.reasoning.delta", "response.output_text.reasoning_delta"} and data.get("delta"):
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


protocols = ("chat", "responses", "anthropic", "ollama")
variants = {
    "omitted": {},
    "explicit": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.05,
    },
}


def request_for(protocol, variant, stream):
    marker = f"Q27-GATEWAY-SAMPLING-{protocol.upper()}-{variant.upper()}-DONE"
    prompt = f"Reply exactly {marker} and nothing else. Do not call tools."
    sampling = variants[variant]
    if protocol == "responses":
        body = {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": 96,
            "enable_thinking": False,
            "stream": stream,
        }
        body.update(sampling)
    elif protocol == "anthropic":
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96,
            "enable_thinking": False,
            "stream": stream,
        }
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling:
                body[key] = sampling[key]
    elif protocol == "ollama":
        options = {"num_predict": 96}
        if sampling:
            options.update({
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "min_p": sampling["min_p"],
                "repeat_penalty": sampling["repetition_penalty"],
            })
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "options": options,
            "stream": stream,
        }
    else:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96,
            "enable_thinking": False,
            "stream": stream,
        }
        body.update(sampling)
        if stream:
            body["stream_options"] = {"include_usage": True}
    return marker, prompt, body


out = {
    "model": MODEL,
    "gateway": BASE,
    "variants": variants,
    "rows": {},
}
for variant in variants:
    for protocol in protocols:
        marker, prompt, stream_body = request_for(protocol, variant, True)
        if protocol == "ollama":
            streamed = stream_ollama(stream_body)
        else:
            path = {
                "chat": "/v1/chat/completions",
                "responses": "/v1/responses",
                "anthropic": "/v1/messages",
            }[protocol]
            streamed = stream_sse(path, stream_body, protocol)

        _, _, nonstream_body = request_for(protocol, variant, False)
        path = {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "anthropic": "/v1/messages",
            "ollama": "/api/chat",
        }[protocol]
        nonstream = post_json(path, nonstream_body)
        extractor = {
            "chat": chat_text,
            "responses": responses_text,
            "anthropic": anthropic_text,
            "ollama": ollama_text,
        }[protocol]
        nonstream_visible = extractor(nonstream["body"])
        out["rows"][f"{protocol}_{variant}"] = {
            "marker": marker,
            "prompt": prompt,
            "request_stream": stream_body,
            "stream": streamed,
            "nonstream": {
                "status": nonstream["status"],
                "elapsed_ms": nonstream["elapsed_ms"],
                "visible": nonstream_visible,
                "body": nonstream["body"],
            },
            "checks": {
                "stream_status_200": streamed["status"] == 200,
                "stream_nonempty": bool(streamed["visible"]),
                "stream_progressive": streamed["content_delta_count"] > 1 and streamed["last_content_ms"] > streamed["first_content_ms"],
                "stream_reasoning_empty": not streamed["reasoning"],
                "stream_terminal": bool(streamed["terminals"]),
                "stream_exact": streamed["visible"].strip() == marker,
                "nonstream_status_200": nonstream["status"] == 200,
                "nonstream_exact": nonstream_visible.strip() == marker,
            },
        }

out["checks"] = {
    "all_status_200": all(row["checks"]["stream_status_200"] and row["checks"]["nonstream_status_200"] for row in out["rows"].values()),
    "all_stream_nonempty": all(row["checks"]["stream_nonempty"] for row in out["rows"].values()),
    "all_stream_progressive": all(row["checks"]["stream_progressive"] for row in out["rows"].values()),
    "all_reasoning_empty": all(row["checks"]["stream_reasoning_empty"] for row in out["rows"].values()),
    "all_terminal": all(row["checks"]["stream_terminal"] for row in out["rows"].values()),
    "all_stream_exact": all(row["checks"]["stream_exact"] for row in out["rows"].values()),
    "all_nonstream_exact": all(row["checks"]["nonstream_exact"] for row in out["rows"].values()),
}

Path("/tmp/q27-gateway-sampling-protocol-ab.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({
    "checks": out["checks"],
    "rows": {
        key: {
            "stream_status": row["stream"]["status"],
            "stream_content_deltas": row["stream"]["content_delta_count"],
            "stream_reasoning_deltas": row["stream"]["reasoning_delta_count"],
            "stream_terminals": row["stream"]["terminals"],
            "stream_visible": row["stream"]["visible"],
            "nonstream_status": row["nonstream"]["status"],
            "nonstream_visible": row["nonstream"]["visible"],
            "checks": row["checks"],
        }
        for key, row in out["rows"].items()
    },
}, indent=2, ensure_ascii=False))
