#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8013"
MODEL = "jangq-ai/Zaya-8B-JANG_4M"
COMMON_PREFIX = "[ZAYA-TYPED-CCA-20260719] " + "".join(
    f"Typed record {i:03d} preserves ZETA-{i:03d} through the cache hierarchy. "
    for i in range(1, 49)
)
PROMPT_A = COMMON_PREFIX + (
    "Do not use tools. Do not explain the records. Reply with exactly "
    "ZAYA-TYPED-A-DONE and nothing else."
)
PROMPT_B = COMMON_PREFIX + (
    "This suffix differs while the preceding prefix remains identical. "
    "Do not use tools. Reply with exactly ZAYA-TYPED-B-DONE and nothing else."
)


def health():
    return requests.get(BASE + "/health", timeout=30).json()


def slim_health(data):
    cache = data.get("cache") or {}
    return {
        "model_loaded": data.get("model_loaded"),
        "last_request_time": data.get("last_request_time"),
        "kv_cache_quantization": data.get("kv_cache_quantization"),
        "turboquant_kv_cache": data.get("turboquant_kv_cache"),
        "native_cache": data.get("native_cache"),
        "scheduler": cache.get("scheduler_cache") or {},
        "block_disk": cache.get("block_disk_cache") or {},
        "totals": cache.get("totals") or {},
    }


def run(label, prompt):
    payload = {
        "model": MODEL,
        "input": prompt,
        "stream": True,
        "temperature": 0.0,
        "max_output_tokens": 64,
        "enable_thinking": False,
        "tool_choice": "none",
    }
    started = time.perf_counter()
    response = requests.post(
        BASE + "/v1/responses", json=payload, stream=True, timeout=600
    )
    response.raise_for_status()
    events = []
    content = []
    reasoning = []
    current_event = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            current_event = None
            continue
        at = round(time.perf_counter() - started, 4)
        if line.startswith("event: "):
            current_event = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        raw_data = line[6:]
        if raw_data == "[DONE]":
            events.append({"at_s": at, "type": "DONE"})
            continue
        data = json.loads(raw_data)
        typ = data.get("type") or current_event or ""
        event = {"at_s": at, "type": typ}
        if typ == "response.output_text.delta":
            event["delta"] = data.get("delta", "")
            content.append({"at_s": at, "text": data.get("delta", "")})
        elif typ in {
            "response.reasoning.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            event["delta"] = data.get("delta", "")
            reasoning.append({"at_s": at, "text": data.get("delta", "")})
        elif typ == "response.output_text.done":
            event["text"] = data.get("text", "")
        elif typ == "response.completed":
            event["response"] = data.get("response")
        events.append(event)
    terminal_types = {
        "response.output_text.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.cancelled",
        "DONE",
    }
    last_content = content[-1]["at_s"] if content else None
    completed_times = [e["at_s"] for e in events if e["type"] == "response.completed"]
    return {
        "label": label,
        "status": response.status_code,
        "prompt_chars": len(prompt),
        "visible": "".join(x["text"] for x in content),
        "reasoning": "".join(x["text"] for x in reasoning),
        "content_delta_count": len(content),
        "reasoning_delta_count": len(reasoning),
        "content_deltas": content,
        "reasoning_deltas": reasoning,
        "terminal_events": [e for e in events if e["type"] in terminal_types],
        "last_content_to_completed_s": (
            round(completed_times[-1] - last_content, 4)
            if last_content is not None and completed_times
            else None
        ),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "events": events,
    }


head = requests.get(BASE + "/health", timeout=30).json()
out = {
    "source_head": "1aa5f8e4994b0af3df63c86a16b48e5c4bb3cd3b",
    "model": MODEL,
    "native_cache_expected": "zaya_cca_v1",
    "generic_turboquant_expected": False,
    "health_before": slim_health(head),
    "runs": [],
}
for label, prompt in (("cold_a", PROMPT_A), ("warm_a", PROMPT_A), ("partial_b", PROMPT_B)):
    out["runs"].append(run(label, prompt))
    out[f"health_after_{label}"] = slim_health(health())

out["checks"] = {
    "all_status_200": all(x["status"] == 200 for x in out["runs"]),
    "all_visible_nonempty": all(bool(x["visible"].strip()) for x in out["runs"]),
    "all_progressive": all(x["content_delta_count"] > 1 for x in out["runs"]),
    "all_output_done_once": all(
        sum(1 for e in x["terminal_events"] if e["type"] == "response.output_text.done") == 1
        for x in out["runs"]
    ),
    "all_completed_once": all(
        sum(1 for e in x["terminal_events"] if e["type"] == "response.completed") == 1
        for x in out["runs"]
    ),
    "all_terminal_under_250ms": all(
        x["last_content_to_completed_s"] is not None
        and x["last_content_to_completed_s"] < 0.25
        for x in out["runs"]
    ),
}
Path("/tmp/zaya-typed-hierarchy.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        {
            "checks": out["checks"],
            "runs": [
                {
                    "label": x["label"],
                    "status": x["status"],
                    "visible": x["visible"],
                    "content_delta_count": x["content_delta_count"],
                    "reasoning_delta_count": x["reasoning_delta_count"],
                    "last_content_to_completed_s": x["last_content_to_completed_s"],
                    "elapsed_s": x["elapsed_s"],
                }
                for x in out["runs"]
            ],
            "final_scheduler": out["health_after_partial_b"]["scheduler"],
            "final_block_disk": out["health_after_partial_b"]["block_disk"],
            "final_native": out["health_after_partial_b"]["native_cache"],
        },
        indent=2,
        ensure_ascii=False,
    )
)
