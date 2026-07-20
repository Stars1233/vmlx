#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8022"
MODEL = "jangq-ai/Step-3.7-Flash-JANGTQ_K"
COMMON = "[STEP-SSD-SHORT-A-20260719] " + "".join(
    f"Step SSD record {i:03d} preserves token STEP-{i:03d} exactly. "
    for i in range(1, 9)
)
PROMPTS = {
    "exact_a": COMMON
    + "Do not use tools. Reply exactly STEP-SSD-SHORT-A-DONE and nothing else.",
    "partial_b": "[STEP-SSD-SHORT-A-20260719] "
    + "".join(
        f"Step SSD record {i:03d} preserves token STEP-{i:03d} exactly. "
        for i in range(1, 7)
    )
    + "Step SSD record 007 now changes token STEP-B07 exactly. "
    + "Step SSD record 008 now changes token STEP-B08 exactly. "
    + "Do not use tools. Reply exactly STEP-SSD-SHORT-B-DONE and nothing else.",
}


def slim_health():
    data = requests.get(BASE + "/health", timeout=30).json()
    cache = data.get("cache") or {}
    return {
        "model_loaded": data.get("model_loaded"),
        "last_request_time": data.get("last_request_time"),
        "native_cache": data.get("native_cache"),
        "turboquant_kv_cache": data.get("turboquant_kv_cache"),
        "scheduler": cache.get("scheduler_cache"),
        "block_disk": cache.get("block_disk_cache"),
        "totals": cache.get("totals"),
    }


def run(label, prompt):
    started = time.perf_counter()
    response = requests.post(
        BASE + "/v1/responses",
        json={
            "model": MODEL,
            "input": prompt,
            "stream": True,
            "temperature": 0.0,
            "max_output_tokens": 512,
            "reasoning_effort": "low",
            "tool_choice": "none",
        },
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    events = []
    visible = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        at_s = round(time.perf_counter() - started, 4)
        body = raw[6:]
        if body == "[DONE]":
            events.append({"at_s": at_s, "type": "DONE"})
            continue
        event = json.loads(body)
        typ = event.get("type", "")
        item = {"at_s": at_s, "type": typ}
        if typ == "response.output_text.delta":
            item["delta"] = event.get("delta", "")
            visible.append(item["delta"])
        elif typ == "response.output_text.done":
            item["text"] = event.get("text", "")
        events.append(item)
    content_times = [e["at_s"] for e in events if e["type"] == "response.output_text.delta"]
    completed = [e["at_s"] for e in events if e["type"] == "response.completed"]
    return {
        "label": label,
        "status": response.status_code,
        "visible": "".join(visible),
        "content_delta_count": len(content_times),
        "output_text_done_count": sum(
            e["type"] == "response.output_text.done" for e in events
        ),
        "completed_count": sum(e["type"] == "response.completed" for e in events),
        "last_content_to_completed_s": (
            round(completed[-1] - content_times[-1], 4)
            if content_times and completed
            else None
        ),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "events": events,
    }


result = {"health_before": slim_health(), "runs": []}
for label, prompt in PROMPTS.items():
    result["runs"].append(run(label, prompt))
    result[f"health_after_{label}"] = slim_health()
Path("/tmp/step37-ssdonly-short-responses.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "runs": [
                {k: run[k] for k in (
                    "label",
                    "status",
                    "visible",
                    "content_delta_count",
                    "output_text_done_count",
                    "completed_count",
                    "last_content_to_completed_s",
                    "elapsed_s",
                )}
                for run in result["runs"]
            ],
            "final_scheduler": result["health_after_partial_b"]["scheduler"],
            "final_block_disk": result["health_after_partial_b"]["block_disk"],
        },
        indent=2,
    )
)
