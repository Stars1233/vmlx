#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8012"
MODEL = "dealignai/DeepSeek-V4-Flash-JANG-CRACK"
COMMON_PREFIX = "[DSV4-0C9436-TERMINAL] " + "".join(
    f"Cache record {i:03d} preserves CEDAR-{i:03d} across native typed blocks. "
    for i in range(1, 49)
)
PROMPT_A = COMMON_PREFIX + (
    "Do not use tools. Do not explain the records. Reply with exactly "
    "DSV4-TERMINAL-A-DONE and nothing else."
)
PROMPT_B = COMMON_PREFIX + (
    "This suffix intentionally differs while the preceding block-aligned prefix stays identical. "
    "Do not use tools. Reply with exactly DSV4-TERMINAL-B-DONE and nothing else."
)


def health():
    return requests.get(BASE + "/health", timeout=30).json()


def slim_health(data):
    cache = data.get("cache") or {}
    scheduler = cache.get("scheduler_cache") or {}
    disk = cache.get("block_disk_cache") or {}
    native = cache.get("native_cache") or cache.get("model_cache") or {}
    return {
        "model_loaded": data.get("model_loaded"),
        "last_request_time": data.get("last_request_time"),
        "scheduler": scheduler,
        "block_disk": disk,
        "native": native,
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
    done_times = [
        e["at_s"] for e in events if e["type"] == "response.output_text.done"
    ]
    completed_times = [
        e["at_s"] for e in events if e["type"] == "response.completed"
    ]
    return {
        "label": label,
        "status": response.status_code,
        "prompt_chars": len(prompt),
        "visible": "".join(x["text"] for x in content),
        "content_delta_count": len(content),
        "content_deltas": content,
        "terminal_events": [e for e in events if e["type"] in terminal_types],
        "last_content_to_output_done_s": (
            round(done_times[-1] - last_content, 4)
            if last_content is not None and done_times
            else None
        ),
        "last_content_to_completed_s": (
            round(completed_times[-1] - last_content, 4)
            if last_content is not None and completed_times
            else None
        ),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "events": events,
    }


out = {
    "head": "0c9436bce7c6c2bdfc0a31c742b324269b203a50",
    "model": MODEL,
    "native_cache_expected": True,
    "generic_turboquant_expected": False,
    "health_before": slim_health(health()),
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
Path("/tmp/dsv4-0c9436-terminal-hierarchy.json").write_text(
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
                    "last_content_to_output_done_s": x["last_content_to_output_done_s"],
                    "last_content_to_completed_s": x["last_content_to_completed_s"],
                    "elapsed_s": x["elapsed_s"],
                }
                for x in out["runs"]
            ],
            "final_scheduler": out["health_after_partial_b"]["scheduler"],
            "final_block_disk": out["health_after_partial_b"]["block_disk"],
        },
        indent=2,
        ensure_ascii=False,
    )
)
