#!/usr/bin/env python3
"""Compare Laguna Auto, explicit On, and explicit Off Responses streams."""

import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8008/v1/responses"
MODEL = "jangq-ai/Laguna-S-2.1-JANG_2L"
PROMPT = (
    "[LAG-S21-AUTO-AB] Privately calculate 143 times 27 and double-check it. "
    "Then answer in one sentence."
)


def run(label: str, explicit: bool | None) -> dict:
    payload = {
        "model": MODEL,
        "input": PROMPT,
        "stream": True,
        "store": False,
        "temperature": 0.0,
        "max_output_tokens": 256,
        "cache_salt": label,
    }
    if explicit is not None:
        payload["enable_thinking"] = explicit
        payload["chat_template_kwargs"] = {"enable_thinking": explicit}

    started = time.perf_counter()
    reasoning_deltas: list[str] = []
    content_deltas: list[str] = []
    terminal: list[str] = []
    timeline: list[dict] = []
    with requests.post(BASE, json=payload, stream=True, timeout=300) as response:
        status = response.status_code
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data_text = raw[6:]
            if data_text == "[DONE]":
                continue
            data = json.loads(data_text)
            event_type = data.get("type", "")
            delta = data.get("delta", "")
            if event_type.endswith(".delta") and "reason" in event_type and delta:
                reasoning_deltas.append(delta)
            if event_type == "response.output_text.delta" and delta:
                content_deltas.append(delta)
            if event_type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
                "response.cancelled",
            }:
                terminal.append(event_type)
            timeline.append(
                {
                    "at_ms": round((time.perf_counter() - started) * 1000, 2),
                    "type": event_type,
                    "delta": delta,
                }
            )

    reasoning = "".join(reasoning_deltas)
    visible = "".join(content_deltas)
    return {
        "label": label,
        "explicit_enable_thinking": explicit,
        "payload": payload,
        "status": status,
        "reasoning": reasoning,
        "visible": visible,
        "reasoning_delta_count": len(reasoning_deltas),
        "content_delta_count": len(content_deltas),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "visible_sha256": hashlib.sha256(visible.encode()).hexdigest(),
        "terminal": terminal,
        "timeline": timeline,
    }


rows = [run("auto", None), run("on", True), run("off", False)]
by_label = {row["label"]: row for row in rows}
checks = {
    "all_http_200": all(row["status"] == 200 for row in rows),
    "auto_reasoning_progressive": by_label["auto"]["reasoning_delta_count"] > 1,
    "auto_content_progressive": by_label["auto"]["content_delta_count"] > 1,
    "auto_matches_explicit_on_reasoning": (
        by_label["auto"]["reasoning"] == by_label["on"]["reasoning"]
    ),
    "auto_matches_explicit_on_visible": (
        by_label["auto"]["visible"] == by_label["on"]["visible"]
    ),
    "off_has_no_reasoning": by_label["off"]["reasoning_delta_count"] == 0,
    "off_content_progressive": by_label["off"]["content_delta_count"] > 1,
}
output = {"checks": checks, "rows": rows}
Path("/tmp/laguna-s21-reasoning-auto-ab.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps({"checks": checks, "summary": [{k: row[k] for k in (
    "label", "status", "reasoning_delta_count", "content_delta_count",
    "reasoning_sha256", "visible_sha256", "terminal"
)} for row in rows]}, indent=2))
