#!/usr/bin/env python3
"""Run cache-bypassed Laguna reasoning SSE probes and preserve full events."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).with_name("laguna-reasoning-soak")
ROOT.mkdir(exist_ok=True)

summaries = []
for index in range(1, 6):
    marker = f"SOAK-{index}=45"
    body = {
        "model": "jangq-ai/Laguna-M.1-JANG_2L",
        "input": (
            f"[LAG-REASON-SOAK-{index}] Compute 17 + 28. Use reasoning "
            f"mode, then reply with exactly {marker} and no other visible text."
        ),
        "stream": True,
        "enable_thinking": True,
        "temperature": 1.0,
        "max_output_tokens": 1024,
        "skip_prefix_cache": True,
    }
    request = Request(
        "http://127.0.0.1:8015/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        raw = response.read().decode()
    (ROOT / f"soak-{index}.sse").write_text(raw)

    events = []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        events.append(json.loads(line.removeprefix("data: ")))

    reasoning = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "response.reasoning_summary_text.delta"
    )
    content = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "response.output_text.delta"
    )
    completed = next(
        (event.get("response", {}) for event in events if event.get("type") == "response.completed"),
        {},
    )
    summary = {
        "index": index,
        "expected": marker,
        "content": content,
        "reasoning": reasoning,
        "reasoning_chars": len(reasoning),
        "content_exact": content == marker,
        "status": completed.get("status"),
        "usage": completed.get("usage"),
        "warnings": completed.get("warnings"),
    }
    summaries.append(summary)
    print(json.dumps(summary, ensure_ascii=False))

(ROOT / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
