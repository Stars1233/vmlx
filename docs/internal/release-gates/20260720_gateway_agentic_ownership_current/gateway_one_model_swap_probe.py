#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8088"
MODELS = [
    ("dealignai/DeepSeek-V4-Flash-JANG-CRACK", "DSV4-GATEWAY-SWAP-DONE"),
    ("JANGQ-AI/MiniMax-M3-Coder-Small", "M3-GATEWAY-SWAP-DONE"),
]


def gateway_health() -> dict:
    return requests.get(BASE + "/health", timeout=10).json()


def processes() -> list[str]:
    output = subprocess.check_output(["ps", "-axo", "pid,rss,etime,command"], text=True)
    return [
        line.strip()
        for line in output.splitlines()
        if "vmlx-engine serve" in line or "vmlx_engine.cli serve" in line
    ]


def stream_chat(model: str, marker: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Reply exactly {marker} and nothing else. Do not call tools.",
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 128,
        "temperature": 0,
        "enable_thinking": False,
    }
    started = time.monotonic()
    rows = []
    content = []
    reasoning = []
    finish = []
    done = 0
    with requests.post(
        BASE + "/v1/chat/completions", json=payload, stream=True, timeout=900
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not raw:
                continue
            at = round(time.monotonic() - started, 6)
            if not raw.startswith("data: "):
                rows.append({"at": at, "raw": raw})
                continue
            value = raw[6:]
            if value == "[DONE]":
                done += 1
                rows.append({"at": at, "data": "[DONE]"})
                continue
            data = json.loads(value)
            rows.append({"at": at, "data": data})
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append({"at": at, "text": delta["content"]})
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning.append(
                        {"at": at, "text": delta.get("reasoning_content") or delta.get("reasoning")}
                    )
                if choice.get("finish_reason") is not None:
                    finish.append(choice["finish_reason"])
    return {
        "status": status,
        "elapsed": round(time.monotonic() - started, 6),
        "events": rows,
        "content": "".join(x["text"] for x in content),
        "content_deltas": content,
        "reasoning": "".join(x["text"] for x in reasoning),
        "reasoning_deltas": reasoning,
        "finish": finish,
        "done": done,
    }


out = {
    "before": {"gateway": gateway_health(), "processes": processes()},
    "steps": [],
}
for model, marker in MODELS:
    result = stream_chat(model, marker)
    time.sleep(1)
    out["steps"].append(
        {
            "model": model,
            "marker": marker,
            "stream": result,
            "gateway": gateway_health(),
            "processes": processes(),
        }
    )

out["checks"] = {
    "single_mode_true_all_states": out["before"]["gateway"].get("single_model_mode") is True
    and all(step["gateway"].get("single_model_mode") is True for step in out["steps"]),
    "one_engine_after_each_swap": all(len(step["processes"]) == 1 for step in out["steps"]),
    "requested_backend_only_running": all(
        [
            backend["model"]
            for backend in step["gateway"].get("backends", [])
            if backend.get("status") == "running"
        ]
        == [step["model"]]
        for step in out["steps"]
    ),
    "exact_progressive_streams": all(
        step["stream"]["status"] == 200
        and step["stream"]["content"] == step["marker"]
        and len(step["stream"]["content_deltas"]) > 1
        and not step["stream"]["reasoning"]
        and step["stream"]["finish"] == ["stop"]
        and step["stream"]["done"] == 1
        for step in out["steps"]
    ),
}
Path("/tmp/gateway-one-model-swap-current.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        {
            "checks": out["checks"],
            "before": out["before"],
            "steps": [
                {
                    "model": step["model"],
                    "stream": {k: step["stream"][k] for k in ("status", "elapsed", "content", "content_deltas", "reasoning", "finish", "done")},
                    "running": [
                        b["model"]
                        for b in step["gateway"].get("backends", [])
                        if b.get("status") == "running"
                    ],
                    "processes": step["processes"],
                }
                for step in out["steps"]
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)
