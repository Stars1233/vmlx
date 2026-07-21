#!/usr/bin/env python3
"""Live gateway proof for in-flight single-model displacement and swap-back."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8088"
QWEN = "dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP"
LAGUNA = "JANGQ-AI/Laguna-XS.2-JANGTQ"
DB = Path("/Users/eric/.vmlx-v1613-responsive-dev/chats.db")
PANEL = Path("/Users/eric/mlx/vllm-mlx-release-1.6.13/panel")
NODE = "/Users/eric/.local/node/bin/node"
DRIVER = "/private/tmp/uidrv-once.cjs"
OUT = Path("/private/tmp/q27-laguna-concurrent-swap-proof.json")


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def gateway_health() -> dict:
    response = requests.get(BASE + "/health", timeout=10)
    response.raise_for_status()
    return response.json()


def session_snapshot() -> list[dict]:
    with sqlite3.connect(DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            select id, model_name, status, port
            from sessions
            where model_name in (?, ?)
            order by model_name
            """,
            (QWEN, LAGUNA),
        ).fetchall()
    return [dict(row) for row in rows]


def process_snapshot() -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if "vmlx_engine.cli serve" in line or "vmlx-engine serve" in line
    ]


def ui_capture(path: str) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "NODE_PATH": str(PANEL / "node_modules"),
            "VMLINUX_CDP": "http://127.0.0.1:9335",
        }
    )
    click = subprocess.run(
        [NODE, DRIVER, "click", "Server"],
        cwd=PANEL,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    time.sleep(1)
    shot = subprocess.run(
        [NODE, DRIVER, "shot", path],
        cwd=PANEL,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    body = subprocess.run(
        [NODE, DRIVER, "text"],
        cwd=PANEL,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "path": path,
        "click_rc": click.returncode,
        "click_output": (click.stdout + click.stderr).strip(),
        "shot_rc": shot.returncode,
        "shot_output": (shot.stdout + shot.stderr).strip(),
        "body": body.stdout,
        "body_rc": body.returncode,
    }


def stream_chat(model: str, prompt: str, max_tokens: int, on_delta=None) -> dict:
    start = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
    }
    result = {
        "model": model,
        "prompt": prompt,
        "request": payload,
        "status": None,
        "content_deltas": [],
        "reasoning_deltas": [],
        "tool_calls": [],
        "errors": [],
        "terminals": [],
        "usage": [],
        "exception": None,
    }
    try:
        response = requests.post(
            BASE + "/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(20, 300),
        )
        result["status"] = response.status_code
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line or not line.startswith("data: "):
                continue
            encoded = line[6:]
            now = elapsed_ms(start)
            if encoded == "[DONE]":
                result["terminals"].append({"at_ms": now, "type": "DONE"})
                continue
            try:
                data = json.loads(encoded)
            except json.JSONDecodeError:
                result["errors"].append(
                    {"at_ms": now, "code": "invalid_json", "raw": encoded}
                )
                continue
            if data.get("error"):
                error = data["error"]
                result["errors"].append(
                    {
                        "at_ms": now,
                        "code": error.get("code"),
                        "message": error.get("message"),
                        "type": error.get("type"),
                    }
                )
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content:
                result["content_deltas"].append({"at_ms": now, "text": content})
                if on_delta:
                    on_delta(len(result["content_deltas"]), now, content)
            if reasoning:
                result["reasoning_deltas"].append({"at_ms": now, "text": reasoning})
            if delta.get("tool_calls"):
                result["tool_calls"].append({"at_ms": now, "value": delta["tool_calls"]})
            if choice.get("finish_reason"):
                result["terminals"].append(
                    {"at_ms": now, "type": choice["finish_reason"]}
                )
            if data.get("usage"):
                result["usage"].append(data["usage"])
    except Exception as exc:  # evidence should retain exact transport failure
        result["exception"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = elapsed_ms(start)
    result["visible"] = "".join(item["text"] for item in result["content_deltas"])
    result["reasoning"] = "".join(
        item["text"] for item in result["reasoning_deltas"]
    )
    result["content_delta_count"] = len(result["content_deltas"])
    result["reasoning_delta_count"] = len(result["reasoning_deltas"])
    result["first_content_ms"] = (
        result["content_deltas"][0]["at_ms"] if result["content_deltas"] else None
    )
    result["last_content_ms"] = (
        result["content_deltas"][-1]["at_ms"] if result["content_deltas"] else None
    )
    return result


trigger = threading.Event()
qwen_holder: dict[str, dict] = {}


def signal_after_three(count: int, _at_ms: float, _text: str) -> None:
    if count >= 3:
        trigger.set()


def qwen_long_worker() -> None:
    qwen_holder["stream"] = stream_chat(
        QWEN,
        "Output exactly 180 numbered lines. Each line must be SWAP-HOLD followed by its line number. Do not call tools.",
        700,
        on_delta=signal_after_three,
    )


proof = {
    "gateway": BASE,
    "qwen_model": QWEN,
    "laguna_model": LAGUNA,
    "health_before": gateway_health(),
    "sessions_before": session_snapshot(),
    "processes_before": process_snapshot(),
}

thread = threading.Thread(target=qwen_long_worker, name="qwen-long-stream", daemon=True)
thread.start()
proof["triggered_after_three"] = trigger.wait(timeout=30)
proof["health_at_trigger"] = gateway_health()

laguna_marker = "CONCURRENT-SWAP-LAGUNA-DONE"
proof["laguna_swap"] = stream_chat(
    LAGUNA,
    f"Reply exactly {laguna_marker} and nothing else. Do not call tools.",
    96,
)
thread.join(timeout=30)
proof["qwen_stream_thread_alive_after_laguna"] = thread.is_alive()
proof["displaced_qwen"] = qwen_holder.get("stream")
proof["health_after_laguna"] = gateway_health()
proof["sessions_after_laguna"] = session_snapshot()
proof["processes_after_laguna"] = process_snapshot()
proof["ui_after_laguna"] = ui_capture("/private/tmp/concurrent-swap-laguna.png")

qwen_marker = "CONCURRENT-SWAP-QWEN-RETURN-DONE"
proof["qwen_swapback"] = stream_chat(
    QWEN,
    f"Reply exactly {qwen_marker} and nothing else. Do not call tools.",
    96,
)
proof["health_after_qwen"] = gateway_health()
proof["sessions_after_qwen"] = session_snapshot()
proof["processes_after_qwen"] = process_snapshot()
proof["ui_after_qwen"] = ui_capture("/private/tmp/concurrent-swap-qwen-return.png")

displaced = proof.get("displaced_qwen") or {}
laguna = proof["laguna_swap"]
qwen_back = proof["qwen_swapback"]
proof["checks"] = {
    "triggered_after_three": proof["triggered_after_three"],
    "displaced_had_progress": displaced.get("content_delta_count", 0) >= 3,
    "displaced_native_error": any(
        error.get("code") == "backend_connection_closed"
        for error in displaced.get("errors", [])
    ),
    "displaced_no_done": not any(
        terminal.get("type") == "DONE" for terminal in displaced.get("terminals", [])
    ),
    "displaced_no_exception": displaced.get("exception") is None,
    "displaced_no_reasoning_or_tool": not displaced.get("reasoning")
    and not displaced.get("tool_calls"),
    "laguna_status_200": laguna.get("status") == 200,
    "laguna_exact": laguna.get("visible", "").strip() == laguna_marker,
    "laguna_progressive": laguna.get("content_delta_count", 0) > 1
    and laguna.get("last_content_ms", 0) > laguna.get("first_content_ms", 0),
    "laguna_terminal": any(
        terminal.get("type") == "DONE" for terminal in laguna.get("terminals", [])
    ),
    "laguna_no_reasoning_tool_or_error": not laguna.get("reasoning")
    and not laguna.get("tool_calls")
    and not laguna.get("errors")
    and laguna.get("exception") is None,
    "laguna_only_running": [
        item["model_name"]
        for item in proof["sessions_after_laguna"]
        if item["status"] in {"running", "loading", "standby"}
    ]
    == [LAGUNA],
    "laguna_one_engine": len(proof["processes_after_laguna"]) == 1,
    "qwen_status_200": qwen_back.get("status") == 200,
    "qwen_exact": qwen_back.get("visible", "").strip() == qwen_marker,
    "qwen_progressive": qwen_back.get("content_delta_count", 0) > 1
    and qwen_back.get("last_content_ms", 0) > qwen_back.get("first_content_ms", 0),
    "qwen_terminal": any(
        terminal.get("type") == "DONE" for terminal in qwen_back.get("terminals", [])
    ),
    "qwen_no_reasoning_tool_or_error": not qwen_back.get("reasoning")
    and not qwen_back.get("tool_calls")
    and not qwen_back.get("errors")
    and qwen_back.get("exception") is None,
    "qwen_only_running": [
        item["model_name"]
        for item in proof["sessions_after_qwen"]
        if item["status"] in {"running", "loading", "standby"}
    ]
    == [QWEN],
    "qwen_one_engine": len(proof["processes_after_qwen"]) == 1,
}

OUT.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
print(
    json.dumps(
        {
            "checks": proof["checks"],
            "qwen_loss": {
                "delta_count": displaced.get("content_delta_count"),
                "errors": displaced.get("errors"),
                "terminals": displaced.get("terminals"),
                "exception": displaced.get("exception"),
            },
            "laguna": {
                "visible": laguna.get("visible"),
                "delta_count": laguna.get("content_delta_count"),
                "elapsed_ms": laguna.get("elapsed_ms"),
                "terminals": laguna.get("terminals"),
            },
            "qwen_swapback": {
                "visible": qwen_back.get("visible"),
                "delta_count": qwen_back.get("content_delta_count"),
                "elapsed_ms": qwen_back.get("elapsed_ms"),
                "terminals": qwen_back.get("terminals"),
            },
        },
        indent=2,
        ensure_ascii=False,
    )
)
