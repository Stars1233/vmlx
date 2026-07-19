#!/usr/bin/env python3
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8014"
MODEL = "jangq-ai/MiniMax-M2.7-Small-JANGTQ"
TOOLS = [
    {
        "type": "function",
        "name": "file_info",
        "description": "Return file metadata for one path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }
]


def parse_stream(payload):
    start = time.perf_counter()
    response = requests.post(BASE + "/v1/responses", json=payload, stream=True, timeout=300)
    out = {
        "status": response.status_code,
        "events": [],
        "content_deltas": [],
        "reasoning_deltas": [],
        "argument_deltas": [],
        "function_items": {},
        "terminals": [],
        "response_id": None,
        "completed_response": None,
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
            out["events"].append({"at_ms": round((time.perf_counter() - start) * 1000, 2), "event": event_name})
            continue
        if not line.startswith("data: "):
            continue
        raw_data = line[6:]
        if raw_data == "[DONE]":
            out["terminals"].append("DONE")
            continue
        data = json.loads(raw_data)
        typ = data.get("type") or event_name or ""
        if typ == "response.created":
            out["response_id"] = (data.get("response") or {}).get("id")
        if typ == "response.output_text.delta" and data.get("delta"):
            out["content_deltas"].append(data["delta"])
        if "reason" in typ and typ.endswith(".delta") and data.get("delta"):
            out["reasoning_deltas"].append(data["delta"])
        if typ == "response.function_call_arguments.delta" and data.get("delta"):
            out["argument_deltas"].append(data["delta"])
        item = data.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            key = item.get("call_id") or item.get("id") or f"item-{len(out['function_items'])}"
            out["function_items"][key] = item
        if typ in {"response.completed", "response.incomplete", "response.failed", "response.cancelled"}:
            out["terminals"].append(typ)
            if isinstance(data.get("response"), dict):
                out["completed_response"] = data["response"]
                out["response_id"] = data["response"].get("id") or out["response_id"]
                for output_item in data["response"].get("output") or []:
                    if output_item.get("type") == "function_call":
                        key = output_item.get("call_id") or output_item.get("id") or f"output-{len(out['function_items'])}"
                        out["function_items"][key] = output_item
    out["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    out["visible"] = "".join(out["content_deltas"])
    out["reasoning"] = "".join(out["reasoning_deltas"])
    out["function_calls"] = list(out.pop("function_items").values())
    return out


round1_payload = {
    "model": MODEL,
    "input": (
        "[M27-RESP-TOOL-1] Call file_info exactly once with path panel/package.json. "
        "You must use the tool and must not answer from memory. Do not emit prose before the call."
    ),
    "store": True,
    "stream": True,
    "max_output_tokens": 512,
    "temperature": 0.0,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": TOOLS,
    "tool_choice": "required",
}
round1 = parse_stream(round1_payload)

call = round1["function_calls"][0] if round1["function_calls"] else {}
call_id = call.get("call_id") or call.get("id")
raw_arguments = call.get("arguments") or "{}"
if isinstance(raw_arguments, str):
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = None
else:
    arguments = raw_arguments

path = Path("panel/package.json")
size = path.stat().st_size
human_size = f"{size / 1024:.1f} KB"
tool_output = (
    "Path: panel/package.json\n"
    "Type: file\n"
    f"Size: {human_size}\n"
    f"Modified: {path.stat().st_mtime_ns}\n"
    f"Permissions: {oct(path.stat().st_mode & 0o777)[2:].zfill(4)}"
)

round2_payload = {
    "model": MODEL,
    "previous_response_id": round1.get("response_id"),
    "input": [
        {"type": "function_call_output", "call_id": call_id, "output": tool_output},
        {
            "role": "user",
            "content": (
                "The tool result is complete. Do not call another tool. Reply exactly "
                f"M27-RESP-TOOL-DONE SIZE={human_size} and nothing else."
            ),
        },
    ],
    "store": True,
    "stream": True,
    "max_output_tokens": 512,
    "temperature": 0.0,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": TOOLS,
    "tool_choice": "auto",
}
round2 = parse_stream(round2_payload) if call_id and round1.get("response_id") else {"status": None, "skipped": True}

expected = f"M27-RESP-TOOL-DONE SIZE={human_size}"
checks = {
    "round1_status_200": round1["status"] == 200,
    "round1_exactly_one_tool": len(round1["function_calls"]) == 1,
    "round1_tool_name": call.get("name") == "file_info",
    "round1_exact_arguments": arguments == {"path": "panel/package.json"},
    "round1_no_visible_prose": not round1["visible"].strip(),
    "round1_terminal": "response.completed" in round1["terminals"],
    "round1_response_id": bool(round1.get("response_id")),
    "round2_status_200": round2.get("status") == 200,
    "round2_no_tool_calls": not round2.get("function_calls"),
    "round2_visible_exact": round2.get("visible", "").strip() == expected,
    "round2_content_progressive": len(round2.get("content_deltas") or []) > 1,
    "round2_terminal": "response.completed" in (round2.get("terminals") or []),
    "previous_response_id_used": round2_payload["previous_response_id"] == round1.get("response_id"),
}
out = {
    "model": MODEL,
    "expected": expected,
    "tool_output": tool_output,
    "checks": checks,
    "round1_payload": round1_payload,
    "round1": round1,
    "round2_payload": round2_payload,
    "round2": round2,
}
Path("/tmp/m27-responses-tool-loop.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"checks": checks, "round1": {"status": round1["status"], "response_id": round1.get("response_id"), "function_calls": round1["function_calls"], "argument_delta_count": len(round1["argument_deltas"]), "visible": round1["visible"], "reasoning_chars": len(round1["reasoning"]), "terminals": round1["terminals"], "events": sorted({e["event"] for e in round1["events"]})}, "round2": {"status": round2.get("status"), "function_calls": round2.get("function_calls"), "content_delta_count": len(round2.get("content_deltas") or []), "visible": round2.get("visible"), "reasoning_chars": len(round2.get("reasoning") or ""), "terminals": round2.get("terminals"), "events": sorted({e["event"] for e in round2.get("events", [])})}}, indent=2, ensure_ascii=False))
