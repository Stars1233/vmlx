# Current Nemotron Omni JANGTQ gate (2026-07-19)

Branch under test: `reconcile/1.5.68`

Remote checkout: `/Users/eric/mlx/vllm-mlx`

Bundle: `/Volumes/EricsLLMDrive/dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK`

Electron profile/CDP: `/Users/eric/.vmlx-v1611-cachefix-dev`, `127.0.0.1:9335`

This ledger is append-oriented. A source trace is not a live pass. A live UI
observation is not an API pass. Rows remain PARTIAL until both required proof
surfaces have current artifacts.

## Issue and proof ledger

| ID | Issue / gate | Current status | Current evidence | Required next proof |
|---|---|---|---|---|
| NEMO-001 | Real Electron Start must load the bundle with JANGTQ (Hadamard/codebook MXTQ), not affine JANG or base MXFP routing. | PASS for this load | `01-server-list-before-start.png`, `02-health-after-ui-start.json`, `04-argv-ui-start.txt`, `06-electron-running.png` | Reconfirm after each engine restart used for a fix. |
| NEMO-002 | Text reasoning/content streaming, multi-turn history, tool continuation, warm cache, and restart behavior. | PASS for the recorded text/tool prompts only | UI/DB/cache artifacts `12` through `47`; raw SSE `23-raw-responses-timed.sse` and `24-raw-chat-timed.sse` | Do not extrapolate these text rows to media or other parser families. |
| NEMO-003 | Streaming `/v1/responses` media request bypassed the Omni dispatcher. | PASS for current image route | Pre-fix `48-image-attached.png` through `52-prepatch-media-fail-current.png`; source trace was `server.py` gating Omni on `not request.stream`. Patched Electron artifacts `65`/`70` and timed API artifacts `73`/`74` all reached the native Omni path and read `vMLX`. | Recheck other modalities; do not extrapolate image proof to audio/video. |
| NEMO-004 | Omni streaming was post-hoc chunking of a completed reply rather than decode-time streaming. | PASS for current image Responses + Chat rails | JANG commit `70b2a7f` adds native token callbacks and true token accounting. `73-raw-responses-image-auto-timed.sse` has 385 reasoning deltas beginning at 2.302 s, followed by 17 content deltas and `response.completed` at 7.023 s. `74-raw-chat-image-off-timed.sse` has progressive content deltas from 2.294 through 2.520 s, usage, stop, and `[DONE]`. | Repeat on audio/video and cancellation. |
| NEMO-005 | Patched media stream aborted the entire Python process after generation. | SOURCE FIXED; LIVE MULTI-TURN SURVIVAL PASS; CANCELLATION PENDING | Electron log at `20:45:18.606`: `Fatal Python error: PyThreadState_Get ... GIL is released`; PID 10148 stopped. Source used a new daemon `threading.Thread` per stream plus `asyncio.to_thread(queue.get)`. Candidate uses persistent `vmlx-omni-model` executor and async queue. New PID 11177 survived five media turns: capped Auto, completed Auto, completed Off, raw Responses, and raw Chat. Health artifacts `67`/`72` remain healthy. | Abort a live decode and complete a following text/tool turn before closing cancellation. |
| NEMO-006 | Auto reasoning at UI Max Tokens 256 ended reasoning-only before visible content. | EXPECTED LENGTH at 256; PASS after explicit saved 1024 override | Row 527 and row 530: image-grounded reasoning, 0 visible chars, exactly 256 completion tokens, finish reason `length`. The first 1024 attempt was an operator proof error: the dirty settings panel was closed without pressing Save, so 256 correctly remained. `63-max-1024-saved.png` plus DB row 173 prove the actual saved override. Row 533 then stopped naturally at 197 tokens with separate reasoning and visible content; thinking-off row `71` returned exact `vMLX` plus marker in 16 tokens. | Keep the failed attempt documented; no hidden global cap or forced synthesis workaround was added. |
| NEMO-007 | Same-media reuse, different-media salt isolation, and L2/restart behavior. | OPEN | No current post-patch live evidence. | Image A cold/warm, image B distinct salt, restart/L2 restore, then post-media text/tool turn; record health/cache and UI artifacts. |
| NEMO-008 | Advertised Omni audio and video paths. | OPEN | Bundle/source capability inspection only; no current live turn. | Real Electron attachment plus raw API stream for audio and video, with coherent non-empty visible output and terminal events. |
| NEMO-009 | Cancellation/disconnect must stop safely and preserve the next request. | OPEN | Cooperative callback/cancel source exists; no live disconnect proof. | Abort a live media decode, verify engine remains healthy, and complete a follow-up turn. |

## Current root-cause change under test

The Stage-1 Omni session owns PyTorch/MPS encoder state and an MLX decoder.
Creating and destroying a raw thread per streaming request lets CPython 3.13
tear down the thread while native thread-local cleanup is still active. The
current candidate pins session construction, encoding, and decode to one
long-lived `ThreadPoolExecutor(max_workers=1)` owned by the dispatcher. Token
callbacks schedule events onto an `asyncio.Queue`; the request no longer calls
blocking `queue.get` through arbitrary default-executor threads.

## Release verdict

`PARTIAL_NO_RELEASE`. Current image routing, decode-time streaming, visible
content, terminal events, and repeated-process survival now have Electron and
raw API evidence. Reuse/salt/L2, audio/video, cancellation recovery, post-media
tool continuation, the broader model/protocol matrix, full suites, packaging,
and release truth remain open.

## Focused source tests for this checkpoint

- `tests/test_omni_multimodal.py`: 15 passed.
- Responses Omni adapter tests in `tests/test_engine_audit.py`: 2 passed.
- `py_compile` and scoped `git diff --check`: passed.
- A full Python/panel suite has not been rerun after this checkpoint; that gate
  remains open.
