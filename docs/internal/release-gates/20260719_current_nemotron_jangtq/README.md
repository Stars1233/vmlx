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
| NEMO-003 | Streaming `/v1/responses` media request bypassed the Omni dispatcher. | PASS for current image/audio/video routes | Pre-fix `48-image-attached.png` through `52-prepatch-media-fail-current.png`; source trace was `server.py` gating Omni on `not request.stream`. Patched Electron artifacts `65`/`70` plus `77`/`82` and timed API artifacts `73`/`74` plus `80`/`85` reached the native Omni path and recovered the image, audio, and video markers. | Do not extrapolate this Omni proof to other media/parser families. |
| NEMO-004 | Omni streaming was post-hoc chunking of a completed reply rather than decode-time streaming. | PASS for current image/audio/video Responses rails and image Chat rail | JANG commit `70b2a7f` adds native token callbacks and true token accounting. `73-raw-responses-image-auto-timed.sse` has 385 reasoning deltas beginning at 2.302 s, followed by 17 content deltas and `response.completed` at 7.023 s. `74-raw-chat-image-off-timed.sse` has progressive content deltas from 2.294 through 2.520 s, usage, stop, and `[DONE]`. Audio artifact `80` streams content from 0.399 through 0.608 s and completes at 0.619 s; video artifact `85` streams content from 2.898 through 3.120 s and completes at 3.131 s. | Cancellation/disconnect behavior remains separate. |
| NEMO-005 | Patched media stream aborted the entire Python process after generation. | SOURCE FIXED; LIVE MULTI-TURN SURVIVAL PASS; CANCELLATION PENDING | Electron log at `20:45:18.606`: `Fatal Python error: PyThreadState_Get ... GIL is released`; PID 10148 stopped. Source used a new daemon `threading.Thread` per stream plus `asyncio.to_thread(queue.get)`. Commit `71286ae2b` uses a persistent `vmlx-omni-model` executor and async queue. New PID 11177 survived image, audio, video, raw Responses, raw Chat, and a subsequent two-iteration tool turn. Health artifacts `67`, `72`, `79`, `84`, and `89` remain healthy. | Abort a live decode and complete a following text/tool turn before closing cancellation. |
| NEMO-006 | Auto reasoning at UI Max Tokens 256 ended reasoning-only before visible content. | EXPECTED LENGTH at 256; PASS after explicit saved 1024 override | Row 527 and row 530: image-grounded reasoning, 0 visible chars, exactly 256 completion tokens, finish reason `length`. The first 1024 attempt was an operator proof error: the dirty settings panel was closed without pressing Save, so 256 correctly remained. `63-max-1024-saved.png` plus DB row 173 prove the actual saved override. Row 533 then stopped naturally at 197 tokens with separate reasoning and visible content; thinking-off row `71` returned exact `vMLX` plus marker in 16 tokens. | Keep the failed attempt documented; no hidden global cap or forced synthesis workaround was added. |
| NEMO-007 | Same-media reuse, different-media salt isolation, and L2/restart behavior. | OPEN; post-media text/tool cache is not media-cache proof | `87-postmedia-tool-final.png`, `88-postmedia-tool-row.json`, and `89-health-after-postmedia-tool.json` prove a later text/tool turn used 256 `paged+ssm+disk+tq-native` tokens. Source trace of the dedicated Omni bridge still shows its own in-memory `OmniSession` cache and fresh media embedding extraction, outside the global media-salted paged/L2 route. | Image A cold/warm, image B distinct salt, restart/L2 restore, with direct Omni media-cache telemetry; do not count the text/tool hit as closure. |
| NEMO-008 | Advertised Omni audio and video paths. | PASS for the current fixtures and endpoints | Electron audio `76`-`79` transcribed `amber seven` and emitted `NEMO-AUDIO1-OFF-DONE`; raw Responses `80` streamed 20 output tokens and completed at 0.619 s. Electron video `81`-`84` read `BANANA8426` and emitted the exact marker; raw Responses `85` streamed 21 output tokens and completed at 3.131 s. Both turns had non-empty visible output and no warnings; PID 11177 survived. | Repeat only if the media bridge changes; same-media reuse/salt/L2 remains NEMO-007. |
| NEMO-009 | Cancellation/disconnect must stop safely and preserve the next request. | PASS for current Responses disconnect and recovery | `90-raw-responses-cancel-after-deltas.txt` closes the client after 24 real reasoning deltas at 8.548 s, not during idle prefill. PID artifact `91` and health artifacts `91`/`93` show PID 11177 survived. `95-raw-responses-media-after-cancel.sse` then streamed 37 content deltas, the exact screenshot-derived model name and recovery marker, and one completed event; `100`-`102` show two subsequent Electron turns with distinct reasoning and exact single visible markers. | Repeat if cancellation or executor ownership changes. |
| NEMO-010 | A text/tool turn after media must retain history, emit one schema-valid call, execute once, and finish visibly. | PASS for current Electron history | `87-postmedia-tool-final.png` shows the video result, one completed Info card, and exact `NEMO-POSTMEDIA-TOOL0-DONE SIZE=5.2 KB`. DB artifact `88` records one `file_info` call with `{"path":"panel/package.json"}`, the 5.2 KB result, non-empty final content, and no warnings. It used 256 `paged+ssm+disk+tq-native` cached tokens. | Do not extrapolate to a tool call inside the dedicated media request itself. |
| NEMO-011 | One post-cancel Electron turn duplicated its exact marker twice. | ISOLATED OUTPUT-FORMAT MISS; NOT REPRODUCED AS CACHE/STREAM DEFECT | Row/screenshot `92` records the duplicate. Fresh raw Responses `97`, raw Chat `98`, thinking-off Responses `99`, and fresh Electron turns `100`-`102` each emitted one marker. Deterministic mixed-history artifacts `103`/`104` produced the same one-marker output on cold, 317-token warm `paged+ssm+disk+tq-native`, and explicit `skip_prefix_cache` runs. The fresh Electron A/B reasoning strings differ verbatim, excluding stale reasoning replay in those turns. | Keep the miss visible; no output-deduplication shim was added. Reopen as an engine bug if it reproduces deterministically under matched history/seed. |

## Current root-cause change under test

The Stage-1 Omni session owns PyTorch/MPS encoder state and an MLX decoder.
Creating and destroying a raw thread per streaming request lets CPython 3.13
tear down the thread while native thread-local cleanup is still active. The
current candidate pins session construction, encoding, and decode to one
long-lived `ThreadPoolExecutor(max_workers=1)` owned by the dispatcher. Token
callbacks schedule events onto an `asyncio.Queue`; the request no longer calls
blocking `queue.get` through arbitrary default-executor threads.

## Release verdict

`PARTIAL_NO_RELEASE`. Current image/audio/video routing, decode-time streaming,
visible content, terminal events, repeated-process survival, cancellation
recovery, and one post-media tool continuation now have Electron and raw API
evidence. Media reuse/salt/L2, the broader model/protocol matrix, full suites,
packaging, and release truth remain open.

## Focused source tests for this checkpoint

- `tests/test_multimodal_routing.py` plus `tests/test_omni_multimodal.py`: 23 passed.
- Responses Omni adapter tests in `tests/test_engine_audit.py`: 2 passed.
- Combined focused checkpoint: 25 passed.
- `py_compile` and scoped `git diff --check`: passed.
- A full Python/panel suite has not been rerun after this checkpoint; that gate
  remains open.
