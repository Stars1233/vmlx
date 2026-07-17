# Qwen 3.6 27B MTP terminal-stream and cache proof

Status: `PASS-LIVE` for the shared terminal-dispatch repair, raw streamed Chat
multi-turn, raw streamed Responses, Electron two-iteration tool use, native MTP
depth 1/3 execution, q4 attention-KV plus native hybrid-state RAM reuse, and
process-restart Block Disk L2 restore. Status remains `PARTIAL` for long native
reasoning latency and broader sampled reliability; this row is not a release
claim.

## Artifact and source trace

- Artifact: `dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP` on port 8032. The real
  bundle name declares MTP, its index contains 23 MTP tensors, and `/health`
  reports the preserved one-head runtime active for text and VL with UI-selected
  effective depth 3.
- The runtime identifies `qwen3_5` hybrid state from the actual layer graph:
  16 attention-KV layers and 48 native companion layers. Stored-prefix
  quantization is TurboQuant q4 on attention KV only; companion state stays
  native and uses clean async prefill rederive or an independently restored SSM
  checkpoint.
- Commit `aa6a3d2ef` fixes the shared scheduler ordering defect. The async LLM
  engine calls `Scheduler.step(defer_finished_cleanup=True)`, puts terminal
  output into the collector, yields the event loop, and only then persists
  cache/TQ/SSM state on the model worker. The MLLM scheduler uses the same order
  through `_dispatch_outputs` and a locked worker cleanup. Direct synchronous
  `step()` callers retain the previous cleanup-before-return default.
- `tests/test_terminal_dispatch_before_cache_cleanup.py` behaviorally pins
  dispatch-before-cleanup for both schedulers and protects deferred finished
  streams from being aborted. The focused file passed 4/4. The affected focused
  suite passed 50/50; a wider 261-test cache/batching slice passed 260 with the
  already-retained unrelated source-string assertion
  `test_streaming_tool_detection_requires_request_tools` as its sole failure.

## Reproduced global terminal-stream defect and repair

- `q27mtp-tqfair1.json` is the pre-fix control. On the resident 4,628-token
  `paged+ssm` hit, the first eleven visible deltas arrived from 0.9990s through
  2.4198s, but the final delta was withheld until 10.6316s while synchronous
  terminal cache persistence ran. This is the UI/API freeze-then-batch symptom.
- `q27mtp-tqfair2.json` is the same class of long-prefix request after the
  shared scheduler repair. Cold visible deltas span 7.7420s-9.2557s; the
  resident hit restores 4,628 tokens and streams all twelve visible deltas from
  1.0256s-2.5516s. The former multi-second terminal gap is absent, and both
  answers are exact.
- This repair is selected by scheduler path, not model family, weight format,
  text/VL mode, parser, or API surface. It does not synthesize output, dedupe
  model text, alter sampling, or skip cache persistence.

## Raw Chat and Responses streaming

- `q27mtp-postdispatch1.json` is a post-fix curl/SSE probe. Chat turn 1 emitted
  334 reasoning and 10 visible-content deltas and exact-finaled. Turn 2 in the
  same conversation emitted 350 reasoning and 18 visible deltas, recalled the
  turn-one codeword, reused 52 `paged+ssm` tokens, and exact-finaled.
- The same probe's `/v1/responses` request emitted 321 reasoning and 11
  `response.output_text.delta` events. The assembled marker exactly matched
  `response.output_text.done`, followed by one `response.completed` terminal.
  Thus the repair is proven below Electron on both OpenAI streaming surfaces.
- `q27mtp-shared1.json` is the earlier independent raw pass and is retained for
  comparison. Its tools-off run exercised real depth-3 MTP with drafted depth
  counts `[7,7,7]`, accepted `[4,2,1]`, and no fallback reason.

## Electron agent loop and settings truth

- `q27mtp-electron-streamfix1.json` is retained as a failed test-setup row.
  Live inspection showed that this newly created chat had **Enable Built-in
  Coding Tools** unchecked and no working directory, so the request exposed no
  functions. The model narrated a simulated call; no parser/tool success is
  claimed from that row.
- Through the visible Chat Settings drawer, built-in tools were enabled, the
  working directory was set to `/Users/eric/mlx/vllm-mlx`, and the setting was
  saved. A fresh chat inherited it. `q27mtp-electron-streamfix2.json` then
  emitted 118 reasoning paints and 11 progressive content paints, executed
  exactly one real `file_info(vmlx_engine/scheduler.py)`, persisted one OpenAI
  function call and one matching result, and returned exact
  `Q27MTP-ELECTRON-STREAMFIX2-DONE` with no warning.
- `q27mtp-electron-streamfix2.png` visibly shows the current model, port, PID,
  reasoning panel, real Info tool card, exact final answer, and usage metrics.
  The persisted DB extract is `q27mtp-electron-streamfix2-db.json`.
- Earlier same-chat Electron turns in `q27mtp-electron-shared1*.json` called
  two distinct file paths exactly once and exact-finaled. Tool requests
  intentionally constrained native MTP to depth 1; the saved health row records
  11 drafted and 4 accepted tokens at D1.
- The settings screenshots record Prefix Cache on, 15% cache memory, required
  Paged Cache with 64-token blocks and 1,000-block capacity, Block Disk L2 on at
  10 GB, Auto stored cache quantization, the truthful
  `TQ4 attention KV + native hybrid state` label, and native MTP depth 3.

## Long-prefix RAM and process-restart L2 restore

- `q27mtp-tqfair2.json` improves matched first-content time from 7.7420s cold
  to 1.0256s after restoring 4,628 resident `paged+ssm` tokens (7.549x). Worker
  reconstruction took 0.500882s.
- A visible Electron Save & Restart preserved L2 and changed the engine PID.
  `q27mtp-tqfair2-l2.json` then restored the same 4,628 tokens as
  `paged+ssm+disk`, streamed twelve exact deltas from 0.9171s-2.4300s, and
  recorded 0.455832s reconstruction. The following RAM request restored
  `paged+ssm` in 0.8803s first-content / 2.3921s last-content.
- `q27mtp-health-after-streamfix-l2.json` records two scheduler hits / 9,256
  tokens saved, 292 native-TQ block hits including 73 disk hits, a real SSM disk
  restore, q4 applied only to the 16 attention slots, native companion policy
  for the other 48 layers, and zero unsafe KV-without-SSM reuse. The final
  short request genuinely ran MTP D3: 30 drafted tokens across `[10,10,10]`,
  accepted `[2,1,1]`, with no fallback.

## Release boundary

This closes the current shared terminal-stream ordering defect for both async
schedulers and closes this artifact's current raw Chat/Responses, Electron
tool-loop, RAM prefix, and restart-L2 rows. It does not close the full release:
Qwen's long reasoning rail remains slow, the broader sampled reliability and
cancel/soak matrix is open, and the full repository/panel/release suites have
not yet passed on the final aggregate head.

## Current shared two-tool continuation falsification

- Server commit `3d32b944b` was first proved on Bonsai, then independently
  falsified here on the official
  `dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP` artifact. It distinguishes terminal
  post-result requests from an explicitly requested/client-narrowed remaining
  Qwen tool instead of always injecting `Do not emit another <tool_call>`.
- A real `/usr/bin/curl -N` three-round Responses harness executed exactly one
  `file_info(panel/package.json)`, exactly one `run_command(pwd)`, then streamed
  exact `Q27-API-MULTI-CURRENT-DONE` over eight timed content deltas and one
  completed terminal with no warning (`q27-api-multi-current.json`).
- A fresh Electron chat executed the same two tools once each and visibly
  returned exact `Q27-UI-MULTI-CURRENT-DONE`. Persisted row 324 contains both
  real calls/results, no warning, 371 output tokens, 18.5 t/s, 1.02s TTFT, and
  25.5s total. The 3s/10s/final screenshots retain the visible progression.
- Identical API replay restored 388 and 206 tokens as `paged+ssm`. A visible
  Electron Stop/Start replaced PID 63193 with PID 63864 without clearing L2;
  the next replay restored the same boundaries as `paged+ssm+disk`, kept the
  exact tools/final, and streamed the final over eight timed deltas.
- Post-restart health records 11 native-TQ q4 disk hits, two SSM-disk hits,
  `disk_hit=true`, `reconstructed=true`, and `dequantized=true`. Auto TQ applies
  only to 16 attention-KV layers; 48 companion layers remain native. MTP is
  active with effective depth 3, but the short final used real draft/verify
  counters only at depth 1; this row is not presented as a new D3 speed proof.
- Electron's single-model swap stopped Bonsai before Qwen started; the saved
  process snapshot contains only the Qwen serve process. The full long-context,
  cancellation, and broader-sampler reliability gates remain `PARTIAL`.
