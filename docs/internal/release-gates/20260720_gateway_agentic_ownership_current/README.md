# Current-source gateway agentic and one-model ownership gate — 2026-07-20

Status: `VERIFIED-LIVE_SCOPED / BROADER_GATEWAY_MATRIX_PARTIAL`.

Cutoff: `9e8a2f67c0816887f27ec02de29e777e287ec0c5` on
`codex/postrelease-ui-drawers-20260720`, exercised on
`erics-m5-max.local` through the real Electron dev app and its gateway at
`127.0.0.1:8088`.

## Source trace

- `panel/src/main/api-gateway.ts` owns the persistent single-model setting,
  serialized backend replacement, health registry, and gateway adapters.
- `panel/src/main/sessions.ts` performs the stop-before-start session ownership
  transition when single-model mode is enabled.
- `vmlx_engine/server.py` owns the native Chat Completions, Responses,
  Anthropic Messages, and Ollama handlers and terminals.
- The current source was exercised with the exact MiniMax-M3 and DSV4 sessions
  already proved in `../20260720_dsv4_m3_current_typed_cache/`; no generic
  output rewriting or synthetic tool execution was introduced for this gate.

## Live Electron and API evidence

| Row | Verdict | Current live evidence |
|---|---|---|
| Current Electron gateway surface | PASS-LIVE scoped | The real API drawer visibly showed Gateway `1 model`, URL `localhost:8088`, LAN Off/local-only, and the Single Model toggle On. `gateway-current-before.png` and `gateway-current-after-swaps.png` preserve the UI. `gateway-health-final.json` records `single_model_mode:true` and only MiniMax-M3 running. |
| Stream/non-stream protocol parity | PASS-LIVE scoped | `m3-gateway-protocol-parity.json` records HTTP 200 for Chat, Responses, Anthropic, and Ollama in stream and non-stream forms. Every stream emitted 40 progressive content deltas, the native terminal, no reasoning while explicitly disabled, and the identical exact eight-line answer. Each non-stream answer was byte-identical to its stream counterpart. |
| Auto reasoning separation | PASS-LIVE transport / PARTIAL economy | `m3-gateway-reasoning.json` records 512 separate reasoning deltas plus 12 progressive visible deltas on all four protocols and exact `M3-GATEWAY-REASON-DONE VALUE=45`, followed by each native terminal. All routes consumed the entire 512-token reasoning budget, so reasoning economy/latency is not promoted. |
| OpenAI agentic tool/result continuation | PASS-LIVE scoped | `m3-gateway-openai-tools.json` records no-tool controls, one schema-valid `file_info(path=panel/package.json)` call, a real 5.2 KB tool result, and exact `M3-GATEWAY-TOOL-DONE SIZE=5.2 KB` continuations for both Chat and Responses. Streaming continuations were progressive and terminal-complete; non-stream controls also completed. |
| Anthropic/Ollama agentic continuation | PASS-LIVE scoped | `m3-gateway-anthropic-ollama-tools.json` records one exact native tool call on each protocol, zero premature visible content, then 29 separate reasoning deltas and 15 progressive content deltas after the real result. Both exact-finaled with no second tool call and their truthful native terminal. |
| One-model gateway auto-swap | PASS-LIVE scoped | `gateway-one-model-swap-current.json` records M3 -> DSV4 -> M3 requests through port 8088. Each request stopped the old backend, eagerly started only the requested backend, left exactly one `vmlx-engine serve` process, progressively exact-finaled, and updated gateway health. DSV4 took 55.87 s and M3 15.71 s including replacement load. |
| Qwen3.6 MXFP4-MTP four-protocol reasoning | PASS-LIVE scoped at `616b0f3c8` | The current API drawer visibly showed the Qwen backend as the only running model, LAN Off, and Single Model On. Chat, Responses, Anthropic, and Ollama each returned HTTP 200, 460-469 separate reasoning deltas, 12 progressive content deltas, exact `Q27-MTP-GATEWAY-REASON-DONE VALUE=95`, and the native terminal. Gateway health and the process snapshot still show one Qwen engine after all four sequential requests. |

## Focused validation

- Python protocol adapters: 92 passed in `python-adapter-tests.log`.
- Panel gateway/session selections: 87 passed, 3 skipped in
  `panel-gateway-tests.log`.
- Probe sources are retained beside the JSON so the request bodies, event
  classification, terminal checks, tool-result continuation, and process
  assertions are reproducible.

## Truth boundaries and retained work

- A stale installed-app process was separately listening on `127.0.0.1:8081`.
  This gate deliberately targeted the current dev Electron gateway at 8088;
  `gateway-listeners.txt` preserves that distinction. It does not claim that
  the older listener was upgraded or removed.
- LAN remained Off and local-only. Prior LAN/rollback evidence is not rerun by
  this row. Current LAN enable/rollback, port-conflict recovery, network-loss
  injection, concurrent-client soak, and signed-app repetition remain OPEN.
- This is one current MiniMax-M3 parser family plus DSV4 ownership swapping.
  Qwen3.6 MXFP4-MTP reasoning transport is now an additional live parser/model
  row, while its direct Electron post-video one-tool continuation is retained
  in `../20260720_qwen36_27b_mxfp4_mtp_video_current/`. Other parser/model
  families, media-bearing gateway tools, longer stochastic agentic use, and
  explicit cancellation/disconnect injection retain their existing status.
- The app-versus-PATH engine version-string mismatch remains OPEN and must be
  reconciled before the next release checkpoint.

No release-readiness claim is made by this scoped gate.
