# vMLX / MLXStudio Production Release Blocker Matrix - 2026-06-06

## Release state

Release status: red.

Do not package, sign, notarize, tag, upload, update appcast/downloads, or call the build production-ready while any row below is red.

Current active worktree:

`/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Deprecated paths not allowed for this work:

- `/Users/eric/vmlx`
- `/Users/eric/vmlx/swift`
- ADLab/TB/RDMA transport work unless explicitly requested for that exact lane

## Required evidence for green

A family row is green only when the relevant model artifacts pass current-source and installed-app proof across:

- `/v1/chat/completions`
- `/v1/responses`
- Anthropic `/v1/messages` where supported
- Ollama chat/generate where supported
- streaming and non-streaming
- 3+ turn multiturn recall/persona/context
- tool required, tool auto, tool-result continuation, and loop stop
- JSON/XML structured output parseability and repair diagnostics
- code/syntax/whitespace-sensitive output
- max output tokens and max context behavior
- prefix cache hit/miss correctness
- paged cache where compatible
- architecture-specific native cache where required
- TurboQuant KV encode/decode/restore where valid
- L2 disk cache write, restart/restore, and hit
- cancellation, timeout, unload/reload, post-error recovery
- UI settings parity with CLI args, `/health`, `/v1/cache/stats`, capabilities, and launched process
- media/VL/audio/video rows if the runtime advertises those modalities

No source-only, load-only, health-only, or one-prompt text smoke may clear a broad row.

## Current family matrix

| Family / artifact lane | Current status | Proven current positives | Current blockers | Next proof/fix |
|---|---:|---|---|---|
| MiMo V2.5 JANG_2L | Red | Text `ACK`; paged cache hit `cached_tokens=67`; L2 block write; multiturn `blue cat`; native mixed full/SWA cache detected; generic flat TQ-KV skipped for rotating cache | Required XML tool call corrupts output; speed around 1-2 tok/s; long/system prompt quality not cleared; VL/audio/video unwired | Source-vs-quant first divergence or replacement artifact proof; then tool decode/template/runtime fix; then full cache+tool+media matrix |
| Qwen 3.6 35B MXFP8 MTP | Partial | Source and installed-app `gdn_sink` no-crash proof exists; native MTP READY D3; 160-token source/installed around 77 tok/s | Full multiturn/tool/cache/VL/UI matrix not complete | Run structured multiturn/tool/cache/L2/UI proof under installed app |
| Qwen 3.6 27B MXFP8/JANG_4M MTP | Partial | Native MTP READY D3; deterministic policy can clear decode floor for MXFP8; JANG_4M decode better than MXFP8 | MXFP8 stochastic defaults slow; PP rows low; TP4 route rank/speed evidence remains red; full UI/cache/tool matrix open | Prove deterministic policy in UI/session; PP root cause; multiturn/tool/L2 proof |
| Nemo / Nemotron Omni | Red | Some source rows exist in older matrix | Omni audio/video processor bridge, tool dialect, cache/media salt, UI proof incomplete | Build live Omni text/audio/video/tool/cache smoke |
| LFM / LFM2.5 | Partial | Source rows exist in older cross-family matrix | Full installed-app multiturn/tool/cache/UI proof incomplete | Run installed-app LFM matrix with loop stop and structured output |
| MiniMax / MiniMax-M2.7 / JANGTQ_K | Red | Public/local route drift narrowed; public DMG cancel route present | Reporter parity/root cause still open; JANGTQ/runtime/speed/tool/cancel recovery not fully cleared | Finish reporter provenance and live MiniMax model proof |
| DSV4 Flash / DeepSeek-V4-Flash-JANGTQ-K | Red | Native composite cache smoke passed; required tool row passed in focused smoke | Long-output/code/file-generation exactness red; thinking-closed rail corrupts identifiers; memory-gated UI rows open | Fix/diagnose exact-code rail and rerun long/code/file-generation proof |
| Step 3.7 Flash | Partial | Text-only guard prevents unsupported advertised-VLM crash; source/staged proof of media rejection and text recovery exists | Real VLM not implemented/proven; tool loops/raw XML and multiturn synthesis still open; installed/public parity depends on release | Live installed-app text/tool/multiturn matrix; keep VLM fail-closed until real support lands |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | Partial | JANG_4M source/installed speed around 46 tok/s; image small/reject text recovery proof exists | Full VL/audio/video/tools/cache/UI matrix incomplete; speed rows per quant not all cleared | Run per-quant media/tool/cache/UI matrix; verify image prefill guard on installed app |
| ZAYA/ZAYA-VL | Partial | ZAYA text pass; ZAYA-VL capability truth and some media/tool/cache rows pass | Reasoning/media/UI matrix needs current installed-app proof | Refresh installed-app ZAYA/ZAYA-VL live matrix |
| Hybrid SSM / nonstandard cache families | Red | Some native cache contracts exist | Async rederive/companion state/L2 restore not exhaustively proven per family | Add architecture-specific cache matrix rows and prove no plain-KV substitution |

## Cross-cutting blockers

### CACHE-001: architecture-specific cache matrix incomplete

Required:

- Standard full-KV: generic TurboQuant KV encode/decode/restore parity.
- MiMo: native mixed full/SWA rotating cache only; no flat generic TQ-KV.
- DSV4: native SWA+CSA/HCA composite cache only.
- ZAYA CCA: typed native CCA cache restore.
- Hybrid SSM: companion state plus async/rederive proof.
- MLLM/media: media-salted prefix/L2 keys and text-after-media recovery.

Current status: red.

### TOOL-001: required/auto tool calls and continuation incomplete across families

Required:

- `tool_choice=required` returns real OpenAI `tool_calls` or classified model/runtime failure.
- Tool result continuation stops and summarizes.
- No raw XML leaks, malformed JSON, repeated tool loops, or hidden reasoning leaks.

Current status: red because MiMo required tools corrupt and Step/LFM/MiniMax/Nemo matrix remains incomplete.

### MEDIA-001: VL/audio/video runtime incomplete

Required:

- Actual media tensors/masks routed through model-specific processor/forward code.
- Capabilities distinguish runtime-supported vs preserved-unwired sidecars.
- UI upload/recovery/settings proof.

Current status: red. MiMo media is preserved but unwired; Step3.7 VLM is guarded text-only; Gemma4 has partial image proof only.

### UI-001: real Electron UI matrix incomplete

Required:

- Installed app settings panel matches source capabilities.
- Cache/MTP/media/max-output/max-context controls launch correct process args.
- Chat UI recovers after media rejection, timeout, cancel, parser failure.
- Streaming and non-streaming parity.

Current status: red.

### RELEASE-001: signing/notarization/public update locked

Required:

- Current release manifest `prepackage_ready=true`.
- Current release manifest `release_ready=true`.
- Source proof green.
- Bundled Python parity green.
- Installed/staged app parity green.
- Real UI matrix green.
- Release notes credit GitHub `@Hornsan1`.

Current status: red. No release action allowed.

## Current next actions

1. Keep MiMo red and classify tool corruption with source-vs-quant or artifact replacement proof.
2. Add harness knobs for per-family cache/TQ/L2 diagnostics so A/B rows are reproducible instead of manual one-off commands.
3. Refresh cross-family smoke matrix after the harness can record runtime modalities and architecture-specific cache evidence.
4. Commit/push only proven source/test/docs/proof artifacts, excluding `.agents/`, local AGENTS override, vendor noise, binary cache dirs, and release outputs.

## 2026-06-06 MiMo KV-none harness update

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-kvnone-extra-args-20260606/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Finding:

- `--kv-cache-quantization none` is now reproducible through the smoke harness.
- MiMo text/cache/multiturn stay green with KV quantization disabled.
- Required tool call still corrupts into `<tool_call>` plus punctuation/fullwidth-comma garbage.
- This formally rules out q4 KV storage and generic TurboQuant KV as the root cause of the MiMo tool failure.

Matrix impact:

- `MiMo V2.5 JANG_2L` remains red.
- `CACHE-001` improves for MiMo text cache evidence but remains red for largest-context, restart/L2 restore, and cross-family rows.
- `TOOL-001` remains red.
