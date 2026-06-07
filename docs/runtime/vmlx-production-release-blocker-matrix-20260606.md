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
| MiMo V2.5 JANG_2L | Red | Current Python path returns text `ACK`; paged cache hit `cached_tokens=67`; L2 block write; multiturn `blue cat`; native mixed full/SWA cache detected; generic flat TQ-KV skipped for rotating cache; current source preserves tool metadata into MLLM decode; keep=0 SWA cache patch fixes required-tool cache decode; narrow live required-tool row returns `record_fact({"value":"blue-cat"})`; tool-result continuation row returns exact `STORED blue-cat` with no second tool call and no raw markup after MiMo template tool-argument normalization; strict JSON row parses exactly; exact code/whitespace row preserves indentation and punctuation; 64-word long-prefix MLLM row passes with `cached_tokens=435`, `cache_detail=paged`, and 7 block-disk writes after tight-memory drain plus live `RotatingKVCache` mixed-SWA detection; expanded no-media source gate now passes 8/8 rows with 16 L2 block writes / 797 L2 tokens | Speed remains about 1-2 tok/s in current MiMo source gates, far below target; reasoning output remains low quality/repetitive; broader auto-tool/adversarial loop-stop/full multi-turn tool matrix not cleared; VL/audio/video unwired; full UI/installed-app matrix incomplete; no local `jang_config.json` in current bundle | Run broader MiMo auto-tool/loop-stop/cache/L2/restart/largest-context/UI smoke; then fix speed/kernel path; then implement/prove media bridge or keep capabilities text-only |
| Qwen 3.6 35B MXFP8 MTP | Partial | Bundled-engine smoke passes text/cache, multiturn, reasoning, required tool, image, video, post-media text recovery; native MTP active D3; paged+SSM hit; block + SSM L2 evidence; deterministic long Responses row activates MTP D3 and writes block/SSM L2; no `gdn_sink` TypeError; saved deterministic required-tool request now passes with configured D3 available, request-local D1 cap logged, and real `function_call` returned; full deterministic long Responses/tool/cache gate passes strict tool-call, tool-evidence, cache-hit, no-loop, and no-raw-markup criteria; expanded no-media Chat Completions gate still proves text cache, multiturn, required tool, tool-result continuation, strict JSON, exact code, native MTP D3, paged+SSM cache, and block/SSM L2 without `gdn_sink` crash | Expanded no-media gate is still red because `reasoning_on` produced 986 hidden-reasoning chars, exhausted `max_tokens=256`, and returned empty visible content; Anthropic/Ollama, streaming parity, real Electron UI settings, largest-context cache, restart/L2 restore, cancellation/recovery, media rows, and installed-app parity incomplete | Diagnose Qwen35 thinking-visible finalization under native MTP/MoE before release; then run missing API/UI/restart/largest-context/media rows |
| Qwen 3.6 27B MXFP4/MXFP8/JANG_4M MTP | Partial | MXFP4-MTP live slice passes text/cache, multiturn, reasoning, required tool, image, video, post-media recovery; Responses text/tool, Anthropic, Ollama, and Chat streaming pass; restart/L2 restore hits paged+SSM+disk; deterministic Responses cancellation/recovery passes with native MTP active D2; paged+SSM and block+SSM L2 evidence; JANG_4M installed-app MTP A/B reaches about 50.65 tok/s and 1.70x over AR; expanded no-media Chat Completions gate passes MXFP4-MTP and MXFP8-MTP across text cache, multiturn, reasoning-on, required tool, tool-result continuation, strict JSON, exact code, native MTP, paged+SSM cache, and block/SSM L2 | JANG_4M expanded no-media parity not rerun in the 2026-06-07 Qwen36 MTP gate; MXFP8 deterministic policy/UI parity, largest-context cache, TP4 route rank/speed evidence, media rows, installed-app parity, and full release API matrix remain open | Run Qwen27 JANG_4M expanded gate, UI/largest-context/media rows, and verify MXFP8 deterministic policy in UI/session |
| Nemo / Nemotron Omni | Red | Some source rows exist in older matrix | Omni audio/video processor bridge, tool dialect, cache/media salt, UI proof incomplete | Build live Omni text/audio/video/tool/cache smoke |
| LFM / LFM2.5 | Red | Expanded installed-source no-media gate confirms all three local LFM2.5 variants still pass required tool and tool-result continuation; MXFP4 and MXFP8 pass strict JSON parsing; hybrid SSM/paged cache telemetry remains present | Expanded structured-output gate is red: JANG_2L wraps JSON in markdown fences and emits `def add(a, b:`; MXFP4/MXFP8 emit `print(add(2, 3)` missing the final `)`; all exact-code failures ended with `finish_reason=stop`, so this is not a max-token false positive; installed-app/UI/API/media/largest-context rows remain incomplete | Keep LFM release-red; add JSON repair diagnostics where appropriate, but do not claim exact-code/codegen green until exact syntax/whitespace passes without runtime fabrication |
| MiniMax / MiniMax-M2.7 / JANGTQ_K | Red | Public/local route drift narrowed; public DMG cancel route present | Reporter parity/root cause still open; JANGTQ/runtime/speed/tool/cancel recovery not fully cleared | Finish reporter provenance and live MiniMax model proof |
| DSV4 Flash / DeepSeek-V4-Flash-JANGTQ-K | Red | Native composite cache smoke passed; required tool row passed in focused smoke | Long-output/code/file-generation exactness red; thinking-closed rail corrupts identifiers; memory-gated UI rows open | Fix/diagnose exact-code rail and rerun long/code/file-generation proof |
| Step 3.7 Flash | Partial | Text-only guard prevents unsupported advertised-VLM crash; expanded installed-source no-media gate passes text cache, paged cache hit, multiturn recall, reasoning-on, required tool, tool-result continuation, strict JSON, and exact code/whitespace; native cache reports `step3p7` / `mixed_swa_kv_v1` / `step3p7_full_sliding_kv`; block L2 writes present | Real VLM not implemented/proven; metadata still advertises MTP next-token layers while bundle index has no `mtp.*` tensors; adversarial loop-stop, streaming, Responses/Anthropic/Ollama, installed-app UI, media rejection/recovery, restart/L2 restore, and public parity remain incomplete | Keep Step text lane partial-green but release-red; keep VLM fail-closed until real support lands; run missing API/UI/restart/L2/loop-stop rows |
| Gemma 4 12B MXFP4/MXFP8/JANG_4M | Red | Expanded no-media source gates prove all three 12B quants can serve tools and tool-result continuation; JANG_4M passes strict JSON, reaches about `49.4 tok/s`, and reports paged+mixed-SWA cache hit `cached_tokens=56`; MXFP4 reaches about `59.9 tok/s`; MXFP8 reaches about `39.1 tok/s`; all three report native `gemma4` / `mixed_swa_kv_v1` cache and block L2 writes | Expanded structured-output rows are red: all three emit an extra leading space before `print(...)` in exact code; MXFP4/MXFP8 also emit JSON value `" blue-cat"` with a leading space; MXFP4 text cache exact-ACK row echoes the prompt instead of `ACK`; full VL/audio/video/tools/cache/UI matrix incomplete | Keep Gemma4 release-red; run media/tool/cache/UI matrix after deciding whether exact-code/JSON value drift is model artifact or runtime sampling/template issue; verify image prefill guard and post-error recovery on installed app |
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

Current status: red. MiMo required tools and one exact tool-result continuation
row now pass in the current no-media source gate, but auto-tool behavior,
adversarial loop-stop behavior, broader raw-markup leakage, and
Step/LFM/MiniMax/Nemo parity remain incomplete.

### STRUCTURED-001: strict JSON/code/XML structured output incomplete

Required:

- Strict JSON parseability and schema/value equality where requested.
- JSON repair/diagnostics where the runtime or benchmark pipeline promises
  repair, without hiding raw invalid output.
- XML/tool markup validation where model families use XML-style tool dialects.
- Exact code, syntax, indentation, punctuation, and whitespace preservation.

Current status: red. The smoke harness now includes strict JSON and exact
code/whitespace rows. MiMo passes those no-media source rows, but the current
LFM2.5 installed source gate is red: JANG_2L fails fenced JSON plus exact-code
syntax, and MXFP4/MXFP8 fail exact-code syntax. Cross-family JSON/code/XML
parity, JSON repair integration, Responses/Anthropic/Ollama parity, streaming,
and installed-app UI proof remain incomplete.

Step 3.7 installed source no-media now also passes the strict JSON and exact
code/whitespace rows, but that does not clear Step VLM, API parity, UI, restart
L2, adversarial loop-stop, or release readiness.

Gemma4 12B no-media source gates are mixed: all three quants pass
tools/tool-result continuation and cache telemetry, but strict exact-code
formatting is red for every quant and MXFP4/MXFP8 also fail exact JSON value
equality.

## 2026-06-07 LFM2.5 expanded no-media structured-output gate

Artifact:

`build/current-all-local-model-smoke-lfm25-installed-json-code-tools-nomedia-20260607/summary.json`

Command scope:

- Models root: `/Users/eric/.mlxstudio/models`
- Filter: `LFM2.5-8B-A1B`
- Media disabled.
- Tools enabled.
- Current source `vmlx_engine.cli serve`.

Results:

- Overall status: `fail`, `failed=3`, `row_count=3`.
- `LFM2.5-8B-A1B-JANG_2L`: required tool passed, tool-result continuation
  passed, strict JSON failed because the model wrapped the correct object in
  markdown fences, exact code failed as `def add(a, b:`.
- `LFM2.5-8B-A1B-MXFP4`: required tool passed, tool-result continuation
  passed, strict JSON passed, exact code failed as `print(add(2, 3)`.
- `LFM2.5-8B-A1B-MXFP8`: required tool passed, tool-result continuation
  passed, strict JSON passed, exact code failed as `print(add(2, 3)`.
- All exact-code failures returned `finish_reason=stop`, not `length`.

Classification:

- Tools/tool-result continuation are not the current LFM failure in this gate.
- JANG_2L fenced JSON is repairable by a JSON-repair pipeline, but the strict
  JSON row intentionally stays red because raw output is not parseable.
- Exact-code failures are model/output quality blockers unless a real guided
  decoding or syntax-constrained mode is implemented and proven; silently
  repairing chat code output would be a fake fix for release claims.

## 2026-06-07 Step 3.7 expanded no-media structured-output gate

Artifact:

`build/current-all-local-model-smoke-step37-jang2l-json-code-tools-nomedia-20260607/summary.json`

Command scope:

- Models root: `/Users/eric/.mlxstudio/models`
- Filter: `Step-3.7-Flash-JANG_2L`
- Media disabled.
- Tools enabled.
- Current source `vmlx_engine.cli serve`.

Results:

- Overall status: `pass`, `failed=0`, `row_count=1`.
- Text cache repeat passed; second repeat reported `cached_tokens=61` with
  `cache_detail=paged`.
- Multiturn recall passed with visible `blue cat`.
- Reasoning-on passed with visible `FINAL=OK`.
- Required tool passed with real `record_fact({"value":"blue-cat"})` and
  `finish_reason=tool_calls`.
- Tool-result continuation passed with visible `STORED blue-cat` and no second
  tool call.
- Strict JSON passed with exact raw JSON
  `{"status":"ok","value":"blue-cat","count":3}`.
- Exact code/whitespace passed exactly:
  `def add(a, b):\n    return a + b\nprint(add(2, 3))`.
- Native cache reported `step3p7` / `mixed_swa_kv_v1` /
  `step3p7_full_sliding_kv`; block-disk L2 wrote `10` blocks / `472` tokens.

Remaining blockers:

- This is text/no-media only. It does not prove Step3p7 VLM.
- MTP metadata remains inconsistent: config expects next-token prediction
  layers, but the bundle index has no `mtp.*` tensors.
- It does not prove adversarial loop-stop, streaming, Responses / Anthropic /
  Ollama parity, installed-app UI, media rejection/recovery, restart/L2 restore,
  public release parity, signing, or notarization.

## 2026-06-07 Gemma4 12B expanded no-media structured-output gates

Artifacts:

- JANG_4M:
  `build/current-all-local-model-smoke-gemma4-12b-jang4m-json-code-tools-nomedia-20260607/summary.json`
- MXFP4/MXFP8:
  `build/current-all-local-model-smoke-gemma4-12b-mxfp-json-code-tools-nomedia-20260607/summary.json`

Command scope:

- Models root: `/Users/eric/models`
- Media disabled.
- Tools enabled.
- Current source `vmlx_engine.cli serve`.
- MXFP candidate folders skipped for the MXFP run.

Positive evidence:

- `gemma-4-12B-it-JANG_4M`: required tool passed, tool-result continuation
  passed, strict JSON passed, paged+mixed-SWA cache hit `cached_tokens=56`,
  block L2 wrote `9` blocks / `402` tokens, aggregate generation about
  `49.4 tok/s`.
- `gemma-4-12B-it-MXFP4`: required tool passed, tool-result continuation
  passed, paged+mixed-SWA cache hit `cached_tokens=56`, block L2 wrote `9`
  blocks / `402` tokens, aggregate generation about `59.9 tok/s`.
- `gemma-4-12B-it-MXFP8`: required tool passed, tool-result continuation
  passed, paged+mixed-SWA cache hit `cached_tokens=56`, block L2 wrote `9`
  blocks / `402` tokens, aggregate generation about `39.1 tok/s`.

Failures:

- `gemma-4-12B-it-JANG_4M`: exact code/whitespace failed by inserting one
  leading space before `print(add(2, 3))`; finish reason was `stop`.
- `gemma-4-12B-it-MXFP4`: exact cache `ACK` rows failed by echoing the stable
  prefix, strict JSON failed with `"value":" blue-cat"`, and exact code failed
  with the same leading-space-before-`print` drift.
- `gemma-4-12B-it-MXFP8`: strict JSON failed with `"value":" blue-cat"`, and
  exact code failed with the same leading-space-before-`print` drift.

Classification:

- Current Gemma4 12B no-media tools/cache/speed evidence is good enough to
  keep testing, but the family is not structured-output green.
- The exact code/JSON value drift is raw model output under current runtime and
  should not be silently repaired for release claims.
- These gates are no-media only and do not prove image/audio/video, media
  prefill/recovery, streaming, Responses / Anthropic / Ollama, installed-app UI,
  restart/L2 restore, signing, or notarization.

## 2026-06-07 Qwen 3.6 MTP expanded no-media structured-output gate

Artifact:

`build/current-all-local-model-smoke-qwen36-mtp-json-code-tools-nomedia-20260607/summary.json`

Command scope:

- Models root: `/Users/eric/models`
- Models:
  `Qwen3.6-27B-MXFP4-MTP`,
  `Qwen3.6-27B-MXFP8-MTP`,
  `Qwen3.6-35B-A3B-MXFP8-MTP`.
- Media disabled.
- Tools enabled.
- Current source `vmlx_engine.cli serve`.

Positive evidence:

- `Qwen3.6-27B-MXFP4-MTP`: passed all expanded no-media rows.
  Text cache repeat hit `cached_tokens=56` with `cache_detail=paged+ssm`;
  multiturn recall returned `blue cat`; reasoning-on returned `FINAL=OK`;
  required tool returned real `record_fact({"value":"blue-cat"})`;
  tool-result continuation returned exact `STORED blue-cat`;
  strict JSON and exact code/whitespace passed.
- `Qwen3.6-27B-MXFP8-MTP`: passed the same expanded no-media rows.
  Text cache repeat hit `cached_tokens=56` with `cache_detail=paged+ssm`;
  required tool and tool-result continuation were real and clean;
  strict JSON and exact code/whitespace passed.
- `Qwen3.6-35B-A3B-MXFP8-MTP`: passed text cache repeat, multiturn recall,
  required tool, tool-result continuation, strict JSON, and exact
  code/whitespace. Required tool returned real OpenAI tool call
  `record_fact({"value":"blue-cat"})`, and tool-result continuation returned
  exact `STORED blue-cat`.
- All three loaded as Qwen3.6 native-MTP VL artifacts with hybrid cache. The
  27B MXFP4 row reported native MTP `READY D2`; 27B MXFP8 and 35B MXFP8
  reported native MTP `READY D3`.
- All three used architecture-aware hybrid caching: attention KV quantization
  `q4` for prefix/paged/L2 boundaries while SSM/GatedDelta companion state
  stayed native full precision. All three wrote block cache files and enabled
  hybrid SSM companion L2.
- No `gdn_sink` TypeError, stream generator crash, or raw XML tool leak occurred
  in this gate.

Failure:

- `Qwen3.6-35B-A3B-MXFP8-MTP` failed only the `reasoning_on` row. The request
  returned HTTP 200 and produced hidden reasoning, but visible content was
  empty. Request-level validation recorded `reasoning_chars=986` with
  `completion_tokens=256`; server logs show `finish=length` for that native MTP
  reasoning request.

Classification:

- The old user-facing `gdn_sink` traceback is not reproduced in this current
  source gate. Current source handles this path for the tested Qwen 3.6 MTP
  artifacts.
- Qwen27 MXFP4/MXFP8 MTP are no-media structured-output/tool/cache green in
  this source gate, but still not release-green because UI, media, restart/L2,
  largest context, and installed-app/API parity remain incomplete.
- Qwen35 MXFP8 MTP is not release-green. The current blocker is thinking-mode
  visible-finalization under native MTP/MoE, not a crash or corrupt required
  tool path in this gate.
- Do not hide this by disabling thinking as a release fix. The runtime needs a
  real diagnosis of the thinking/template/parser/decode-loop interaction, or
  the model artifact metadata/defaults need correction if that is proven by
  source-vs-artifact comparison.

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

1. Keep MiMo red; current normal Python engine proof now shows text/cache/multiturn, narrow required tool, and 64-word long-prefix cache work, but speed, full tool-result/multiturn tool behavior, media, UI, and installed-app rows remain open.
2. Do not synthesize fake tool calls from `tool_choice=required`; if more tool failures appear, fix parser/template/decode compatibility or use a real constrained/guided XML decoder.
3. Enumerate all old/local MiMo copies before deleting anything. Delete bad past MiMo copies only after a replacement/current artifact is proven better by the same tool/cache/long-prompt/speed rows.
4. Add harness knobs for per-family cache/TQ/L2 diagnostics so A/B rows are reproducible instead of manual one-off commands.
5. Refresh cross-family smoke matrix after the harness can record runtime modalities and architecture-specific cache evidence.
6. Commit/push only proven source/test/docs/proof artifacts, excluding `.agents/`, local AGENTS override, vendor noise, binary cache dirs, and release outputs.

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

## 2026-06-06 MiMo direct XML continuation probe

Artifact:

`build/current-mimo-v25-tool-continuation-probe-20260606`

Command boundary:

- Installed model: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Runtime: SimpleEngine, no continuous batching, prefix cache disabled, `--kv-cache-quantization none`.
- API: `/v1/completions`, not Chat Completions tool parsing.
- Prompt: real MiMo chat template plus concrete `mimo_xml_function` fallback example.

Results:

- Base prompt generated repeated `<think>` plus punctuation/fullwidth-comma garbage.
- Prompt prefilled through `<tool_call>\n<function=record_fact>\n<parameter=value>\n` generated only `blue` and then punctuation/newline garbage, not `blue-cat` and not closing XML.
- Base probe speed: 48 completion tokens in 50.42s, about `1.0 tok/s`.
- Prefilled-function probe speed: 48 completion tokens in 29.52s, about `1.6 tok/s`.

Updated classification:

- MiMo tool failure is not parser-only and not caused solely by Chat `tool_choice=required`.
- It reproduces outside tool parsing and after the runtime supplies the concrete function/parameter prefix.
- Do not implement a fake runtime tool-call synthesis as a release fix.
- Valid paths remain either a corrected MiMo artifact that naturally emits the XML tool grammar, or a real constrained/guided XML tool decoder proven end-to-end without fabricating arguments.
- MiMo speed remains red until an optimized routed-expert decode path is implemented/proven; current generic path is around 1-2 tok/s on this 106GB bundle.

## 2026-06-06 MiMo remote runtime delta and compiled-router fallback

Remote checked:

`erics-m5-max2.local:/Users/eric/jang`

Findings:

- Local bundled `jang_tools.mimo_v2.mlx_model.py` SHA-256: `7b6a0a524b486907f5e2fd2c5a122c1dc58726be1d8d6c2c1304a9aafda57f7d`.
- Remote `~/jang/jang-tools/jang_tools/mimo_v2/mlx_model.py` SHA-256: `a33ec86fc409e83feccf049c74d4c19be813ec6f5d130a6a20bf19d713d9372d`.
- Remote runtime includes a compiled single-token MiMo router and JANGTQ weighted decode hooks missing from the local bundled package.
- Remote quant contract says the coherent canonical affine JANG_2L path measured about `1.974-2.640 generation tok/s` and explicitly is not a `30 tok/s` path on the local M5 Max.
- The faster `5.626 generation tok/s` fit candidate failed arithmetic and was rejected.

vMLX source change:

- vMLX MiMo registration now installs a fallback compiled decode router when bundled `jang_tools` is stale.
- The patch is a no-op when newer `jang_tools` already exposes `run_compiled_mimo_decode_router`.
- This preserves vMLX `inputs_embeds` compatibility and does not synthesize tool calls.

Live proof:

`build/current-mimo-v25-compiled-router-live-probe-20260606`

- Server log confirms `Installed vMLX fallback compiled MiMo-V2 decode router`.
- Direct XML continuation still generated `blue` plus punctuation/newline garbage instead of `blue-cat` and closing XML.
- First short request speed was 16 completion tokens in 30.67s, about `0.5 tok/s`; this does not clear speed.

Matrix impact:

- `MIMO-TOOL-001` remains red.
- `MIMO-SPEED-001` remains red.
- Runtime stale-package integration is partially fixed, but MiMo remains blocked for release.

## 2026-06-07 MiMo ChatML XML-tool fallback framing fix

Source change:

- `vmlx_engine/api/tool_calling.py` now keeps MiMo/XML-function fallback tool
  instructions inside the rendered ChatML system turn when the MiMo template
  drops synthetic fallback messages during re-render.
- `tests/test_tool_fallback_injection.py` now covers this regression.
- This is a prompt-framing fix only; it does not fabricate tool calls or
  synthesize arguments.

No-heavy proof:

- `py_compile` passed for the edited tool-calling source and fallback test.
- Focused parser/fallback tests passed: 12 passed.
- Render proof with the real local MiMo tokenizer showed the fallback
  instructions are inside the system turn and no longer prefix the ChatML
  conversation.

Live proof:

`build/current-all-local-model-smoke-mimo-v25-jang2l-after-chatml-tool-fallback-20260607`

- Current source venv.
- Installed model:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Continuous batching, paged cache, block-disk L2, native mixed full/SWA cache.
- `--kv-cache-quantization none`.
- No media/video/reasoning; tools included.

Live positives:

- `text_cache_repeat_1`: HTTP 200, `ACK`.
- `text_cache_repeat_2`: HTTP 200, `ACK`, `cached_tokens=67`,
  `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, `blue cat`.
- Block-disk L2 wrote 4 blocks / 141 tokens.
- Native cache remained `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`.

Live failure:

- `tool_required`: HTTP 400, no parsed tool calls.
- Raw preview remained `<tool_call>` followed by punctuation/fullwidth
  punctuation garbage, without `<function=record_fact>`.
- Tool failure generation speed remained about `1.7 tok/s`.

Matrix impact:

- `MIMO-TEMPLATE-001` for outside-ChatML fallback placement is fixed.
- `MIMO-TOOL-001` remains red.

## 2026-06-07 MiMo long-prefix MLLM cache and tight-memory fix

Source changes:

- `vmlx_engine/mllm_batch_generator.py` drains MLX allocator state before MLLM
  prefills and after batch finish when startup detects a tight Metal
  working-set configuration.
- `vmlx_engine/mllm_scheduler.py` detects live `RotatingKVCache` in extracted
  MLLM cache objects and routes those rows through the clean mixed-SWA
  prompt-boundary store path.
- `tests/test_mllm_scheduler_cache.py` pins both contracts.

Live proof:

`build/current-local-long-context-cache-mimo-v25-installed-64w-after-tight-memory-rotating-store-20260607`

Result:

- `status=pass`.
- First request returned `LONGCTX-OK`.
- First cleanup detected mixed-SWA from extracted `RotatingKVCache`.
- Paged/L2 store wrote 7 blocks / 435 tokens.
- Second request returned `LONGCTX-OK`.
- Second request reported `cached_tokens=435`, `cache_detail=paged`.
- No Metal OOM/server disconnect on the row that previously crashed.

Matrix impact:

- MiMo long-prefix cache row moves from red to narrow green for 64 words.
- `CACHE-001` improves for MiMo MLLM mixed-SWA + block L2, but remains red for
  restart/L2 restore, largest-context, UI parity, and other families.
- MiMo release status remains red because speed, full tool behavior, media, UI,
  and installed-app parity remain incomplete.

## 2026-06-07 MiMo required-tool prefix propagation proof

Source changes under test:

- Chat/Responses request paths now forward `tool_choice` into the engine kwargs.
- Batched MLLM requests now preserve `_vmlx_template_tools` and
  `_vmlx_tool_choice` through prefill into decode-time sampler construction.
- A MiMo-only required-tool XML prefix logits processor constrains the opening
  structure for known request tools, without inventing argument values or
  fabricating `tool_calls`.

Focused source proof:

- `py_compile` passed for the edited source/test files.
- Focused MiMo decode tests passed: `3 passed, 39 deselected`.

Live proof:

- `build/current-mimo-v25-required-tool-prefix-toolonly-pending-token-20260607`
  proves the prefix processor activates and raw output starts with
  `<tool_call>\n<function=record_fact>\n<parameter=value>`.
- The same live row remains a failure because the model then emits punctuation
  instead of `blue-cat</parameter></function></tool_call>`.
- `build/current-mimo-v25-required-tool-prefix-toolonly-auto-tool-20260607`
  proves `--enable-auto-tool-choice` does not fix the malformed generation.
- `build/current-mimo-v25-xml-classification-default-cache-20260607` proves
  simple text can be coherent, but exact XML copying and speed remain red.
- `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-prefix-preflight-20260607.json`
  proves current source-vs-quant classification is still blocked by missing
  endpoints, not missing model paths.
- `build/current-mimo-v25-required-tool-system-fallback-live-20260607`
  proves system-scope MiMo XML fallback still fails live required-tool E2E:
  structural prefix appears, but the argument/closing XML is punctuation
  garbage and speed remains about `1.2 tok/s`.
- `build/current-mimo-v25-required-tool-cache-vs-full-logits-20260607.json`
  proves the current MiMo failure is an incremental cache/decode divergence:
  full forward after `blue` ranks `-cat` first, while cached decode after
  `blue` ranks newline/punctuation and puts `-cat` far down.
- `build/current-mimo-v25-cache-decode-mask-ab-20260607.json` proves simply
  disabling cached decode masks does not fix the divergence.
- `build/current-mimo-v25-required-tool-cache-vs-full-logits-after-attn-cache-patch-20260607.json`
  records that a diagnostic attention cache patch did not fix the divergence;
  that patch was removed and is not a claimed fix.
- `build/current-mimo-v25-rotating-cache-keep-ab-20260607.json` proves
  `RotatingKVCache(keep=4)` is the source of the bad cached continuation and
  `keep=0` restores `blue -> -cat` parity.
- `build/current-mimo-v25-cache-vs-full-after-keep0-patch-20260607.json`
  proves the vMLX registration patch creates `keep=0` SWA caches and cached
  decode now ranks `-cat` after `blue` and `</` after `blue-cat`.
- `build/current-mimo-v25-required-tool-live-after-keep0-patch-20260607`
  proves the narrow live required-tool row now returns HTTP 200 with
  `record_fact({"value":"blue-cat"})`.

Matrix impact:

- `MIMO-TOOL-METADATA-001` is improved at source-test level.
- `MIMO-TOOL-PROMPT-SCOPE-001` is improved at source-test/render level.
- `MIMO-TOOL-001` has a narrow required-tool pass for `record_fact`; broader
  tool-result continuation and multi-turn tool matrix remain open.
- `MIMO-SPEED-001` remains red.
- `MIMO-CACHE-DECODE-001` is fixed for the required-tool continuation, but
  larger cache/L2/restart rows remain open.
- `MIMO-SOURCE-VS-QUANT-001` remains red until source and quant endpoints are
  intentionally running and prompt rows execute.
- `RELEASE-001` remains red.
  fabrication.

## 2026-06-07 MiMo source-vs-quant harness tool row and local cleanup

Source/test changes:

- `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py` now
  includes the release-blocking XML tool row:
  `xml_tool_required_record_fact`.
- The runner now records source/quant OpenAI `tool_calls` signatures and
  classifies tool divergence separately from visible-text divergence.
- MiMo current-audit and release-manifest proof pointers now use:
  `build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`.
- `tests/test_mimo_v2_source_vs_quant_probe.py` covers the new tool-row
  classification.

Validation:

- `py_compile` passed for the edited source-vs-quant runner, current-audit
  runner, release manifest, and tests.
- `tests/test_mimo_v2_source_vs_quant_probe.py`: 8 passed.

Current source-vs-quant preflight:

`build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json`

- `status=missing_prerequisites`.
- Max2 source path exists:
  `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`.
- Local quant path exists:
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Source endpoint `http://erics-m5-max2.local:8126` is not healthy.
- Quant endpoint `http://127.0.0.1:8897` is not currently running.

Current audit:

`build/current-mimo-v2-jang2l-current-audit-after-tool-row-source-quant-preflight-stale-clean-20260607.json`

- `status=open`.
- Stale MiMo HF remote-code cache was deleted:
  `/Users/eric/.cache/huggingface/modules/transformers_modules/MiMo_hyphen_V2_dot_5_hyphen_JANG_2L`.
- Audit now reports `stale_local_state_absent=true`.

Matrix impact:

- MiMo local stale-state blocker is cleared again.
- MiMo source-vs-quant classification is still missing because endpoints are
  not running.
- MiMo remains red for long-prompt coherence, tool protocol, speed, CB system
  prompt pressure, source-vs-quant first divergence, and media wiring.

## 2026-06-07 MiMo source-vs-quant default preflight refresh

Source/test change:

- `tests/cross_matrix/run_mimo_v2_source_vs_quant_first_divergence.py` now has
  current defaults for the known source and quant endpoints/paths:
  `http://erics-m5-max2.local:8126`,
  `http://127.0.0.1:8897`,
  `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`, and
  `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- This prevents future agents from creating null-path/null-endpoint preflight
  artifacts when they run the systematic MiMo checklist.

Validation:

- `py_compile` passed for the source-vs-quant runner and test.
- `tests/test_mimo_v2_source_vs_quant_probe.py`: 9 passed.

Fresh preflight:

`build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-refresh-defaults-after-qwen35-20260607.json`

- `status=missing_prerequisites`.
- Quant model path exists locally.
- Source model path exists on `erics-m5-max2.local`.
- Quant endpoint `http://127.0.0.1:8897` is not listening.
- Source endpoint `http://erics-m5-max2.local:8126` is not listening.
- Rows did not execute; this still cannot classify model upload/artifact issue
  versus runtime/decode-loop issue.

Current MiMo audit refresh:

`build/current-mimo-v2-jang2l-current-audit-refresh-after-qwen35-20260607.json`

- `status=open`.
- Artifact integrity remains clean.
- Stale local state remains absent.
- Blockers remain:
  `mimo_long_prompt_coherence_blocked`,
  `mimo_tool_protocol_blocked`,
  `mimo_decode_speed_below_release_target`,
  `mimo_cb_system_prompt_working_set_pressure_blocked`,
  `mimo_source_vs_quant_first_divergence_missing_or_failed`, and
  `mimo_vl_audio_video_unwired`.

## 2026-06-06 Qwen3.6 35B MXFP8-MTP bundled smoke proof

Artifact:

`build/current-all-local-model-smoke-qwen35-mxfp8-mtp-tools-media-20260606/JANGQ_Qwen3.6-35B-A3B-MXFP8-MTP/result.json`

Command boundary:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`.
- Runtime Python: `panel/bundled-python/python/bin/python3.12`.
- Served with MLLM route, continuous batching, paged cache, block-disk L2, `--ssm-state-cache-mb 1024`, and default thinking off.
- Source regression check also passed: `tests/test_engine_audit.py -k "qwen35_dense_mtp_patch"` with 2 tests passed.

Result:

- Overall row: `pass`, zero failures.
- Model type: `qwen3_5_moe`.
- `is_mllm=true`, supports video, thinking, tools, and MTP.
- Native MTP: `runtime_active=true`, `effective_depth=3`, source `vmlx_mtp_tuning.json:native_mtp.best_depth`, runtime scope `text+vl`, artifact tensors present (`mtp_tensor_count=31`).
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=56`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `937`.
- `tool_required`: HTTP 200, visible content empty, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value": "blue-cat"}`.
- `vl_blue_image`: HTTP 200, visible `blue`.
- `vl_blue_image_repeat`: HTTP 200, visible `blue`.
- `vl_red_image_changed`: HTTP 200, visible `Red`.
- `vl_blue_video`: HTTP 200, visible `Blue`.
- `text_no_media_after_image`: HTTP 200, visible `NONE`.
- `text_no_media_after_video`: HTTP 200, visible `NONE`.
- Logs show repeated `MLLM native MTP path activated ... depth=3`.
- Logs show a real acceptance row on multiturn: `accepted=3/3 (100.0%)`.
- Logs show reasoning row MTP throughput: `242 tokens in 2.98s (81.3 tok/s)`.
- Logs show tool row: `27 tokens in 0.71s (37.8 tok/s)`.
- No `gdn_sink` `TypeError` appears in the smoke log.

Cache / L2 / architecture evidence:

- Native cache: `qwen3_5_moe` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV is enabled only for hybrid attention KV with reason `hybrid_attention_kv_only`.
- Live attention TQ-KV is active for attention KV layers only; SSM companion state stays native full precision.
- Storage-boundary attention KV quantization is q4, group size 64, attention layers only.
- Block disk L2 after run: 10 blocks, 431 block tokens on disk, 3 disk hits.
- SSM companion L2 after run: 9 disk entries, 687 SSM tokens on disk, about 565 MB.
- Cache totals report 1118 total L2 tokens across block and SSM stores.
- Media requests correctly skip VLM prefix cache store because media embeddings are path-dependent.

Nuance / still open:

- This is a strong bundled-engine smoke for Qwen35 MXFP8-MTP, but it is still not full release clearance.
- Missing rows: `/v1/responses`, Anthropic `/v1/messages`, Ollama, streaming/non-streaming equivalence, installed Electron UI settings, largest-context cache, restart/L2 restore, cancellation/recovery, and 27B parity.
- MTP acceptance is prompt-dependent: some short rows finish from seed/init emits with no draft cycles, while multiturn/reasoning/tool rows do exercise MTP differently.
- Keep Qwen 3.6 family status `Partial` until the missing API/UI/largest-context/restart rows are current.

## 2026-06-07 Qwen3.6 35B MXFP8-MTP long Responses/tool/cache diagnostic

Artifacts:

- `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-20260607`
- `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607`

Harness change:

- `tests/cross_matrix/run_responses_long_tool_cache_gate.py` now accepts
  explicit `--temperature`, `--top-p`, `--top-k`, and
  `--repetition-penalty` request overrides.
- Defaults remain omitted so normal rows still use model/server metadata; Qwen
  MTP rows can now request deterministic sampling to avoid silently skipping
  native MTP.
- Focused no-heavy validation passed: `py_compile` clean and
  `tests/test_engine_audit.py -k responses_long_context_tool_cache_gate`
  passed 19 selected tests.

Stochastic row:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`.
- `/v1/responses`, 3 turns, `previous_response_id`, required tools on turns 1
  and 2, final no-tools/no-thinking turn.
- Cache hits were observed on turns 2 and 3: `cached_tokens=128` and `256`.
- No raw tool markup leak and no loop-like tail.
- Row failed strict grounding: turn 1 called `inspect_symbol` with nonexistent
  `vmlx_engine/engine.py`; turn 2 received real file-line tool output but did
  not cite an exact marker in the visible answer.
- Server log shows native MTP skipped because default stochastic sampling
  resolved to `temperature=1.0`, `top_p=0.95`, `top_k=20`.

Deterministic row:

- Same model and server settings, but request sampling used
  `temperature=0.0`, `top_p=1.0`, `top_k=0`, and `repetition_penalty=1.0`.
- Server log reports native MTP `READY D3` and
  `MLLM native MTP path activated for request=resp_f19cad9242db depth=3`.
- MTP ran for 165 cycles, accepted 97 of 189 drafted tokens, and adapted from
  D3 to D1 after low deeper-depth acceptance.
- Block-disk cache and SSM companion L2 were active; the log queued block-disk
  writes and stored inline-captured clean SSM state.
- The row failed before summary completion because `/v1/responses` returned
  HTTP 400: `tool_choice='required'` was set but the model produced no tool
  calls.
- No `gdn_sink` TypeError or stream-generator crash was observed.

Current classification:

- Qwen35 MTP runtime activation and hybrid cache are working in this row.
- Qwen35 deterministic MTP plus required-tool behavior is not cleared.
- Do not count the stochastic cache/tool row as MTP proof, because the server
  explicitly skipped MTP for that request.
- Do not count the deterministic row as tool proof, because it failed required
  tool generation.

## 2026-06-07 Qwen3.6 35B MXFP8-MTP tool-request native-MTP depth cap

Source change:

- `vmlx_engine/mllm_batch_generator.py` now selects native-MTP depth per
  request instead of using only the configured global depth at seed time.
- Tool-bearing MLLM requests are capped to D1 when the configured depth is
  greater than 1.
- Plain non-tool requests still keep the configured depth, including D3.
- The runtime logs the cap explicitly:
  `MLLM native MTP depth capped to D1 ... because tools are present`.

Reason:

- A deterministic Qwen35 required-tool request with native MTP disabled returned
  HTTP 200 and produced a real `function_call`.
- The same deterministic request with native MTP D1 returned HTTP 200 and
  produced a real `function_call`.
- The same deterministic request with native MTP D3 returned HTTP 400 because
  no tool call was produced, while logs showed D3 MTP activated and then adapted
  down only after generation was already off the tool-call path.

Boundary:

- This is not a parser repair, prompt rewrite, or synthetic tool-call fallback.
- It does not claim D2/D3 tool-bearing MTP is safe.
- It keeps native MTP enabled for tool requests at the proven-safe D1 boundary
  until deeper-depth tool equivalence is fixed and live-proven.

Validation:

- `py_compile` passed for `server.py`, `mllm_batch_generator.py`,
  `engine/batched.py`, `mllm_scheduler.py`, and the native-MTP test file.
- Focused native-MTP depth tests passed: 5 selected tests.
- Live saved deterministic request proof:
  `build/current-qwen35-mxfp8-mtp-required-tool-deterministic-tool-depth-cap-20260607`.
- Startup health still reports native MTP active with effective depth D3 from
  `vmlx_mtp_tuning.json:native_mtp.best_depth`.
- Server log records:
  `MLLM native MTP depth capped to D1 ... because tools are present
  (configured D3)`.
- Server log records:
  `MLLM native MTP path activated ... depth=1`.
- `/v1/responses` returned HTTP 200 with output types `message` and
  `function_call`.
- The returned function call was real API output:
  `inspect_symbol({"path":"vmlx_engine/server.py","symbol":"get_engine",
  "context_lines":10})`.
- No `gdn_sink` TypeError or stream-generator crash was observed.

Full deterministic long gate after cap:

- Artifact:
  `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-after-tool-depth-cap-20260607`.
- Overall result remains `FAIL`, but for the expected reason: strict
  tool-evidence grounding failed on turn 1.
- The prior runtime blocker is cleared in this row: no HTTP 400 from
  `tool_choice=required`.
- Tool calls were observed on each required tool turn.
- Post-first cache reuse passed strict mode: turn 2 `cached_tokens=128`, turn 3
  `cached_tokens=256`.
- Final turn had tools disabled and returned visible text.
- Acceptance flags true: `tool_call_each_required_turn`,
  `cache_reuse_each_turn_after_first`, `final_turn_visible_output`,
  `no_loop_like_tail`, and `no_tool_markup_leak`.
- Acceptance flag false: `tool_evidence_each_required_turn`.
- Native MTP health still reports configured effective depth D3, while every
  tool-context generation logs the request-local D1 cap and activates native
  MTP at depth 1.
- Scheduler telemetry shows native MTP D1 acceptance in the final row:
  `116/189` drafted tokens accepted, about `61.4%`.

Full deterministic long gate after stable tool-evidence fallback:

- Source/test change:
  `tests/cross_matrix/run_responses_long_tool_cache_gate.py` now returns a real
  fallback file:line marker when a tool inspects a readable file/tree but finds
  no symbol or pattern match.
- This keeps strict grounding honest: the final answer must still cite a marker
  from actual tool output, but no-match inspections of real files no longer
  fail only because the tool returned markerless prose.
- Focused validation passed: `py_compile` clean and
  `tests/test_engine_audit.py -k responses_long_context_tool_cache_gate`
  passed 20 selected tests.
- Live artifact:
  `build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-evidence-marker-fallback-20260607`.
- Overall result: `PASS`.
- Acceptance flags true: `tool_call_each_required_turn`,
  `tool_evidence_each_required_turn`, `cache_reuse_each_turn_after_first`,
  `final_turn_visible_output`, `no_loop_like_tail`, and `no_tool_markup_leak`.
- Turn 1: required tool call `grep_repo`, exact evidence marker
  `vmlx_engine/server.py:1`.
- Turn 2: required tool call `inspect_symbol`, exact evidence marker
  `vmlx_engine/mllm_batch_generator.py:51`.
- Turn 3: tools disabled, visible synthesis returned, `cached_tokens=256`.
- Strict cache hits after first turn passed: turn 2 `cached_tokens=128`, turn 3
  `cached_tokens=256`.
- Native MTP health still reports configured D3, while every tool-context
  generation logs the request-local D1 cap and activates native MTP depth 1.

## 2026-06-06 Qwen3.6 27B MTP live slice and installed-app speed proof

Live slice artifact:

`build/current-all-local-model-smoke-live-slice-tools-media-continuation-20260606/JANGQ_Qwen3.6-27B-MXFP4-MTP/result.json`

Speed A/B artifact:

`build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json`

Live slice result:

- Overall row: `pass`, zero failures.
- Model: `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
- Model type: `qwen3_5`.
- `is_mllm=true`, supports video, thinking, tools, and MTP.
- Native MTP: `runtime_active=true`, `effective_depth=2`, source `vmlx_mtp_tuning.json:native_mtp.best_depth`, runtime scope `text+vl`, artifact tensors present (`mtp_tensor_count=23`).
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=56`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `829`.
- `tool_required`: HTTP 200, visible content empty, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value": "blue-cat"}`.
- `vl_blue_image`: HTTP 200, visible `blue`.
- `vl_blue_image_repeat`: HTTP 200, visible `blue`.
- `vl_red_image_changed`: HTTP 200, visible `red`.
- `vl_blue_video`: HTTP 200, visible `Blue.`.
- `text_no_media_after_image`: HTTP 200, visible `NONE`.
- `text_no_media_after_video`: HTTP 200, visible `NONE`.
- Logs show native MTP path activated at depth 2 and real acceptance rows including `accepted=2/2 (100.0%)`, reasoning `accepted=101/139 (72.7%)`, and tool `accepted=16/22 (72.7%)`.
- Logs show no `gdn_sink` `TypeError`.

Cache / L2 / architecture evidence:

- Native cache: `qwen3_5` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV is enabled only for hybrid attention KV with reason `hybrid_attention_kv_only`.
- Live attention TQ-KV is active for attention KV layers only; SSM companion state stays native full precision.
- Storage-boundary attention KV quantization is q4, group size 64, attention layers only.
- Block disk L2 after run: 10 blocks, 431 block tokens on disk, 3 disk hits.
- SSM companion L2 after run: 9 disk entries, 687 SSM tokens on disk, about 1321 MB.
- Cache totals report 1118 total L2 tokens across block and SSM stores.
- Media requests correctly skip VLM prefix cache store because media embeddings are path-dependent.

Installed-app native MTP speed A/B:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP`.
- Baseline AR row: 320 completion tokens at about `30.62 generation tok/s`.
- Native MTP row: 320 completion tokens, wall speed about `50.65 tok/s`, scheduler generation about `53.17 tok/s`.
- Speedup vs baseline: `1.7046857277044767`.
- Native MTP totals: 102 cycles, 218 accepted tokens, 228 drafted tokens, acceptance rate about `95.6%`.
- Depth acceptance: d1 about `99.0%`, d2 about `95.1%`, d3 about `83.3%`.

Nuance / still open:

- This strengthens Qwen27 MXFP4 and JANG_4M evidence, but the family remains `Partial`.
- Missing rows: MXFP8 deterministic policy through UI/session, `/v1/responses`, Anthropic `/v1/messages`, Ollama, streaming/non-streaming equivalence, real Electron UI settings, largest-context cache, cancellation/recovery, and TP4 route rank/speed evidence.
- The speed A/B was cache-off and text-only; it does not by itself clear media/cache/UI release rows.

## 2026-06-07 Qwen3.6 27B MXFP4-MTP API parity probe

Artifact:

`build/current-qwen27-mxfp4-mtp-api-parity-20260607`

Command boundary:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
- Runtime: source venv `vmlx serve`.
- Served with MLLM route, continuous batching, paged cache, block-disk L2, `--ssm-state-cache-mb 1024`, default thinking off, and `--is-mllm`.
- This was not a packaging/signing/notarization run.

Results:

- `/v1/responses` text: HTTP 200, `output_text=ACK`.
- `/v1/responses` `tool_choice=required`: HTTP 200, output item `type=function_call`, `name=record_fact`, arguments `{"value": "blue-cat"}`, `output_text=null`.
- Anthropic `/v1/messages`: HTTP 200, text `ACK`, stop reason `end_turn`.
- Ollama `/api/chat`: HTTP 200, assistant content `ACK`, done reason `stop`.
- Chat Completions streaming SSE: non-empty stream, contains `ACK`.
- Server log shows `/v1/responses` native MTP path activated at depth 2 for both text and required-tool rows.
- Required-tool Responses row exercised MTP with `accepted=16/22 (72.7%)`.
- No `gdn_sink` `TypeError` or server error was observed.

Cache / L2 / architecture evidence:

- Health after probe: native MTP active, effective depth `2`, runtime scope `text+vl`.
- Scheduler cache hit requests: 4, cache hit tokens: 36, `cache_detail=paged+ssm`.
- Block disk L2 after probe: 4 blocks, 149 block tokens on disk, 12 disk hits.
- SSM companion L2 after probe: 3 disk entries, 277 SSM tokens on disk.
- Cache totals report 426 total L2 tokens across block and SSM stores.
- Native cache remains `qwen3_5` / `hybrid_ssm_v1` / `hybrid_ssm_typed` with attention KV, SSM companion state, and async rederive.
- Live attention TurboQuant KV remains attention-layers-only; SSM companion state remains native full precision.

Nuance / still open:

- This clears current Qwen27 MXFP4-MTP non-Chat API parity for short text/tool rows, not the whole family.
- Missing Qwen27 rows remain: real Electron UI settings, largest-context cache hit, MXFP8 deterministic policy through UI/session, and TP4 route rank/speed evidence.

## 2026-06-07 Qwen3.6 27B MXFP4-MTP restart/L2 restore proof

Artifact:

`build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260607`

Command boundary:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
- Runtime: source venv `vmlx serve`.
- Two separate server processes used the same block cache directory:
  `shared_block_cache`.
- Served with MLLM route, continuous batching, paged cache, block-disk L2,
  `--ssm-state-cache-mb 1024`, default thinking off, and `--is-mllm`.
- This was not a packaging/signing/notarization run.

Phase 1 result:

- Text request returned visible `ACK`.
- Usage reported `prompt_tokens=64`, `completion_tokens=2`, and no cache hit.
- Block disk cache wrote 1 block and 63 block tokens.
- SSM companion disk wrote 1 entry and 63 SSM tokens.
- Native MTP was active at depth 2.
- No server errors were observed.

Phase 2 fresh-process restore result:

- Text request returned visible `ACK`.
- Usage reported `prompt_tokens_details.cached_tokens=63`.
- Usage reported `cache_detail=paged+ssm+disk`.
- Scheduler reported `cache_hit_tokens=63` and
  `cache_hit_tokens_by_detail={"paged+ssm+disk":63}`.
- `last_cache_execution.disk_hit=true`.
- Disk reconstruction and dequantization succeeded.
- Reconstruction time was about `0.040493s`; dequantization time was about
  `0.000007s`.
- Block disk cache reported 1 disk hit, 1 block, 63 tokens on disk, and no new
  disk writes in phase 2.
- SSM companion disk reported 1 hit, 1 entry, 63 tokens on disk, and hit rate
  `1.0`.

Cache / L2 / architecture evidence:

- Native cache remains `qwen3_5` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components remain attention KV, SSM companion state, and async rederive.
- Generic TurboQuant KV remains attention-KV-only for the hybrid attention
  layers; SSM companion state remains native.
- Storage-boundary attention KV quantization remains q4, group size 64,
  attention layers only.
- Phase 2 server log includes `VLM HYBRID cache HIT ... 63 cached (KV+SSM)`.
- Phase 2 server log shows native MTP active at depth 2.
- No `gdn_sink` `TypeError` or server error was observed.

Matrix impact:

- This clears the current Qwen27 MXFP4-MTP restart/L2 restore row for the short
  text path with hybrid attention KV plus SSM companion state.
- Qwen27 remains `Partial`, not green, because real Electron UI settings,
  largest-context cache, MXFP8 deterministic policy/session parity, TP4 route
  rank/speed, and full media/UI release rows remain open.
- Responses streaming and tool-result continuation were not covered in this probe.

## 2026-06-07 Qwen3.6 27B MXFP4-MTP deterministic Responses cancel/recovery proof

Artifact:

`build/current-qwen27-mxfp4-mtp-responses-cancel-mtp-deterministic-20260607.json`

Harness change:

- `tests/cross_matrix/run_issue179_responses_cancel_probe.py` now accepts
  per-family parser and sampling knobs while preserving the original MiniMax
  defaults.
- Focused no-heavy validation passed: `py_compile` clean and
  `tests/test_issue179_responses_cancel_probe.py` passed 11 tests.

Live proof:

- Model: `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
- Request used `/v1/responses`, streaming, `max_output_tokens=1024`,
  `temperature=0.0`, `top_p=1.0`, `top_k=0`, `--tool-call-parser qwen`,
  and `--reasoning-parser qwen3`.
- Server log reports native MTP `READY D2` and
  `MLLM native MTP path activated for request=resp_76a19f72835c depth=2`.
- Cancel route returned HTTP 200 with response id `resp_76a19f72835c`.
- No `gdn_sink` TypeError, stream generator error, or bad-text capture was
  observed in this row.

Cache / L2 / architecture evidence:

- Health before the request reports native cache family `qwen3_5`, schema
  `hybrid_ssm_v1`, cache type `hybrid_ssm_typed`, prefix and paged cache
  enabled, block-disk L2 enabled, generic TurboQuant KV enabled only for
  attention KV layers, and native full-precision SSM companion state.
- Server log reports VLM block disk cache and SSM companion L2 enabled, paged
  cache enabled, and q4 attention-KV storage boundary active.

Matrix impact:

- This clears the current Qwen27 MXFP4-MTP deterministic Responses
  cancel/recovery row.
- Qwen27 remains `Partial`, not green, because real Electron UI settings,
  largest-context cache, MXFP8 deterministic policy/session, TP4 route
  rank/speed evidence, and full media/UI release rows remain open.

## 2026-06-06 LFM2.5 MXFP4 focused source smoke

Artifact:

`build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json`

Result:

- Overall row: `pass`.
- Model type: `lfm2_moe`.
- Cache family: `hybrid_ssm`.
- Native cache: `lfm2` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=64`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Block disk L2 after tool row: 6 blocks on disk, 265 block tokens, 265 SSM companion tokens, 3 disk hits.

Nuance / still open:

- Exact visible output is correct, but short exact prompts report high `completion_tokens` counts (`208`, `380`, `252`). This needs a follow-up parser/stop-token/accounting check before calling LFM fully clean.
- This is source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 Nemotron Omni MXFP4 focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json`

Initial result:

- Text/cache/multiturn/tool passed.
- `reasoning_on` failed because the default 256-token reasoning probe stopped with hidden reasoning only and empty visible output.

Harness fix:

- Nemotron/Omni reasoning probes now receive the same 512-token budget treatment as ZAYA while retaining strict visible `FINAL=OK` validation.

Passing artifact:

`build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-after-reasoning-budget-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json`

Result:

- Overall row: `pass`.
- Model type: `nemotron_h`.
- Cache family: `hybrid_ssm`.
- Native cache: `nemotron_h` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=57`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `1824`, completion tokens `488`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Logs show hybrid paged hit plus SSM clean companion rederive and next-fetch-ready storage.

Nuance / still open:

- Server log warns `nemotron_h: unresolved eos strings ['<|im_end|>']`; registry/tokenizer EOS mapping needs follow-up before full release clearance.
- This is source-runtime no-media proof only. Omni audio/video/image bridge, installed app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 LFM2.5 MXFP8 focused source smoke

Artifact:

`build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260606/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json`

Result:

- Overall row: `pass`.
- Model type: `lfm2_moe`.
- Cache family: `hybrid_ssm`.
- Native cache: `lfm2` / `hybrid_ssm_v1` / `hybrid_ssm_typed`.
- Components: `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Generic TurboQuant KV correctly disabled with reason `hybrid_ssm_state`.
- Attention KV storage quantization active: q4, group size 64, storage-boundary only.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=64`, `cache_detail=paged+ssm`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.
- Logs show hybrid paged hit plus SSM clean companion rederive and next-fetch-ready storage.

Nuance / still open:

- Same as MXFP4: visible output is exact, but short exact rows report high completion-token counts (`156`, `229`, `128`). Follow-up needed for parser/stop-token/accounting before full LFM clearance.
- This is source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, and UI settings remain open.

## 2026-06-06 Step3.7 Flash JANG_2L CRACK focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json`

Initial result:

- Overall row passed, but the smoke harness incorrectly classified the advertised-vision Step3.7 artifact as `is_mllm=true` and passed `--is-mllm`.
- Server correctly overrode the advertised VLM route to text-only, but future proof should not depend on that override.

Harness fix:

- Step3p7 advertised media is classified text-only for the smoke launcher.
- Step3p7 cache family is classified as mixed SWA/full KV (`swa_rotating` in harness terms), not hybrid SSM.

Passing artifact after harness fix:

`build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json`

Result:

- Overall row: `pass`.
- Served command did not include `--is-mllm`.
- Server loaded `BatchedEngine ... (mllm=False)`.
- Server log shows `tier=step3p7_advertised_vlm_text_only result=False`.
- Model type: `step3p7`.
- Runtime media: `text` only.
- Declared media: `vision`, `image`, `video` are declared but not runtime supported.
- Native cache: `step3p7` / `mixed_swa_kv_v1` / `step3p7_full_sliding_kv`.
- Storage-boundary KV quantization active: q4, group size 64.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=61`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `390`.
- `tool_required`: HTTP 200, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value":"blue-cat"}`.

Nuance / still open:

- This proves the current text-only guard and text/tool/cache route, not Step3.7 VLM.
- The bundle still advertises vision/image/video while runtime support is intentionally text-only. Model upload metadata should be clarified or vMLX-specific metadata supplied.
- Source-runtime no-media proof only. Installed-app parity, streaming, Responses/Anthropic/Ollama, largest-context cache, restart/L2 restore, real Step3.7 VLM, and UI settings remain open.

## 2026-06-06 Gemma4 12B JANG_4M focused source smoke

Initial artifact:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-20260606`

Initial result:

- Both local rows structurally passed:
- `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`
- `/Users/eric/models/dealignai/Gemma-4-12B-it-JANG_4M-CRACK`
- The `tool_required` probe returned a valid OpenAI `tool_calls[0].function.name=record_fact` with arguments `{"value": "blue-cat"}`.
- The same response also leaked visible content `thought`, which is not acceptable for a tool-only response.
- The harness did not previously reject this marker leak, so the initial pass was not release-clean proof.

Harness/server fix:

- `tool_required` validation now rejects visible content alongside tool calls as `tool_visible_text_leak`.
- Chat/Responses finalization now drops only known hidden-channel marker literals (`thought`, `analysis`) when native tool calls are present.
- The fix does not force a fake tool call, does not rewrite tool arguments, and does not hide arbitrary visible text. It only removes leaked channel marker text after native tool parsing has already produced tool calls.
- Gemma4/Gemma4 unified harness classification is `swa_rotating`, matching runtime mixed full/sliding-window KV, not hybrid SSM.

Failing artifact after stricter validation:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-tool-leak-check-20260606`

Failure:

- Both local rows failed only `tool_visible_text_leak`.
- Leaked visible content: `thought`.
- Native OpenAI tool call was still present and correctly parsed.

Passing artifact after server cleanup and cache-family fix:

`build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-cache-family-fix-20260606/JANGQ_gemma-4-12B-it-JANG_4M/result.json`

Result:

- Overall row: `pass`.
- Model type: `gemma4_unified`.
- Harness cache family: `swa_rotating`.
- Runtime media declares `text`, `vision`, `audio`, and `video` as runtime supported.
- Capabilities: tools supported, thinking supported, `tool_parser=gemma4`, `reasoning_parser=gemma4`.
- Native cache: `gemma4` / `mixed_swa_kv_v1` / `mixed_swa_kv`.
- Components: `full_attention_kv`, `sliding_window_kv`, `rotating_window_metadata`.
- Generic TurboQuant KV correctly disabled for this native mixed-SWA route.
- Storage-boundary KV quantization active: q4, group size 64, applies to full and sliding attention KV, preserving rotating-window metadata.
- Prefix/paged/block-disk L2 active.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`, completion tokens `6`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=56`, `cache_detail=paged+mixed_swa`, completion tokens `6`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`, reasoning chars `402`.
- `tool_required`: HTTP 200, visible content empty, OpenAI `tool_calls[0].function.name=record_fact`, arguments `{"value": "blue-cat"}`.
- Cache after run: 5 block-L2 files, 227 block tokens on disk, cache hit tokens `56`.
- Short smoke generation stats: `generation_tps=50.28720244969087`.

Nuance / still open:

- This proves the source-runtime no-media text/cache/reasoning/tool path for JANG_4M only.
- It does not yet prove MXFP4 or MXFP8 Gemma4 rows.
- It does not yet prove real image/audio/video quality, VLM image prefill guard behavior, media cache salting under live media requests, or video/audio processing.
- It does not yet prove installed-app parity, MLXStudio UI settings, max-output-token settings, streaming, Responses/Anthropic/Ollama parity, largest-context cache, restart/L2 restore, or notarized app behavior.
- Runtime reports media as supported, but that remains release-red until real VL/audio/video E2E probes pass with visible semantic proof.

## 2026-06-07 MiMo current normal-engine refresh after Qwen35 work

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-refresh-after-qwen35-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Finding:

- Current local model path is `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`.
- Current bundle has `model_type=mimo_v2`, full plus SWA rotating attention, media sidecars, and native XML tool grammar in the chat template.
- Current bundle does not have `jang_config.json`; runtime classification comes from `config.json`, tokenizer/template metadata, and vMLX registry/runtime detection.
- Normal Python MLLM path with continuous batching, paged prefix cache, block-disk L2, native mixed full/SWA cache, and no media/video proves text/cache/multiturn positives:
  - first repeat: HTTP 200, visible `ACK`;
  - second repeat: HTTP 200, visible `ACK`, `cached_tokens=67`, `cache_detail=paged`;
  - multiturn recall: HTTP 200, visible `blue cat`;
  - block-disk L2 wrote 4 blocks / 141 tokens;
  - native cache reports `mimo_v2` / `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`.
- Required tool row is still red:
  - request: `tool_choice=required`, tool `record_fact`, expected value `blue-cat`;
  - HTTP 400 after 96 generated tokens;
  - no parsed `tool_calls`;
  - raw preview starts `<tool_call>\n\n：，，，。...`;
  - server log explicitly reports `tool_choice='required' but model produced no tool calls`;
  - generation speed in the failing tool row is about `1.6 tok/s`.

Classification:

- This is not a cache-only failure: the same class had already reproduced with cache disabled, and the current run proves cache can work while tools fail.
- This is not missing schema injection: fallback/schema placement has been fixed and current template already contains native XML tool instructions.
- This is not release-cleared MiMo. The remaining blocker is model artifact versus runtime decode/tool-grammar compatibility, pending source-vs-quant first divergence or a real constrained/guided XML decoder.

## 2026-06-07 MiMo no-media source smoke after keep=0 cache fix

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-keep0-cache-fix-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

What changed:

- Current source now passes the focused MiMo no-media smoke with tools enabled.
- Runner summary is `status=pass`, `row_count=1`, `failed=0`.
- The server loaded `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L` as MLLM.
- Server log confirms both MiMo runtime patches: fallback compiled decode router and keep=0 SWA rotating cache.

Proof details:

- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=67`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `tool_required`: HTTP 200, parsed OpenAI `record_fact` tool call with `{"value": "blue-cat"}` and no visible text leak.
- Native cache reports `mimo_v2` / `mixed_swa_kv_v1` / `mimo_v2_asymmetric_swa`.
- Prefix, paged cache, and block-disk L2 are enabled.
- Block-disk L2 stores `173` tokens across `5` written blocks by the end of the smoke.

Still red:

- Generation speed is still about `1.76 tok/s` in final health and `0.1-1.7 tok/s` in request logs, far below the `40+ tok/s` target.
- `reasoning_on` did not fail the runner, but its visible text is repetitive/messy and is not release-quality reasoning proof.
- Tool-result continuation, auto-tool behavior, loop-stop, streaming tool parity, Anthropic/Ollama parity, largest-context cache, restart/L2 restore, cancellation cleanup, installed-app/UI settings parity, and MiMo media remain unproven.
- MiMo image/audio/video support remains unwired/unproven and must not be advertised as working.

Release boundary:

- This supersedes the earlier narrow statement that current MiMo required tools always corrupt under the normal source runner.
- It does not clear MiMo, vMLX, or MLXStudio for release.
- Do not sign, notarize, tag, publish, or update public downloads from this proof.

## 2026-06-07 MiMo rejected compiled-MoE speed attempt

Artifact:

`build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-compiled-moe-decode-20260607/JANGQ_MiMo-V2.5-JANG_2L/result.json`

Findings:

- A temporary source patch compiled the single-token MiMo MoE decode block.
- The patch activated and preserved correctness: the smoke still passed and `tool_required` still returned parsed `record_fact({"value": "blue-cat"})`.
- It did not improve speed: generation throughput was `1.7457 tok/s`, compared with `1.7586 tok/s` before.
- The temporary patch was removed. Do not carry it forward as a release fix.

Next speed work:

- Profile or replace the actual routed-expert `gather_qmm`/SwitchGLU decode path.
- Confirm whether a different MiMo JANG_K/non-TQ or prepacked expert artifact exists with the expected `40+ tok/s` target.
- Do not advertise MiMo speed as fixed from the keep=0 tool/cache repair.

## 2026-06-07 LFM2.5 installed no-media tools/cache smoke

Artifact:

`build/current-all-local-model-smoke-lfm25-installed-tools-nomedia-20260607/summary.json`

Scope:

- Current source `vmlx_engine.cli serve`.
- Installed local models under `/Users/eric/.mlxstudio/models`.
- No media.
- Required-tool probe enabled.
- Repeated prompt cache probe and multiturn recall probe included.

Rows:

- `LFM2.5-8B-A1B-JANG_2L`: pass.
- `LFM2.5-8B-A1B-MXFP4`: pass.
- `LFM2.5-8B-A1B-MXFP8`: pass.

Proof details:

- Summary: `status=pass`, `row_count=3`, `failed=0`.
- All three rows returned visible `ACK` for cache-repeat probes.
- All three rows returned visible `blue cat` for multiturn recall.
- All three rows returned parsed OpenAI `record_fact({"value": "blue-cat"})` for `tool_required` with no visible tool text leak.
- All three rows reported cache hit evidence on the second repeat: `cached_tokens=64`, `cache_detail=paged+ssm`.
- All three rows reported typed native cache family `lfm2` / `hybrid_ssm_v1` with components `attention_kv`, `ssm_companion_state`, and `async_rederive`.
- Generic TurboQuant KV is correctly disabled for LFM hybrid SSM state; attention KV storage-boundary q4 is active with native companion SSM state.
- Block-disk L2 is enabled and disk hits were reported in the run summary.

Still open:

- This is not media proof.
- This is not restart/L2 restore proof.
- This is not largest-context proof.
- This is not UI/settings or installed-app launch-arg proof.
- This is not Anthropic/Ollama/Responses/streaming parity proof.
- This is not structured JSON/XML benchmark repair proof.

## 2026-06-07 Step 3.7 Flash installed no-media tools/cache smoke

Artifact:

`build/current-all-local-model-smoke-step37-flash-installed-tools-nomedia-20260607/summary.json`

Scope:

- Current source `vmlx_engine.cli serve`.
- Installed local model `/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L`.
- No media.
- Required-tool probe enabled.
- Repeated prompt cache probe and multiturn recall probe included.

Proof details:

- Summary: `status=pass`, `row_count=1`, `failed=0`.
- `text_cache_repeat_1`: HTTP 200, visible `ACK`.
- `text_cache_repeat_2`: HTTP 200, visible `ACK`, `cached_tokens=61`, `cache_detail=paged`.
- `text_multiturn_recall`: HTTP 200, visible `blue cat`.
- `reasoning_on`: HTTP 200, visible `FINAL=OK`.
- `tool_required`: HTTP 200, parsed OpenAI `record_fact({"value": "blue-cat"})`.
- Native cache reports `step3p7` / `mixed_swa_kv_v1` / `step3p7_full_sliding_kv`.
- Prefix, paged cache, storage-boundary q4 KV, and block-disk L2 are enabled.
- Server log explicitly routes Step3.7 advertised VLM metadata to text-only:
  `step3p7_advertised_vlm_text_only`.

Important blockers:

- VLM remains not production-cleared for Step3.7. The successful row is text-only.
- MTP metadata is inconsistent for this bundle: config advertises `num_nextn_predict_layers=3`, but the bundle index has no `mtp.*` tensors.
- The run proves short no-media tools/cache/multiturn only, not full loop-stop, long context, streaming/API parity, media, UI, or restart/L2 restore.

## 2026-06-07 local restart/L2 restore gate

Runner:

`bench/local_restart_l2_gate.py`

Purpose:

- Launch each installed local model once with paged cache plus block-disk L2.
- Send a fixed short prompt and shut the server down.
- Launch a fresh server process against the same block-disk cache directory.
- Send the same prompt and require the second process to report positive cached
  tokens plus disk-backed restore evidence.
- Report cache-restore status separately from visible-output exactness so cache
  durability is not hidden by model instruction-following quality.

Runner fixes made during this gate:

- Preserve the venv Python executable instead of resolving it to the base
  interpreter; otherwise `-s` hid `uvicorn`.
- Skip auxiliary weighted config directories such as `audio_tokenizer` so media
  sidecars are not treated as serveable model roots.

Artifact:

`build/current-local-restart-l2-lfm-step-installed-cache-status-20260607/summary.json`

LFM / Step cache restore results:

- `LFM2.5-8B-A1B-JANG_2L`: cache restore pass, output exactness review.
- `LFM2.5-8B-A1B-MXFP4`: cache restore pass, output exactness pass.
- `LFM2.5-8B-A1B-MXFP8`: cache restore pass, output exactness review.
- `Step-3.7-Flash-JANG_2L`: cache restore pass, output exactness pass.

Evidence:

- LFM JANG_2L second process: `cached_tokens=35`, `cache_detail=paged+ssm+disk`,
  block-disk hits `1`, native cache `lfm2` / `hybrid_ssm_v1`.
- LFM MXFP4 second process: `cached_tokens=30`, `cache_detail=paged+ssm+disk`,
  block-disk hits `1`, native cache `lfm2` / `hybrid_ssm_v1`.
- LFM MXFP8 second process: `cached_tokens=30`, `cache_detail=paged+ssm+disk`,
  block-disk hits `1`, native cache `lfm2` / `hybrid_ssm_v1`.
- Step second process: `cached_tokens=25`, `cache_detail=paged+disk`,
  block-disk hits `1`, native cache `step3p7` / `mixed_swa_kv_v1`.

Output-quality notes:

- LFM JANG_2L and MXFP8 restored cache correctly, but did not follow the exact
  visible `ACK` formatting probe.
- LFM MXFP4 included `ACK` but still produced explanatory text; treat as
  output-review for stricter exact-format tasks even though the runner's
  substring rail passes.
- Step returned exact `ACK` in both processes.

MiMo artifact:

`build/current-local-restart-l2-mimo-v25-installed-cache-status-weighted-only-20260607/summary.json`

MiMo cache restore result:

- `MiMo-V2.5-JANG_2L`: cache restore pass, output exactness review.
- Second process reported `cached_tokens=43`, `cache_detail=paged+disk`,
  block-disk hits `1`, and `last_cache_execution.disk_hit=true`.
- Native cache reports `mimo_v2` / `mixed_swa_kv_v1` /
  `mimo_v2_asymmetric_swa`; generic TurboQuant KV remains disabled and
  storage-boundary q4 is active for full/sliding KV while preserving rotating
  metadata.
- Server log confirms the MiMo compiled-router fallback and keep=0 rotating
  cache patch are active.

MiMo blockers exposed by restart/L2:

- Second-process cache reconstruction took about `19.43s`, so restart/L2 is
  functionally present but not performance-cleared.
- Visible output was not the requested `ACK`; first process emitted Chinese
  explanatory text and second process emitted punctuation.
- Generation remains far below the release target.

Release boundary:

- This clears short restart/L2 cache-restore evidence for the named installed
  local LFM, Step, and MiMo rows.
- It does not clear largest-context restore, UI-launched restore, media-salted
  restore, cancellation cleanup, sleep/wake, unload/reload, streaming/API
  parity, output exactness, or any release packaging/signing/notarization row.

## 2026-06-07 local long-context cache gate and mixed-SWA store fix

Runner:

`bench/local_long_context_cache_gate.py`

Purpose:

- Launch each installed local model with paged cache plus block-disk L2.
- Send the same long prompt twice in one process.
- Require the second request to report a large positive `cached_tokens` count.
- Report output exactness separately from cache correctness.

Initial failure:

`build/current-local-long-context-cache-lfm-step-installed-2048w-20260607/summary.json`

- 2048-word prompts were rejected with HTTP 413 for LFM and Step even with
  explicit `--max-prompt-tokens 8192`; this is not yet a largest-context pass.

Accepted long-context gate before the Step fix:

`build/current-local-long-context-cache-lfm-step-installed-512w-20260607/summary.json`

- LFM JANG_2L/MXFP4/MXFP8 cache hits passed at 512 words.
- Step returned correct `LONGCTX-OK` but reported `cached_tokens=0` on the
  repeat at `1582` prompt tokens.
- Step also failed cache reuse at 256 words / `814` prompt tokens.
- Step still passed at 128 words / `430` prompt tokens.

Root cause:

- Step uses mixed full/sliding-window attention with `RotatingKVCache` layers.
- The generic scheduler path tried to trim live post-decode cache back to the
  prompt boundary.
- At longer prompts, rotating metadata could not be safely rewound and
  `_extracted_cache` was never set.
- New telemetry made the skip explicit:
  `Skipping paged cache store ... no extracted cache (prompt_tokens=814, gen_prompt_len=8, block_cache_enabled=True)`.

Fix:

- `vmlx_engine/scheduler.py` now uses clean prompt-boundary re-prefill for
  generic mixed-SWA cache families, keyed to the same N-1 prompt-prefix contract
  used by paged cache hits.
- This avoids unsafe live-cache rewind for Step/Gemma/MiMo-style full plus
  sliding-window attention models in the standard text scheduler.
- The patch also logs skipped paged-cache store reasons instead of silently
  dropping cache stores.

Post-fix proof:

`build/current-local-long-context-cache-step37-installed-256w-after-mixed-swa-prefill-20260607/summary.json`

- Step 256 words / `814` prompt tokens: pass.
- Repeat reported `cached_tokens=805`, `cache_detail=paged`, visible
  `LONGCTX-OK`.

`build/current-local-long-context-cache-step37-installed-512w-after-mixed-swa-prefill-20260607/summary.json`

- Step 512 words / `1582` prompt tokens: pass.
- Repeat reported `cached_tokens=1573`, `cache_detail=paged`, visible
  `LONGCTX-OK`.
- Server log confirms `Mixed-SWA prefix cache store using clean prompt-boundary
  re-prefill (1573 cache-key tokens from 1574 prompt tokens)`.
- Server log confirms block-disk write-through of `25` blocks.

Combined post-fix artifact:

`build/current-local-long-context-cache-lfm-step-installed-512w-after-mixed-swa-prefill-20260607/summary.json`

- Summary `status=pass`, `row_count=4`.
- LFM JANG_2L: `cached_tokens=2604`, `cache_detail=paged+ssm`,
  output exactness review.
- LFM MXFP4: `cached_tokens=1580`, `cache_detail=paged+ssm`,
  output exactness pass.
- LFM MXFP8: `cached_tokens=1580`, `cache_detail=paged+ssm`,
  output exactness review.
- Step 3.7 Flash: `cached_tokens=1573`, `cache_detail=paged`,
  output exactness pass.

MiMo long-context result:

`build/current-local-long-context-cache-mimo-v25-installed-64w-after-mixed-swa-prefill-20260607/summary.json`

- MiMo 64-word long-context probe errored on the second request.
- Server log shows the first request completed in `28.17s` at about `0.2 tok/s`.
- The second request terminated the process with Metal OOM:
  `Command buffer execution failed: Insufficient Memory`.
- This is an MLLM scheduler/runtime memory blocker, not cleared by the text
  scheduler mixed-SWA fix.

Release boundary:

- This improves the Step/LFM long-prefix cache lane and adds a reusable gate.
- It does not clear largest possible context, MiMo long-context, Gemma4 media,
  UI launch parity, restart at long context, or release packaging.
