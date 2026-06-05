# Cross-Model Runtime Issue Register and Proof Tracker

Date: 2026-06-05

Purpose: never lose track of model/runtime/config/VL/audio/tool/cache regressions, proof requirements, and release status. This is an itemized tracker. Do not use a broad green check to close a narrow row unless the row's exact proof exists.

Related narrative document: `docs/internal/CROSS_MODEL_RUNTIME_FAILURE_CLASSES_2026_06_05.md`

Current known release state:

- vMLX 1.5.56 DMG hotfix shipped, signed, notarized, stapled, public updater/download current.
- `jjang-ai/vmlx` main after 1.5.56: `fa9f455b` includes structured JSON repair and DSV4 completions rail fix.
- PyPI is not current: PyPI latest remains `1.5.49`; `1.5.56` upload blocked by PyPI trusted-publisher/API-token config.
- Full cross-family runtime matrix remains open. Do not claim all model families production-cleared.

## Status Legend

- `[ ]` Not started or no current proof.
- `[~]` Partial proof exists; scope is narrower than row.
- `[x]` Current proof exists for this exact row.
- `[!]` Known failure or blocker.
- `[D]` Deferred by explicit release exception; still open.

## Golden Rule

A model/runtime/config path is production-cleared only when the exact model family, quant/runtime, modality, API path, cache mode, streaming mode, and lifecycle path being claimed have current proof. Load-only, health-only, or one text smoke does not clear image/video/audio/tool/cache behavior.

## Master Failure Classes

### CM-001 Unsupported advertised modality routes into unsafe runtime

Status: `[!]` Known concrete repro on Step3p7 CRACK. Must audit every family.

Symptom:

- Model metadata advertises vision/audio/video support.
- vMLX routes into MLLM/VLM/omni.
- Family-specific modality runtime is absent, incomplete, or unsafe.
- Server can crash or silently disconnect mid-request.

Concrete evidence:

- Step-3.7 Flash JANG_2L CRACK has `vision_config` and `jang_config.architecture.has_vision=true`.
- vMLX 1.5.55 classified it as `MLLM=True`.
- Unsupported MLLM path died after `MLLM.chat()` / chat template application.
- Text-only metadata view with `has_vision=false` loaded as `MLLM=False` and was stable.

Required checks:

- [ ] Print modality-detection source for every load: config, jang_config, registry, sidecar, CLI override.
- [ ] Assert unsupported modality routes fail closed before native forward.
- [ ] Require explicit override to enter known-unsupported modality path.
- [ ] Verify text-only workaround still works after guard.
- [ ] Verify error recovery: next text request works after rejected media/modality request.
- [ ] Add release-gate row for metadata-advertised unsupported modality.

Families to audit:

- [!] Step3p7: concrete failure exists; needs vMLX guard.
- [~] Gemma4 unified: 1.5.56 image hotfix verified for 12B JANG_4M/MXFP4/MXFP8, but audio/video full proof still open.
- [ ] Qwen VL/MTP: audit metadata modality vs runtime support.
- [ ] LFM: audit text and any VL-advertised variants.
- [ ] Nemotron Omni: audit audio/image/video dispatch and fallback.
- [D] DSV4: text/cache quality still deferred; audit media sidecars if present.
- [ ] Zaya/MiMo/Kimi VLM: audit advertised modality vs parser/runtime support.

Proof artifacts required per family:

```text
model_path
config modality fields
jang_config architecture fields
vMLX classification log
load mode: LLM or MLLM or omni
request route
one accepted supported modality request
one rejected unsupported modality request
post-error health
post-error text generation
server alive after all probes
```

### CM-002 Native crash without Python traceback

Status: `[!]` Known concrete repro on unsupported Step3p7 MLLM path.

Symptom:

- Client sees disconnected/no response or connection refused.
- Logs end at last Python stage.
- No Python traceback, so diagnosis is slow.

Required checks:

- [ ] Parent-process crash sentinel records last request id, model path, family, modality, cache mode, thinking mode, parser IDs.
- [ ] Last-stage breadcrumb before Metal/native forward.
- [ ] Crash reproducer uses safe prompt too, not only unsafe/adversarial prompt.
- [ ] Test unsupported route returns controlled 4xx/5xx instead of process death.

Architectures/runtime paths to probe:

- [!] Step3p7 unsupported MLLM.
- [~] Gemma4 image prefill: 1.5.56 now fails closed for oversized prefill and recovers.
- [ ] Video prefill/contact-sheet paths.
- [ ] Audio/omni dispatch.
- [ ] Qwen MTP/VL gated-delta paths.
- [ ] TurboQuant/JANGTQ kernels.
- [ ] DSV4 composite cache/state paths.

### CM-003 Tool-call dialect ambiguity

Status: `[!]` Known behavior failure on Step Flash CRACK eval. Cross-family open.

Symptom:

- Model receives API tool schemas.
- Sometimes emits structured `tool_calls`.
- Sometimes emits visible raw tool text like `<tool_call><function=...>`.
- Harness/client does not execute tool.

Concrete Step evidence:

- `A01`, `A17`, and several terminal-tool tasks emitted valid tool calls.
- `A20` emitted literal `<tool_call><function=write_file>...` text.
- `P11` emitted literal `<tool_call><function=terminal>...` text.

Required checks:

- [ ] For each parser family, run same tool prompt under Chat, Responses, Ollama, Anthropic adapter if applicable.
- [ ] Verify native API `tool_calls` object exists, not just visible raw tool text.
- [ ] Detect raw known dialect in visible text when tools were supplied.
- [ ] Convert raw dialect to tool_calls only when unambiguous and parser-supported, otherwise flag `raw_tool_dialect_leak`.
- [ ] Streaming path: verify streamed tool deltas and final structured tool calls.
- [ ] Multi-turn path: tool result is consumed and final answer is produced.

Parser families to audit:

- [!] Step3p5 / Step3p7 parser.
- [ ] Qwen parser.
- [ ] Gemma3/Gemma4 parser.
- [D] DSV4 DSML parser.
- [ ] Zaya XML parser.
- [ ] MiniMax parser.
- [ ] Nemotron parser.
- [ ] Hunyuan parser.
- [ ] Kimi parser.
- [ ] Generic fallback injection.

### CM-004 Tool loop after mock/tool output

Status: `[!]` Known behavior failure on Step Flash CRACK eval. Cross-family open.

Symptom:

- Model makes reasonable first diagnostic call.
- Receives enough tool output to answer.
- Repeats same/near-same command until turn budget exhausted.
- No final answer or weak synthesis.

Concrete Step tasks affected:

- `A14`
- `A16`
- `H09`
- `H12`
- `P09`

Required metrics:

- [ ] `tool_repeat_count`
- [ ] `same_tool_same_args_count`
- [ ] `near_duplicate_tool_call_count`
- [ ] `turn_budget_exhausted`
- [ ] `final_answer_after_tool_output`
- [ ] `used_tool_output_in_final_answer`
- [ ] `stopped_after_sufficient_observation`

Required checks:

- [ ] Repeated `id`, `whoami`, `top`, `systemctl`, `journalctl`, SSH, package install, file read loops.
- [ ] Mock permission error then final diagnostic.
- [ ] Mock OOM/service log then final diagnosis.
- [ ] File-read comparison then final synthesis.
- [ ] Tool-call budget pressure: model must stop, not continue blindly.

Families to audit:

- [!] Step3p7 CRACK text-only.
- [ ] Step3p5/Step3p7 non-CRACK.
- [ ] Qwen 27B/35B MTP and non-MTP.
- [ ] Gemma4 12B/27B/31B quants.
- [ ] LFM 8B/other.
- [D] DSV4.
- [ ] Nemotron Omni.
- [ ] Zaya/MiMo/Kimi.

### CM-005 Thinking/template mismatch

Status: `[!]` Known warning on Step Flash CRACK. Cross-family open.

Symptom:

- Runtime requests `enable_thinking=false`.
- Template still injects `<think>` or family-specific reasoning markers.
- Parser may hide leakage, but behavior and tool use can still drift.

Concrete Step evidence:

```text
Template for models/Step-3.7-Flash-JANG_2L-CRACK always injects <think> (ignores enable_thinking=False)
```

Required checks:

- [ ] Rendered prompt inspection under thinking off.
- [ ] Raw output inspection under thinking off.
- [ ] Visible output inspection after parser cleanup.
- [ ] Tool-call output under thinking off.
- [ ] Same prompt under thinking on/off; compare behavior.
- [ ] Verify defaults come from model/runtime metadata, not hidden agent knobs.

Families to audit:

- [!] Step3p5/Step3p7.
- [~] Gemma4: 1.5.56 set default visible-answer mode for Gemma4 unified; broader proof open.
- [ ] Qwen reasoning templates.
- [D] DSV4 thinking/direct/max rails.
- [ ] Zaya/MiMo XML thinking templates.
- [ ] GPT-OSS/MiniMax/Nemotron reasoning parsers.

### CM-006 Structured JSON/XML parse and repair reliability

Status: `[~]` Partial fix landed on main in `fa9f455b`; retry/guided decoding still open.

Problem:

- Models can understand task but emit malformed JSON/XML.
- Benchmark/database pipelines should not store broken structured output.

Concrete malformed JSON example:

```json
"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"
```

Fix landed:

- JSON repair in `parse_json_output` path.
- Handles fences, trailing commas, Python literals, prose, obvious missing closers, schema string-to-array coercion, adjacent string fragments.
- Conservative XML extraction helper added.
- Chat Completions route-level test proves canonical JSON output under `response_format=json_schema`.

Verified:

```text
.venv/bin/python -m pytest -q tests/test_structured_output.py
=> 38 passed, 2 skipped

.venv/bin/python -m pytest -q tests/test_server.py tests/test_api_models.py tests/test_ollama_adapter.py
=> 208 passed, 3 deselected

.venv/bin/python -m pytest -q tests/test_xml_function_tool_parser.py tests/test_tool_parsers.py -k 'xml or json or structured or schema'
=> 16 passed, 75 deselected
```

Still open:

- [ ] Retry model with "fix this JSON/XML only" when repair fails.
- [ ] Benchmark runner integration: score repaired object while tracking raw parse failure.
- [ ] Public docs: repair/validation vs hard constrained decoding.
- [ ] Investigate runtime-guided JSON/schema grammar support.
- [ ] Streaming structured output repair/canonicalization policy.

### CM-007 Cache/config/runtime interaction regressions

Status: `[~]` Some 1.5.56 hotfix paths verified; full matrix open.

Risk:

- A model works only with narrow flags, or cache/MTP/TurboQuant combinations break quality/stability.

Known relevant examples:

- Gemma4 12B JANG needed conservative runtime flags in earlier notes.
- Qwen35 MXFP8 MTP had reported packaged `gdn_sink` crash in old app; source 1.5.56 path verified no crash.
- Step Flash CRACK text-only launch used no continuous batching, no prefix cache, no KV quant, no native MTP.
- DSV4 cache/runtime has architecture-specific composite cache requirements; generic TurboQuant KV is not a drop-in substitute.

Required matrix per model:

- [ ] continuous batching on/off.
- [ ] prefix cache on/off.
- [ ] paged cache on/off where applicable.
- [ ] L2 disk cache hit/miss/restore.
- [ ] KV quantization none/default/TurboQuant where applicable.
- [ ] native MTP off/on/depth variants.
- [ ] max output tokens low/high.
- [ ] top_k/top_p/temp defaults vs explicit.
- [ ] soft sleep and JIT wake.
- [ ] recovery after rejected request.

### CM-008 VL/image/video/audio request recovery

Status: `[~]` Gemma4 image prefill hotfix verified in 1.5.56; broader matrix open.

Risk:

- A failed media turn remains in UI/database history and poisons later text prompts.
- Oversized media prefill kills server or logs scary tracebacks.
- Unsupported video/audio route silently downgrades or crashes.

1.5.56 proof:

- Gemma4 12B JANG_4M image request visible output.
- Gemma4 MXFP4/MXFP8 image request visible output.
- Oversized image prefill rejects with clean 413 and no traceback.
- Fresh text request after 413 returns visible response.
- Panel rollback removes failed media user message when there was no visible activity.

Still open:

- [ ] Step3p7 VLM unsupported route guard.
- [ ] Step video processing proof.
- [ ] Qwen VL video/image proof with MTP variants.
- [ ] Nemotron Omni audio/image/video proof.
- [ ] LFM VL proof if applicable.
- [ ] UI settings panel media/cache/max-token combination proof.
- [ ] Streaming media request proof.

### CM-009 Public release surface drift

Status: `[~]` 1.5.56 public surface fixed; PyPI still stale.

Known issue:

- Public sites and updater feeds can stay old after GitHub release.

1.5.56 proof:

- `mlx.studio/update/latest.json` returns `1.5.56`, final hashes, no-store/DYNAMIC.
- `mlx.studio/download/` links to 1.5.56 Sequoia/Tahoe and contains no 1.5.55 refs.
- `vmlx.net/update/latest.json` redirects/returns 1.5.56.
- `vmlx.net/download/` returns 1.5.56 links.
- GitHub releases `jjang-ai/vmlx` and `jjang-ai/mlxstudio` have all four assets.

Still open:

- [!] PyPI latest is `1.5.49`; `1.5.56` absent.
- [!] Local twine upload got HTTP 403.
- [!] GitHub trusted publishing failed with `invalid-publisher` for both main and tag OIDC claims.
- [ ] Fix PyPI project trusted publisher or add valid `PYPI_API_TOKEN` secret.
- [ ] Re-run Publish PyPI workflow and verify `https://pypi.org/pypi/vmlx/1.5.56/json`.

### CM-010 Release-gate proof quality

Status: `[D]` Current umbrella manifest still red for broad rows.

Risk:

- Agents claim production-ready from partial checks.

Known current state:

- Packaged 1.5.56 installed-app gate passed for Gemma4 JANG_4M hotfix path.
- Umbrella release manifest still has deferred/open rows for DSV4 long-output/code exactness and full cross-family Electron UI matrix.

Required policy:

- [ ] Do not mark `production-cleared` from `/health` or load-only.
- [ ] Do not clear a model family from one model size/quant.
- [ ] Do not clear tool use from one first-turn tool call.
- [ ] Do not clear VL from text-only stability.
- [ ] Do not clear cache from cache disabled.
- [ ] Do not clear streaming from non-streaming.
- [ ] Do not clear app/UI from source server only.

## Per-Family Proof Matrix

### Gemma4

Current status: `[~]` 12B image hotfix path verified; full family open.

Proofs present:

- [x] 12B JANG_4M image/text/Responses/multi-turn/cache/recovery source proof.
- [x] 12B MXFP4 image source proof.
- [x] 12B MXFP8 image source proof.
- [x] Packaged installed 1.5.56 Gemma4 JANG_4M API/cache/sleep proof.

Still needed:

- [ ] 27B and 31B if shipped/supported.
- [ ] Video path.
- [ ] Audio path if advertised.
- [ ] Full UI settings panel proof.
- [ ] Structured JSON/schema live proof after repair.
- [ ] Tool-call proof.
- [ ] MTP variants if present.

### Step / Step3p7

Current status: `[!]` Text-only stable for CRACK view; unsupported VLM route crash class open.

Proofs present from review:

- [x] Step Flash CRACK text-only view stable under vMLX 1.5.55.
- [x] Fast eval completed 18/18, 0 failed requests.
- [x] Runtime crash reproduced in unsupported MLLM route.

Still needed:

- [ ] Implement vMLX unsupported Step3p7 VLM guard.
- [ ] Reproduce on local vMLX 1.5.56/HEAD.
- [ ] Test Step 3.7 Flash JANG_2L non-CRACK and CRACK metadata variants.
- [ ] Tool dialect leak tests.
- [ ] Tool-loop eval rows.
- [ ] Thinking template render tests.
- [ ] Video processing/cache proof.

### Qwen

Current status: `[~]` Qwen35 MTP gdn_sink source proof exists for 1.5.56; broader open.

Proofs present:

- [x] Qwen3.6 35B MXFP8 MTP source Chat/Responses/160-token decode no `gdn_sink` crash.
- [x] MTP accepted-token logging observed.

Still needed:

- [ ] Packaged app Qwen35 MTP proof after 1.5.56 install.
- [ ] Qwen27 MTP proof.
- [ ] Qwen VL image/video proof.
- [ ] Tool dialect and loop proof.
- [ ] Structured JSON repair benchmark proof.

### LFM

Current status: `[ ]` Needs current matrix.

Required:

- [ ] MXFP4 text.
- [ ] MXFP8 text.
- [ ] Any VL-advertised variants.
- [ ] Tool/JSON/schema.
- [ ] Cache/sleep/UI path.

### DSV4

Current status: `[D]` Runtime/cache exactness still deferred.

Known:

- [x] Main now includes DSV4 `/v1/completions` chat rail fix test restored in `fa9f455b`.
- [D] Long-output/code exactness remains open.
- [D] Full real UI DSV4 proof remains open.

Required:

- [ ] Native SWA/CSA/HSA composite cache verification.
- [ ] Long output full-tail read.
- [ ] Code/file-generation exactness.
- [ ] Tool loops and DSML parser proof.
- [ ] Memory preflight on 128 GB/large model paths.

### Nemotron Omni / audio-video models

Current status: `[ ]` Needs matrix.

Required:

- [ ] Text-only smoke.
- [ ] Audio input.
- [ ] Audio output if supported.
- [ ] Image input.
- [ ] Video input.
- [ ] Unsupported modality failure closed.
- [ ] Cache/sleep/gateway proof.

### JANG / JANGTQ / MXFP cross-cutting

Current status: `[~]` Some hotfix paths verified; full matrix open.

Required:

- [ ] JANG_K non-TQ text/tool/cache proof.
- [ ] JANG_4M text/VL/tool proof.
- [ ] JANG_2L text/VL/tool proof.
- [ ] JANGTQ MoE codebook routing proof.
- [ ] MXFP4/MXFP8 dense and VL proof.
- [ ] Metadata mixed-precision/config consistency proof.
- [ ] TurboQuant kernel import and runtime proof.

## Required Evidence Template Per Run

Every model-family proof should record:

```json
{
  "date": "YYYY-MM-DD",
  "vmlx_version": "",
  "commit": "",
  "app_or_source": "source|packaged-app",
  "model_path": "",
  "model_family": "",
  "quant_runtime": "",
  "metadata": {
    "config_model_type": "",
    "architectures": [],
    "has_vision_config": false,
    "has_audio_config": false,
    "jang_has_vision": null,
    "jang_has_audio": null,
    "tool_parser": "",
    "reasoning_parser": "",
    "supports_thinking": null,
    "think_in_template": null
  },
  "route": {
    "api": "chat|responses|completions|anthropic|ollama|ui",
    "stream": false,
    "modality": "text|image|video|audio|mixed",
    "cache_mode": "",
    "mtp": "off|on|depth=N"
  },
  "checks": {
    "load_classification_logged": false,
    "visible_text_ok": false,
    "structured_json_or_xml_ok": false,
    "tool_calls_structured_not_raw_text": false,
    "multi_turn_recall_ok": false,
    "cache_hit_or_disabled_proven": false,
    "unsupported_modality_fails_closed": false,
    "post_error_recovery_ok": false,
    "soft_sleep_wake_ok": false,
    "server_survived": false
  },
  "metrics": {
    "eval_tok_s": null,
    "latency_p50_s": null,
    "latency_p95_s": null,
    "latency_p99_s": null,
    "tool_repeat_count": null,
    "same_tool_same_args_count": null,
    "turn_budget_exhausted": null
  },
  "artifacts": {
    "server_log": "",
    "summary_json": "",
    "release_gate_summary": ""
  },
  "verdict": "pass|partial|fail|blocked"
}
```

## Immediate Next Work Queue

1. `[!]` Build vMLX guard for Step3p7 advertised vision when Step3p7 VLM runtime is unsupported.
2. `[!]` Add release-gate metadata/runtime modality mismatch audit.
3. `[!]` Add raw tool dialect leak detector or parser repair for known XML-ish tool text.
4. `[!]` Add loop-control eval metrics and rows.
5. `[~]` Finish structured output retry/guided decoding work after `fa9f455b` repair layer.
6. `[!]` Fix PyPI trusted publisher or token so Python package release is not stale.
7. `[D]` Re-run DSV4 long-output/code exactness and real UI proof when memory/model state allows.
8. `[ ]` Execute per-family matrix for Gemma4, Qwen, LFM, Step, DSV4, Nemotron, Zaya/MiMo/Kimi, JANG/JANGTQ/MXFP.

## Non-Negotiable Release Notes

- Do not say "all good" for a model family until this register has a proof row for that family and route.
- Do not say "VL works" if only text-only worked.
- Do not say "tools work" if raw tool text leaked in any path without a parser/repair/flag.
- Do not say "cache works" if the successful run disabled cache.
- Do not say "packaged app works" from source-server proof.
- Do not say "PyPI released" until PyPI has the exact version JSON.

## GitHub Tracking Issues

- Master cross-model matrix: https://github.com/jjang-ai/vmlx/issues/188
- Unsupported advertised modality/runtime guard: https://github.com/jjang-ai/vmlx/issues/189
- Raw tool dialect leaks, tool loops, thinking-template mismatch: https://github.com/jjang-ai/vmlx/issues/190
- Structured JSON/XML repair follow-up: https://github.com/jjang-ai/vmlx/issues/187
