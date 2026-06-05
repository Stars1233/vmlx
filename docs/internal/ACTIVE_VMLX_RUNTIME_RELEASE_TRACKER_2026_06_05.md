# Active vMLX Runtime Release Tracker - 2026-06-05

Purpose: durable checklist for the current vMLX/JANG release push so runtime work, proofs, release state, and model-family nuances do not get dropped or overclaimed.

Working repo: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`

Do not use: `/Users/eric/vmlx` for active app/runtime work.

Current shipped app release: `v1.5.56`

Current Step3p7 guard/proof base on `main`: `3782f8179f2aa23a45d8f41bf1fc64988d4d20ce`

Release posture: `v1.5.56` is shipped, signed, notarized, and publicly surfaced, but it is stale for the Step3p7 advertised-VLM guard. A clean staged `1.5.56` app built from `3782f817` passes the Step3p7 guard proof, but the public installed app still fails that proof and must not be called fixed until a rebuilt, notarized, installed release is shipped.

---

## 0. Non-Negotiable Proof Rules

- [ ] Never call a runtime production-ready from load-only, `/health`, import-only, or a single narrow prompt.
- [ ] Never claim cache correctness without cache hit counters, cache reuse behavior, and post-error recovery evidence.
- [ ] Never claim VL/video/audio works unless media requests produce visible output, recover after failure, and do not poison later text turns.
- [ ] Never claim tool support works unless native OpenAI `tool_calls` are emitted when tools are supplied, tool outputs are consumed, and the model stops rather than looping.
- [ ] Never hide bad defaults by forcing agent-side sampling/parser/cache knobs without tracing `generation_config.json`, `jang_config.json`, tokenizer/chat-template metadata, request assembly, parser selection, and cache mode.
- [ ] Never present MTP as a speed win unless active runtime proof, stable streaming, and comparable output behavior exist.
- [ ] Never pack or publish dirty untracked runtime/vendor material.
- [ ] Never use deprecated `/Users/eric/vmlx` notes as current guidance.

---

## 1. Release / Distribution State

### vMLX app `1.5.56`

Status: shipped.

- [x] Commit `510174e62cee10dd26dfa908957e27c95cdf54a3` pushed to `main`.
- [x] Tag `v1.5.56` created.
- [x] GitHub releases created for `jjang-ai/vmlx` and `jjang-ai/mlxstudio`.
- [x] Sequoia DMG signed, notarized, stapled, and release-verified.
- [x] Tahoe DMG signed, notarized, stapled, and release-verified.
- [x] `/Applications/vMLX.app` installed from Tahoe DMG.
- [x] Installed app reports app version `1.5.56` and bundled engine `1.5.56`.
- [x] Gatekeeper accepts installed app as notarized Developer ID.
- [ ] Installed `/Applications/vMLX.app` is stale for CM-001 Step3p7 guard: neutral-cwd packaged proof routed Step3p7 as `MLLM=True` and accepted media instead of failing closed.
- [x] Clean staged app from `3782f817` passes CM-001 Step3p7 guard proof from neutral cwd, but is not notarized.
- [x] Clean staged app release-gate packaged checks pass when the gate is pointed at `/tmp/vmlx-clean-step37-3782f817/panel/release/mac-arm64/vMLX.app`.
- [ ] Notarization is blocked: no Apple/ASC/CSC/NOTAR environment credentials are present, `xcrun notarytool history --keychain-profile vmlx-notary` reports default keychain locked, and `security find-generic-password` did not expose `vmlx-notary` while locked.
- [ ] Public DMG build is blocked by `--require-prepackage-ready`; current proof sweep is still open and must not be bypassed for production release.
- [x] `mlx.studio/update/latest.json` serves `1.5.56` with no-store cache policy.
- [x] `mlx.studio/download/` serves `1.5.56` links, not stale `1.5.55` links.
- [x] `vmlx.net/update/latest.json` and `/download/` resolve to `1.5.56`.
- [ ] Umbrella release gate is still open because broad cross-family proof is incomplete.

### PyPI `vmlx-engine`

Status: blocked.

- [x] Clean archive build produced `vmlx-1.5.56` wheel/sdist with zero `vmlx_engine/vendor/` entries.
- [x] `twine check` passed on clean artifacts.
- [x] Local upload failed HTTP 403.
- [x] GitHub trusted publishing failed for `main` and tag `v1.5.56` with `invalid-publisher`.
- [ ] PyPI still latest `1.5.49`; `1.5.56` absent.
- [ ] Fix PyPI trusted-publisher config or token path.
- [ ] Re-run publish from clean workflow/artifacts only.
- [ ] Verify `pip install vmlx-engine==1.5.56` after publish.

---

## 2. Current Commit / Issue Ledger

Commits:

- [x] `510174e6` - Release vMLX 1.5.56 Gemma4 VLM hotfix.
- [x] `fa9f455b` - Add structured output repair and restore DSV4 completions rail.
- [x] `a404cd44` - Document cross-model runtime issue matrix.
- [x] `50ad9225` - Guard Step3p7 advertised VLM routing.
- [x] `7de5debd` - Prove Step3p7 media rejection recovers.
- [x] CM-001 source/live proof completed for the local Step3p7 artifact; packaged installed-app release proof still pending.

Docs:

- [x] `docs/internal/CROSS_MODEL_RUNTIME_FAILURE_CLASSES_2026_06_05.md`
- [x] `docs/internal/CROSS_MODEL_RUNTIME_ISSUE_REGISTER_2026_06_05.md`
- [x] `docs/internal/ACTIVE_VMLX_RUNTIME_RELEASE_TRACKER_2026_06_05.md`

GitHub issues:

- [x] `#187` - Structured JSON/XML repair and guided-output follow-up.
- [x] `#188` - Master cross-model runtime matrix: metadata, modality, tools, cache, VL/audio.
- [x] `#189` - Unsupported advertised modality/runtime guard.
- [x] `#190` - Raw tool dialect leaks, tool loops, thinking-template mismatch.

---

## 3. Failure Classes That Must Stay Visible

### CM-001: Unsupported advertised modality routes into unsafe runtime

Status: source guard implemented, live-source proven, and clean staged packaged app proven. Public installed app is stale and failed the guard proof; notarized installed release proof is pending.

Classification:

- Model-side hazard: CRACK-style Step3p7 artifacts advertise vision through `vision_config` and JANG `architecture.has_vision=true`, while the vMLX Step3p7 VLM path is not production-cleared. Model uploads should ship a vMLX text-only metadata view or explicit vMLX card guidance until Step3p7 VLM support lands.
- Runtime issue: vMLX must not trust unsupported advertised modality and route into an unsafe MLLM path that can crash the server. The engine must fail closed, explain the guard, reject media cleanly, and recover later text requests.

Observed Step3p7 shape:

- Metadata advertises `vision_config` and `jang_config.architecture.has_vision=true`.
- vMLX classifies the model as MLLM.
- Longer generation can kill the server with no Python traceback.
- Text-only metadata view with `jang_config.architecture.has_vision=false` is stable.

Required behavior:

- [x] Step3p7 advertised-VLM metadata fails closed to text-only unless a production-cleared VLM route is introduced.
- [x] `force_mllm=True` does not bypass the safety guard accidentally.
- [x] Text-only override remains honored.
- [x] Route-level media request against text-only Step3p7 route returns controlled unsupported-media error, not generation/crash.
- [x] Logs must explain why advertised modality was refused or ignored.

Proof checklist:

- [x] Unit test CRACK-style metadata with `model_type=step3p7`, `vision_config`, and `has_vision=true`.
- [x] Unit test text-only view with `has_vision=false`.
- [x] Unit test force behavior: `force_mllm=True` still routes text-only by default.
- [x] Live Step 3.7 Chat text smoke.
- [x] Live Step 3.7 Responses text smoke.
- [x] Live Step 3.7 Chat media request returns controlled text-only unsupported-media rejection.
- [x] Live Step 3.7 Responses media request returns controlled text-only unsupported-media rejection.
- [x] Server route-level media request returns controlled text-only unsupported-media rejection.
- [x] Server route-level post-media text request still works after rejection.
- [x] Live post-media text request still works after rejection.
- [x] Clean staged packaged app proof for the same Step3p7 guard and recovery.
- [ ] Notarized installed-app proof for the same Step3p7 guard and recovery.

Live proof artifact:

- `build/step3p7-live-proof-20260605/proof.json`: `pass=true`.
- Passed assertions: Chat text before media HTTP 200, Chat media rejected HTTP 400, rejection mentions text-only, Chat text after media HTTP 200, Responses media rejected HTTP 400, rejection mentions text-only, Responses text after media HTTP 200, `/health` HTTP 200.
- Server log evidence: `tier=step3p7_advertised_vlm_text_only result=False` and `SimpleEngine loaded ... (MLLM=False)`.

Packaged proof artifacts:

- Stale installed app failure: `build/step3p7-installed-app-proof-20260605/proof.json`: `pass=false`. `/Applications/vMLX.app` launched from `/tmp` routed Step3p7 as `MLLM=True`, accepted image requests, returned `White background`, and stayed healthy. This proves the public installed app is not fixed for CM-001.
- Clean staged app pass: `build/step3p7-clean-staged-app-proof-20260605/proof.json`: `pass=true`. `/tmp/vmlx-clean-step37-3782f817/panel/release/mac-arm64/vMLX.app` launched from `/tmp` routed Step3p7 as text-only, rejected Chat and Responses media with HTTP 400, recovered later text requests, and logged `tier=step3p7_advertised_vlm_text_only result=False` plus `MLLM=False`.
- Clean staged app signing status: `codesign --verify --deep --strict --verbose=2` passed; `spctl --assess --type execute --verbose=2` rejected it as `Unnotarized Developer ID`.
- Clean staged release-gate app check: `docs/internal/release-gates/20260605_142643/SUMMARY.md` shows packaged app exists/version/imports/source hashes/JANG hashes/Developer ID signature/codesign strict verify all `PASS`; gate remains `FAIL` overall because dist artifacts are absent, active bundled app is stale, objective proof digest has open requirements, and notarization is absent.
- Step3p7 crash-falsification contract updated to accept the packaged text-only guard proof as the current supported behavior. `build/current-step37-crash-falsification-contract-20260604.json`: `status=pass`.

Expanded no-heavy route audit:

- `build/current-local-generation-metadata-route-audit-20260605-step3p7-expanded.json`: `status=pass`, `rows=21`, `hard_failures=[]`.
- Step3p7 rows flagged as intentionally guarded text-only despite advertised vision: `Step-3.7-Flash-JANG_2L-CRACK`, `Step-3.7-Flash-JANG_K-CRACK`, `Step-3.7-Flash-JANG_2L`, `Step-3.7-Flash-JANG_K`.
- Advertised-VL rows that still route MLLM and therefore require live media proof, not metadata suppression: Gemma4 12B MXFP4/MXFP8/JANG_4M, Gemma4 31B JANG_4M-MTP, Qwen3.6 27B/35B MTP MXFP rows, ZAYA1-VL JANGTQ4/MXFP4.
- Omni rows route through dispatcher rather than generic MLLM and remain covered by separate omni media proof rows.

Current regression-suite refresh:

- `tests/cross_matrix/summarize_objective_proof.py --out build/current-objective-proof-audit-gemma4-release-boundary-20260604.json` regenerated the objective digest and still reports open requirements for DSV4 cache/tool/code quality, Qwen MTP speed/equivalence, Ling/Gemma4 quality, cross-family live smoke, MiniMax reporter parity, real Electron UI, and DSV4 long-output/code/file-generation quality.
- `tests/cross_matrix/run_current_regression_suite.py` regenerated the missing no-heavy artifacts, but the run remained `status=open`; after the Step3p7 contract fix, `step37_crash_falsification_contract` is no longer in `failed_steps`.
- Remaining failed suite steps in the stopped refresh artifact: `tool_call_contracts`, `packaged_integrity_contracts`, `issue175_179_release_boundary_audit`, `public_app_issue_audit`.

### CM-002: Native crash without Python traceback

Status: open.

Required checks:

- [ ] Add/capture crash-boundary logging around family dispatch, MLLM wrapper call, Metal forward, and stream finalization.
- [ ] Verify unsupported paths return controlled 4xx/5xx instead of process death.
- [ ] Capture last log lines for every model family that dies natively.

### CM-003: Tool-call dialect ambiguity

Status: open.

Observed shape:

- Same model can emit real OpenAI `tool_calls` in one task and raw XML-ish tool text in another.

Required checks:

- [ ] OpenAI Chat Completions tool call with supplied tools.
- [ ] Responses API tool call with supplied tools.
- [ ] Tool output follow-up turn synthesizes final answer.
- [ ] No raw XML/tool markup leaks when native tools are supplied.
- [ ] XML parser fallback only used where intentionally configured and logged.

### CM-004: Tool loop after mock/tool output

Status: open.

Observed shape:

- Repeated `top`, `id`, `systemctl`, `journalctl`, `ssh`, or file reads.
- Model does not synthesize final answer after enough evidence.

Required checks:

- [ ] Multi-turn loop cap test with repeated mock output.
- [ ] Expected final-answer stop after sufficient tool output.
- [ ] Parser/runtime does not re-inject stale tool calls.
- [ ] Max-turn exhaustion is reported as failure, not success.

### CM-005: Thinking/template mismatch

Status: open.

Required checks:

- [ ] Trace request-level `enable_thinking`.
- [ ] Trace model registry default.
- [ ] Trace tokenizer/chat-template behavior.
- [ ] Confirm visible output excludes hidden-thinking artifacts.
- [ ] Confirm thinking-off does not silently break tool formatting.

### CM-006: Structured JSON/XML parse and repair reliability

Status: partially implemented.

Already done:

- [x] Markdown fence stripping.
- [x] Trailing comma repair.
- [x] Python literal repair for `True`, `False`, `None`.
- [x] Missing closer repair for obvious brace/bracket cases.
- [x] Schema-driven adjacent string array repair.
- [x] Schema-driven string-to-array coercion.
- [x] Conservative XML extraction helper.
- [x] Route-level Chat Completions fake-engine test proves repaired JSON returns canonical valid JSON under schema response format.

Still needed:

- [ ] Add retry path: ask model to fix invalid JSON only when repair fails.
- [ ] Integrate repair/validation into video catalogue benchmark runner.
- [ ] Report raw parse success vs repaired parse success vs retry success separately.
- [ ] Investigate guided JSON/schema decoding support instead of repair-only.
- [ ] Define streaming repair policy: repair at final aggregation, not mid-token.

Verification already run:

- [x] `tests/test_structured_output.py` passed: `38 passed, 2 skipped`.
- [x] `tests/test_server.py tests/test_api_models.py tests/test_ollama_adapter.py` passed: `208 passed, 3 deselected`.
- [x] Tool/XML focused tests passed: `16 passed, 75 deselected`.

### CM-007: Cache/config/runtime regressions

Status: open.

Cache modes that must be tested per architecture:

- [ ] Prefix cache.
- [ ] Paged cache.
- [ ] L2 disk cache.
- [ ] TurboQuant KV cache.
- [ ] Native architecture cache, such as DSV4 SWA/CSA/HSA composite cache.
- [ ] KV quantization disabled path.
- [ ] Post-error cache cleanup and recovery.

Required checks:

- [ ] Cache counters visible in API usage/stats.
- [ ] Cache type reflected in settings panel and runtime logs.
- [ ] Cache hit occurs on repeated/multi-turn request.
- [ ] Cache miss occurs when prompt/model/session changes.
- [ ] Error during media/tool/stream request does not poison later cache state.

### CM-008: VL/image/video/audio request recovery

Status: partially fixed for Gemma4 image guard and media rollback.

Already done in `1.5.56`:

- [x] Adaptive VLM image prefill guard based on Metal max working set.
- [x] Explicit env override preserved for guard.
- [x] Failed media turn rollback in panel chat IPC.
- [x] Gemma4 JANG_4M image request visible in source and installed app proof.
- [x] Oversized image/prompt request returns clean rejection and subsequent text recovers.
- [x] Gemma4 MXFP4/MXFP8 image visible in source proof.

Still needed:

- [ ] Step 3.7 Flash video processing path.
- [ ] Step 3.7 Flash image/media unsupported-path guard.
- [ ] Qwen VL image and video smoke.
- [ ] LFM VL image and video smoke.
- [ ] Gemma4 video path.
- [ ] Audio request path and fallback/rejection by architecture.
- [ ] UI upload flow from settings/panel through server and back.
- [ ] Post-media-error text recovery from UI and API.

### CM-009: Public release surface drift

Status: fixed for `1.5.56`, keep checking on every release.

Required checks every release:

- [ ] GitHub release assets are correct and final.
- [ ] `latest.json` hashes match final DMGs.
- [ ] `mlx.studio/download/` points at current release.
- [ ] `vmlx.net/download/` points at current release.
- [ ] CDN/cache headers do not serve stale release.
- [ ] Installed app version equals public version.
- [ ] Bundled engine version equals app version.

### CM-010: Release-gate proof quality

Status: open umbrella.

Required proof dimensions:

- [ ] Installed app proof, not source-only.
- [ ] Python engine proof.
- [ ] UI/panel proof.
- [ ] OpenAI Chat proof.
- [ ] Responses API proof.
- [ ] Anthropic API proof.
- [ ] Ollama API proof.
- [ ] Multi-turn proof.
- [ ] Tool proof.
- [ ] Structured output proof.
- [ ] Cache proof.
- [ ] Media proof.
- [ ] Streaming proof.
- [ ] Sleep/wake or JIT soft-wake proof.
- [ ] Settings panel changes reflected in runtime behavior.

---

## 4. Model-Family Matrix

### Gemma 4 12B

Bundles:

- `MXFP4`
- `MXFP8`
- `JANG_4M`

Current state:

- [x] Bundles uploaded to `OsaurusAI` and `JANGQ-AI` from prior work.
- [x] Text works under conservative vMLX flags.
- [x] Gemma4 registry defaults thinking off for cleaner visible answers.
- [x] `1.5.56` fixes false 8GB VLM image prefill guard issue.
- [x] Image request visible for JANG_4M in installed app proof.
- [x] Image request visible for MXFP4/MXFP8 source proof.
- [x] Post-oversize media error text recovery proved.

Still needed:

- [ ] Full UI upload path proof.
- [ ] Video path proof.
- [ ] Audio path support or clean rejection proof.
- [ ] Prefix cache, paged cache, L2 disk cache, TurboQuant KV, no-KV-quant modes.
- [ ] Speed target investigation: user expects around `45 tok/s` for Gemma4 12B; identify top-k, SWA, heterogeneous attention, cache, or sampling regression if slower.

### Qwen 35B MXFP8 with MTP

Current state:

- [x] `gdn_sink` MTP regression fixed for source and release path.
- [x] Qwen35 MTP Chat/Responses/160-token decode did not crash after fix.
- [x] MTP active in proof.
- [x] TurboQuant KV and paged hybrid cache proof existed in source proof.
- [x] Logged speed around `102.6 tok/s` in source proof.

Still needed:

- [ ] Installed app UI proof for Qwen35 MTP.
- [ ] 27B family regression check.
- [ ] MTP equivalence and output-quality check, not just speed/crash.
- [ ] VL/video behavior if model advertises media.

### Step 3.7 Flash JANG_2L / JANG_K / CRACK

Current state:

- [x] Text-only view with `has_vision=false` is stable in reported eval.
- [x] Fast eval reported `113/180`, `62.78%`, `18/18` runtime completed, `16.5 eval tok/s`.
- [x] Main runtime issue identified: default advertised vision can route unsupported VLM path and crash.
- [ ] CM-001 guard not yet implemented/verified.

Behavioral issues:

- [ ] Raw XML-ish tool text can leak instead of native `tool_calls`.
- [ ] Tool-call loops after mock outputs.
- [ ] Weak multi-turn synthesis and persona retention on harder tasks.
- [ ] Template may inject think markers despite thinking disabled.

Required Step checks:

- [ ] Text-only Chat and Responses.
- [ ] Tool-call native protocol.
- [ ] Multi-turn tool output stop/synthesis.
- [ ] Image/video unsupported-path guard or working path.
- [ ] Step Flash video processing and caching.
- [ ] Prefix/paged/L2/TurboQuant/no-KV-quant cache matrix.

### LFM 2.5 8B

Bundles:

- `MXFP4`
- `MXFP8`

Current state:

- [x] Prior local Osaurus proof said uploaded and working.
- [ ] vMLX installed app proof still needed in current matrix.
- [ ] VL/video/audio behavior still needs architecture-specific check.
- [ ] Cache matrix still needed.

### DSV4 / DeepSeek V4 Flash

Current state:

- [x] DSV4 completions rail fix committed in structured-output commit.
- [ ] Do not treat generic TurboQuant KV as substitute for native DSV4 SWA/CSA/HSA composite cache.
- [ ] Long-output/code exactness remains a known umbrella gate.

Required checks:

- [ ] Chat.
- [ ] Completions.
- [ ] Responses.
- [ ] Long output.
- [ ] Code output exactness.
- [ ] Native composite cache behavior.
- [ ] TurboQuant/JANGTQ interaction only if architecture-appropriate.

### JANG / JANGTQ / MXFP quants

Required checks for every quant family:

- [ ] Loader detects quant type correctly.
- [ ] Config metadata matches actual safetensor shapes or mismatch is explicitly expected.
- [ ] No modules silently dequantized after quant loading unless intended.
- [ ] TurboQuant router/kernel path exists if claimed.
- [ ] JANGTQ only claimed where architecture actually supports it.
- [ ] Dense non-MoE models do not get fake JANGTQ claims.
- [ ] JANG_K non-TQ bundles considered as alternate quality path where relevant.

### Mimo

Current scope decision:

- [x] Mimo is not in the active release list for now.
- [ ] Do not spend release-critical time on Mimo unless Eric explicitly re-adds it.
- [ ] If revisited, use `~/adlab` notes on `erics-m5-max2.local` and first determine whether issue is architecture support or corrupt quants.

---

## 5. UI / Settings Panel Proof Matrix

Required installed-app checks:

- [ ] Launch installed `/Applications/vMLX.app`.
- [ ] Confirm app version and bundled engine version in UI/logs.
- [ ] Select model from UI.
- [ ] Change max output tokens in settings panel and prove runtime request obeys it.
- [ ] Change cache type in settings panel and prove runtime logs/stats reflect it.
- [ ] Toggle thinking setting and prove request/template behavior follows it.
- [ ] Toggle MTP/speculative setting and prove active/inactive behavior.
- [ ] Upload image and prove visible output or clean unsupported error.
- [ ] Upload video and prove visible output or clean unsupported error.
- [ ] Submit text after failed media request and prove recovery.
- [ ] Multi-turn chat remembers prior context.
- [ ] Tool request persists visible/native tool call where UI supports it.
- [ ] Sleep/wake or JIT soft-wake remains healthy.

---

## 6. API Proof Matrix

For each target model and quant:

- [ ] `/v1/chat/completions` text smoke.
- [ ] `/v1/chat/completions` streaming smoke.
- [ ] `/v1/chat/completions` tools smoke.
- [ ] `/v1/chat/completions` structured output smoke.
- [ ] `/v1/responses` text smoke.
- [ ] `/v1/responses` streaming smoke.
- [ ] `/v1/responses` tools smoke.
- [ ] `/v1/completions` text smoke where supported.
- [ ] Anthropic-compatible message smoke.
- [ ] Ollama-compatible message smoke.
- [ ] Multi-turn context carryover.
- [ ] Error recovery after malformed request.
- [ ] Error recovery after unsupported media request.

---

## 7. Runtime Launch Flag Matrix

Conservative known-good flags for fragile Gemma4 path:

```bash
--no-continuous-batching
--disable-prefix-cache
--kv-cache-quantization none
--disable-native-mtp
```

Need systematic proof for each feature before enabling by default:

- [ ] Continuous batching on/off.
- [ ] Prefix cache on/off.
- [ ] Paged cache on/off.
- [ ] L2 disk cache on/off.
- [ ] KV cache quantization none/TurboQuant where valid.
- [ ] Native MTP on/off.
- [ ] Max output tokens from CLI and UI.
- [ ] Thinking on/off.
- [ ] Tool parser selection.
- [ ] Reasoning parser selection.

---

## 8. Evidence Template For Every Proof Row

Use this shape in logs/docs/issues:

```json
{
  "date": "2026-06-05",
  "repo": "/Users/eric/mlx/vllm-mlx-finite-launch-guard",
  "commit": "unknown",
  "app_version": "unknown",
  "engine_version": "unknown",
  "installed_app": false,
  "model": "unknown",
  "quant": "unknown",
  "architecture": "unknown",
  "api_surface": "chat|responses|completions|anthropic|ollama|ui",
  "modality": "text|image|video|audio|tool|structured",
  "cache_mode": "unknown",
  "mtp": "disabled|enabled|not_applicable",
  "thinking": "disabled|enabled|not_applicable",
  "prompt_shape": "short description",
  "expected": "short description",
  "observed": "short description",
  "visible_output": true,
  "tool_calls_native": null,
  "cache_hit_evidence": null,
  "speed_tok_s": null,
  "passed": false,
  "logs_or_artifacts": []
}
```

---

## 9. Immediate Next Work Queue

1. [x] Finish CM-001 Step3p7 advertised-VLM source guard.
2. [x] Add focused tests for Step3p7 default metadata, text-only override, and force behavior.
3. [x] Run focused Step3p7 model-family tests.
4. [ ] Push guard commit to `main` if green.
5. [ ] Re-run installed-app proof row if guard needs release build.
6. [ ] Build next release candidate only after cross-family crash regressions are pinned or explicitly deferred with honest release notes.
7. [ ] Fix PyPI publisher before claiming engine distribution is current.
8. [ ] Execute UI/settings/cache/media matrix for Gemma4, Qwen35/27B, Step3.7, LFM, DSV4.
9. [ ] Add benchmark JSON repair integration and re-score structured video catalogue tests with raw/repaired/retry split.
10. [ ] Update GitHub issues #187-#190 with exact proof rows and remaining red gates.

---

## 10. Stop/Ship Criteria

Do not ship the next release unless one of these is true:

- [ ] All blocking model-family rows are green with evidence.
- [ ] A release note explicitly scopes the release to a narrow hotfix and lists every deferred red/yellow gate honestly.

Minimum hotfix release criteria:

- [ ] Source tests pass for touched area.
- [ ] Installed app builds cleanly.
- [ ] DMGs signed/notarized/stapled.
- [ ] Public update/download endpoints updated and cache-safe.
- [ ] Installed app proof passes for the hotfix model/path.
- [ ] No dirty/untracked packaged runtime files included.
- [ ] Main pushed.
- [ ] Tag pushed.
- [ ] GitHub releases updated for both repos if app release.
