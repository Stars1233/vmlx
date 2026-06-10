# 128GB Checkpoint Proof Matrix - MiMo, N2, Gemma

Scope: current Python engine worktree
`/Users/eric/mlx/vllm-mlx-finite-launch-guard`. This file is for the
release/checkpoint lane. Do not use it to claim full release clearance; it
separates what was actually loaded and proven from what remains red.

## Proven Live On 128GB Host

### Dev-App Detector and Settings Launch Parity

Artifacts:

- `build/current-panel-settings-contract-proof-20260610-mimo-n2-gemma-launch-parity.json`
- `build/current-panel-exact-local-model-detect-mimo-n2-gemma-20260610.json`

Proven:

- Panel settings/launch contract is current-source green:
  `status=pass`, `missing_source_markers=[]`, panel settings tests passed
  `315`, model registry tests passed `66` in the contract artifact,
  engine model registry passed `140`, and CLI flag contract passed `9`.
- Exact local Gemma 12B MXFP4 and JANG4M directories now autodetect as
  `family=gemma4`, `cacheType=rotating_kv`, `usePagedCache=true`,
  `toolParser=gemma4`, `reasoningParser=gemma4`, and multimodal.
- Exact local MiMo JANG_2L and JANGTQ_2 directories now autodetect as
  `family=mimo_v2`, `cacheSubtype=mimo_v2_asymmetric_swa`,
  `usePagedCache=true`, `toolParser=xml_function`, and no automatic reasoning
  claim.
- Exact local N2 JANGTQ2 directory autodetects as `qwen3.5-moe`,
  `cacheType=hybrid`, paged, Qwen tools, Qwen3 reasoning, TurboQuant, and
  multimodal.
- Exact local N2 JANG_1L directory autodetects as `qwen3.5-moe`,
  `cacheType=hybrid`, paged, Qwen tools, Qwen3 reasoning, but
  `forceTextOnly=true` until VL and memory-safe runtime proof exist.

Fixes made:

- Added panel `gemma4_unified` and `gemma4_unified_text` model-type aliases so
  Gemma 4 unified bundles do not fall to `unknown`.
- Aligned panel MiMo detection with the Python registry: XML tools remain on,
  `mimo_v2_asymmetric_swa` cache subtype is exposed, paged cache is forced for
  that subtype, and MiMo reasoning is not auto-advertised without visible-final
  thinking proof.

Not proven:

- Electron UI clicked chat transcript for these exact rows.
- Installed-app packaged parity for these exact rows after this source change.
- N2 JANG_1L memory-safe live startup.
- MiMo JANGTQ_2 exactness.
- Gemma audio/video semantic E2E.
- Same-model direct/gateway/tunnel raw SSE deployed parity.

### Gemma 4 12B MXFP4 and JANG4M

Artifacts:

- `build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json`
- `build/current-gemma4-12b-mxfp4-jang4m-live-runtime-audit-20260610.json`

Proven:

- MXFP4 and JANG4M both loaded through current vMLX server.
- Image/media path returned correct visible color answer.
- Conservative text runtime passed for both rows.
- Visible answer, multi-turn recall, reasoning-on visible answer, required
  tool call, and cache endpoint sanity passed.

Not proven:

- Installed-app/UI parity for these exact new artifacts.
- Audio/video weight-backed E2E.
- Full larger Gemma QAT matrix through UI/installed app.
- Tunnel/gateway parity for these exact Gemma rows.

### Gemma 4 12B JANG4M Real Dev-App Proof

Artifact:

- `build/current-real-ui-live-model-gemma4-12b-jang4m-dev-app-proof-20260610.json`

Raw ignored proof captures:

- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-responses-tools-cache-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-gemma4-12b-jang4m-image-cache-20260610-proof.json`

Proven:

- Real Electron dev app launched with `npm run dev`, connected to a real vMLX
  server loading `/Users/eric/models/JANGQ-AI/gemma-4-12B-it-JANG_4M`.
- Responses UI rail passed with built-in `run_command` tools enabled.
- Two-turn tool loop executed real commands, wrote/read
  `REAL_UI_LIVE_TOOL_ONE`, wrote `REAL_UI_LIVE_TOOL_TWO`, and persisted tool
  execution/result phases in the chat UI state.
- Responses stream trace carried visible content deltas and server metrics.
  The two turns ended with `cachedTokens=2687` and `cachedTokens=2901`,
  `cacheDetail=paged+mixed_swa`, and live decode around `45 t/s`.
- Chat Completions UI rail passed with image attachment enabled.
- The attached red image reached the Gemma MLLM media path and returned visible
  `Red`; `media.imageSemanticVerified=true`.
- Server/cache controls were visible in the app session and matched the
  expected cache labels.
- Gemma native cache surfaced as mixed-SWA with prefix cache, paged cache, and
  block-disk L2 writes.
- No raw parser/tool markup leak and no hidden reasoning leak were observed in
  either dev-app run.

Not proven:

- Installed packaged app parity.
- DMG package/sign/notarize/release readiness.
- Local panel session manager starting this exact model from launch args; these
  app proofs used a remote session connected to the server started by the proof
  harness.
- Gemma audio/video semantic E2E.
- N2 or MiMo dev-app chat proof.
- Same-model public tunnel raw SSE parity.

### MiMo V2.5 JANGTQ_2

Artifacts:

- `build/current-mimo-v25-jangtq2-live-cb-cache-text-20260610.json`
- `build/current-mimo-v25-jangtq2-exactness-variant-probe-live-20260610/result.json`

Proven:

- JANGTQ_2 bundle loaded and served on the 128GB host.
- Model footprint is about `79.45 GiB` on disk; final health showed about
  `76.4GB` active memory in the earlier cache/text row.
- Native MiMo mixed full/sliding attention cache surfaced as
  `mixed_swa_kv_v1`, subtype `mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, q8 storage-boundary KV quantization, and
  block-disk L2 were active.
- Live cache proof observed paged cache hit and L2 block writes.
- Exact HTTP requests completed without hidden think tags.

Red:

- Exact output fidelity is not release-green. The same live server mutated:
  `blue-cat -> blue`, `B7-CAT-09 -> B7CAT-09` / `B7 CAT-09`, and tool args
  likewise lost characters.
- Short cache/text row also returned `ACKCB-742` instead of `ACK-CB-742`.

Next implementation target:

- Do not chase cache/parser/JSON repair for this red row. The current evidence
  says MiMo JANGTQ_2 cache plumbing works, but decode/artifact/logit exactness
  is wrong. Investigate artifact quant contract, codebook/decode path, or
  runtime token/logit contract.

### MiMo V2.5 JANG_2L

Artifact:

- `build/current-mimo-v25-jang2l-live-cb-cache-text-20260610.json`
- `build/current-real-ui-live-model-mimo-v25-jang2l-dev-app-proof-20260610.json`

Proven:

- 105 GiB local JANG_2L bundle loaded on 128GB host.
- Final health showed about `104997.8 MB` active and `105956.2 MB` peak.
- Uses `mlx_affine_quantized_matmul`; Metal NA active on host.
- Native MiMo mixed full/sliding cache surfaced as `mixed_swa_kv_v1`, subtype
  `mimo_v2_asymmetric_swa`.
- Prefix cache, paged cache, q8 storage-boundary KV quantization, and
  block-disk L2 were active.
- Paged cache hit observed: `cached_tokens=38`, `cache_detail=paged`.
- L2 block writes observed: `l2_tokens_on_disk=62`.
- Exact short output preserved: both repeats returned `ACK-CB-742`; first-token
  system probe returned `OK`.

Not proven:

- Required/auto tool call exactness in the dev-app row. The real app proof
  loaded the model and proved cache/L2, but failed the tool loop: no probe files
  were created and the second assistant turn had empty visible content.
- Responses stream/nonstream tool path for JANG_2L.
- Fresh-process L2 restore for JANG_2L.
- VL/audio/video runtime, even though media assets/weights are present.
- Long context usability beyond the short cache proof.

Next implementation target:

- Extend JANG_2L from the current passing short cache/text row into tool,
  Responses, fresh-process L2 restore, and media honesty. This is the best
  MiMo checkpoint candidate right now.
- For the dev-app red row, do not chase cache/L2: the failed app proof still
  showed `cached_tokens=3407`, `cache_detail=paged`,
  `l2_block_tokens_on_disk=3544`, and `disk_writes=57`. Next compare raw
  chat/tool output against panel dev-app chat trace to isolate parser/template
  selection versus model output quality.

### Nex/N2 Pro JANGTQ2

Artifact:

- `build/current-n2-jangtq2-live-chat-cache-responses-l2-20260610.json`

Proven:

- 101 GiB local JANGTQ2 bundle loaded and served on 128GB host.
- Final health showed about `104202.2 MB` active and `105212.9 MB` peak.
- Native cache is `hybrid_ssm_v1` with components:
  `attention_kv`, `ssm_companion_state`, `async_rederive`.
- Live attention TurboQuant KV is enabled only for attention KV layers;
  SSM companion state remains native.
- Tight-memory allocator drains occurred during prefill/decode.
- Chat cache proof passed with stable text `ACK`.
- Chat cache hit returned `cached_tokens=8`, `cache_detail=paged+ssm`.
- Required chat tool passed with args `{"query": "alpha"}`.
- Responses nonstream required tool passed.
- Responses tool-result follow-up with `previous_response_id` passed and
  returned `DONE`.
- Responses streaming required tool passed with args present across surfaces.
- Fresh-process L2 restart restore passed:
  `cache_detail=paged+ssm+disk`, block disk `disk_hits=1`, SSM companion disk
  `hits=1`.
- Final L2 totals: `l2_block_tokens_on_disk=271`,
  `l2_ssm_tokens_on_disk=271`, store sum `542`.

Not proven:

- Installed-app/UI path for this exact 20260610 proof.
- VL/audio/video.
- Same-model direct/gateway/tunnel public parity.
- JANG_1L profile.

Next implementation target:

- Treat JANGTQ2 as the N2 checkpoint candidate. It is the profile with real
  live 128GB cache/API/tool/L2 proof.

### Nex/N2 Pro JANGTQ2 Real Dev-App Proof

Artifact:

- `build/current-real-ui-live-model-n2-jangtq2-dev-app-proof-20260610.json`
- `build/current-n2-jangtq2-responses-stream-boundary-20260610.json`

Raw ignored proof captures:

- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-cache-20260610-proof.json`
- `docs/internal/agent-notes/current-real-ui-live-model-n2-jangtq2-responses-tools-cache-longdelta-20260610-proof.json`

Proven:

- Real Electron dev app launched with `npm run dev`, connected to a real vMLX
  server loading `/Users/eric/.mlxstudio/models/JANGQ-AI/Nex-N2-Pro-JANGTQ2`.
- The 101 GiB N2 JANGTQ2 profile loaded in the app proof harness; final health
  showed about `103807.6 MB` active and `108294.9 MB` peak in the longer
  attempt.
- Responses UI rail reached `/v1/responses` and completed two turns.
- Built-in `run_command` tool loop executed and wrote/read the expected probe
  files: `REAL_UI_LIVE_TOOL_ONE` and `REAL_UI_LIVE_TOOL_TWO`.
- Native cache surfaced as `hybrid_ssm_v1` / `hybrid_ssm_typed` with
  `attention_kv`, `ssm_companion_state`, and `async_rederive`.
- TurboQuant KV was active for attention KV layers only; SSM companion state
  stayed native.
- App/cache endpoints showed `paged+ssm` cache hits, block-disk L2 writes, and
  SSM companion disk stores. The longer attempt ended with
  `l2_block_tokens_on_disk=4626`, `l2_ssm_tokens_on_disk=23454`, and
  `l2_tokens_on_disk=28080`.
- Server cache controls were visible and verified in the dev app.
- No raw parser/tool markup leak and no hidden reasoning leak were observed.

Red:

- The dev-app proof did not clear `responses_delta_streaming`. Both attempts
  failed with `requested Responses API mode but proof did not record
  responses_delta_streaming surface`.
- The rerun used longer post-tool prompts, but the first post-tool visible
  answer still collapsed to `Created`; only the second assistant message
  produced multi-delta visible content (`count=31` in the longer attempt).
- The first prompted post-tool response did not include the requested
  `REAL_UI_LIVE_TOOL_ONE` phrase in visible content, although the tool file was
  written correctly.

Next implementation target:

- Raw direct server and panel gateway SSE are now proven green for N2 Responses
  tool plus tool-result continuation content deltas:
  `build/current-n2-jangtq2-responses-stream-boundary-20260610.json` is
  `status=pass`. Direct follow-up produced `16` output-text deltas and gateway
  follow-up produced `14` output-text deltas. Both completed on the same served
  model and returned `N2_DIRECT_DELTA_ONE` / `N2_DIRECT_DELTA_TWO`.
- This narrows the still-red dev-app proof to the chat renderer/tool-loop
  harness path or first post-tool answer behavior, not the core server or panel
  gateway SSE transport. Do not mark N2 dev-app Responses streaming green until
  the missing first-turn app stream trace is reproduced and fixed/proven.

## Red Live Attempts

### Nex/N2 Pro JANG_1L

Artifact:

- `build/current-n2-jang1l-live-chat-cache-responses-20260610.json`
- server log: `build/current-n2-jang1l-live-chat-cache-responses-20260610.server.log`

Observed:

- 110.59 GiB local bundle exists.
- Run was launched with `--jang1l-required-extra-headroom-gib 1`, so this was
  not a default preflight skip.
- Server startup began, model detection selected `qwen3_5_moe`, qwen parser,
  qwen3 reasoning parser, hybrid cache, and JANG text-only route.
- Quant shape inference patched 482 modules because config said uniform
  bits=2/group_size=128 while safetensors shapes required per-module overrides.
- Loader attempted 123 safetensors shards, bfloat16 for 512 experts, hidden
  4096.
- Loader set wired limit to the Metal cap: `Wired limit set to 115 GB (model
  119 GB)`.
- Process aborted before health with:
  `[METAL] Command buffer execution failed: Insufficient Memory`.

Conclusion:

- JANG_1L is not proven usable on this 128GB host. It is not merely untested and
  not merely skipped by a conservative gate; a live startup attempt crashed in
  Metal OOM before health.

Next implementation target:

- Implement an actual 128GB runtime strategy for JANG_1L before claiming it:
  lower peak loader/eval pressure, deferred/chunked eval that does not require
  full model command-buffer residency, smaller prefill/eval staging, or a
  JANG_1L-specific memory path. Do not claim N2 JANG_1L support from JANGTQ2.

## What To Tell The Other Agent

- Prioritize checkpoint release around proven rows: Gemma 12B MXFP4/JANG4M,
  MiMo JANG_2L short cache/text, and N2 JANGTQ2 full chat/cache/Responses/L2.
- Do not spend time proving generic cache. For MiMo use `mixed_swa_kv_v1`; for
  N2 use `hybrid_ssm_v1` with attention TQ KV plus native SSM companion.
- MiMo JANGTQ2 is loaded/cached but exactness-red; do artifact/logit/decode
  diagnosis, not parser repair.
- MiMo JANG_2L is the stronger MiMo checkpoint candidate; extend it to tools,
  Responses, fresh-process L2 restore, and honest media.
- N2 JANGTQ2 is the stronger N2 checkpoint candidate; it has live hybrid
  SSM/TQ/L2/tool/Responses proof.
- N2 JANG_1L needs a real memory-strategy fix. The current failure is a Metal
  OOM during loader/eval, not lack of attempt.
- Other agent should keep the new panel detector boundary: MiMo auto mode is
  XML tools + asymmetric-SWA paged cache, not auto reasoning; Gemma unified
  aliases must stay mapped to Gemma4 parsers and rotating mixed-SWA cache.
- Keep signed DMG release notes honest: say which profiles are checkpoint
  supported and which are experimental/red.
