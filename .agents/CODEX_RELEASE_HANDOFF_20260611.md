# vMLX Python Engine Release Handoff - 2026-06-11

## Current Commit

- Latest movement commit: `Classify MiMo speed root cause` (see `git log`
  for the final pushed hash)
- Pushed to: `origin/codex/pr-intake-manifest` and `origin/main`
- Latest checklist artifact:
  `build/current-full-release-objective-checklist-after-mimo-speed-root-cause-classification-20260611.json`
- Latest checklist state: `status=open`, `failed_count=15`

## Hard Boundaries

- Do not work from deprecated `/Users/eric/vmlx`.
- Do not spawn/manage subagents from Python, shell, MCP, or wrappers.
- Do not touch Nex/N2 `JANG_1L` unless Eric explicitly reopens it in the
  current turn.
- Do not sign, notarize, tag, upload, PyPI-release, update downloads, or touch
  site/updater release paths unless Eric explicitly unlocks release actions in
  the current turn.
- Do not fake parser/cache/media fixes: no synthetic tool args, no disabling
  reasoning to hide failures, no metadata-only audio/video claims, no parser
  repair of model-generated literal mutations.

## Proven In Latest Movement

- Nemotron Omni current all-local JANGTQ proof is consumed by the release
  checklist:
  `build/current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/dealign.ai_Nemotron-Omni-Nano-JANGTQ-CRACK/result.json`
- Nemotron proof covers: text cache repeat, reasoning-on visible output,
  required `record_fact` tool call with `{"value":"blue-cat"}`, tool-result
  continuation, JSON/code/whitespace exactness, blue image, blue video, blue
  audio, post-media text recovery, native hybrid SSM cache, paged+SSM reuse,
  block L2 writes/hits, and SSM L2 writes.
- Qwen35 direct/gateway source proof remains good, but direct/gateway/tunnel
  parity is not release-green because the fresh public tunnel recapture is
  still stale:
  `build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-public-recapture-still-stale-20260611.json`
- Qwen35 public tunnel still preserves model, authoritative
  `{"value":"blue-cat"}` arguments, function-call argument deltas/done, final
  response consistency, and valid message/function output indices, but it lacks
  complete streamed reasoning output item lifecycle. Keep this row red until
  the tunnel runtime is rebuilt and recaptured.
- N2 JANGTQ2/non-JANG_1L checkpoint profile is now explicit in the checklist
  and green for source runtime/API/cache, fresh-process L2 restore, Electron UI
  previous_response_id/tool/cache, strict loopback `tool_choice=auto`,
  direct/gateway Responses stream boundary, and no-heavy policy/cache
  contracts. This does not clear N2 JANG_1L.
- MiMo installed-app proof accounting is now explicit and green for:
  JANGTQ2 Responses tool/cache with block L2, JANG_2L Responses tool/cache with
  block L2, and JANGTQ2 installed-app media transport with block L2. The
  checklist intentionally keeps MiMo red for exact literal quality, decode
  speed, media semantics, audio/video release quality, and JANG_2L live media
  L2.
- MiMo JANG_2L media/L2 is now classified as not applicable when the metadata
  contract proves `weights_preserved_text_runtime` with text-only runtime and
  preserved/unwired media weights. Do not keep treating it as missing live media
  proof unless a future artifact changes JANG_2L to a real media runtime.
- MiMo decode-speed proof is now traced into the checklist instead of being a
  generic red row. The current speed row is still red, but it now carries:
  `build/current-mimo-v2-jang2l-current-audit-after-speed-root-cause-classification-20260611.json`
  with `bundle_decode_tps=39.2`, `wall_decode_tps=39.12980168982385`, and root
  cause evidence from
  `build/current-mimo-jang2l-speed-cache-root-cause-20260611/SUMMARY.json`.
  That root-cause summary says body decode/cache are not the primary current
  bottleneck; classic JANG_2L is logits/lm_head materialization-bound, with
  traced body model step around 3.1 ms and logits materialization around
  2532.17 ms on token 1 / 606.22 ms on token 2 after warmup. Do not chase
  prefix cache, L2, RotatingKVCache metadata, or parser/tool code as the MiMo
  speed fix unless new traces contradict this.

## Still Open

- `prepackage_ready` and `release_ready`: intentionally false while blocker
  rows remain open.
- `n2_pro_397b_release_clearance`: dominated by N2 `JANG_1L`, which is
  Eric-owned/off-limits for this lane. Do not claim or launch it here.
  The N2 JANGTQ2 profile has explicit green rows; keep broad release clearance
  red until JANG_1L is reopened and proven.
- MiMo: exactness/media semantics/speed remain open. Current classification
  says JANGTQ_2 literal mutations happen after valid parser structure; do not
  clear by rewriting parsed tool arguments or repairing generated JSON. Do not
  keep chasing JANGTQ2 missing tool/cache/L2 as the primary blocker: the current
  installed-app rows prove those surfaces separately. JANG_2L media/L2 is not
  applicable for the current text-runtime artifact. The remaining useful MiMo
  work is artifact/logit/quant/decode quality, visual semantics, audio/video
  release semantics, and the specific speed path above. For speed, either prove
  sustained current-source/current-installed JANGTQ_2 text runtime over the 40
  tok/s floor, or optimize the logits/lm_head/materialization path. Do not lower
  the speed floor or count short forced-MLLM overlay samples as the sustained
  text-runtime speed proof.
- Qwen35 raw SSE direct/gateway/tunnel parity remains open on public tunnel
  reasoning lifecycle, even though current direct/gateway source proof is
  green.
- MiniMax issue179: current source and local installed probes are clean, but
  reporter parity is not proven. The current audit is intentionally open
  because reporter artifact/session evidence is missing and reporter installed
  server hash differs from the latest public/local server hash.
- DSV4: memory/exactness preflight is skipped/open due insufficient available
  memory for that gate.

## Other-Agent Next Best Work

- If deploy/tunnel access exists, refresh exact-served-model raw SSE parity
  artifacts only when the served route really advertises and serves the target
  model; do not pointer-refresh to a stale or partial capture.
- For MiniMax issue179, collect reporter parity metadata/artifacts: reporter
  model shard/codebook hashes, installed app server hash, chat/session/settings
  DB state, response active-state at cancel time, raw SSE/cancel lifecycle, and
  original prompt/log ordering. Without that evidence, keep #179 open.
- For MiMo, focus on artifact/logit/quant contract or runtime decode diagnosis.
  Current parser/cache/L2 evidence does not support a parser or cache-only fix.
  If optimizing speed, start from the traced logits/lm_head materialization
  bottleneck and the JANGTQ_2 39.2 tok/s near-miss; do not reopen already-green
  JANGTQ2 tool/cache/L2 or JANG_2L media/L2 rows.
- For release packaging, first rerun the bundled Python parity gate and
  installed-app source-vs-bundle checks after source rows are worth packaging;
  then follow the documented signing/notarization workflow only after Eric
  unlocks release actions.
