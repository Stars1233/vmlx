# Parallel Release Lane Handoff - 2026-06-09

Active worktree only: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`.
Do not use deprecated `/Users/eric/vmlx` or old Swift notes unless Eric names
that path in the current turn.

## Current Green Source Rows

- Gemma4 QAT `JANG_4M` downloads are complete for E2B, E4B, 12B, 26B, and 31B.
- No-media source smokes pass for all five QAT `JANG_4M` rows after
  `82166ce0 Strip Gemma4 modality sentinel tool residue`.
- Covered in those smokes: visible text, multi-turn recall, required tool call,
  tool-result continuation, reasoning separation, exact JSON/code, mixed-SWA
  cache-hit telemetry, block-disk L2 write, and fresh-process L2 restore.
- 12B QAT `JANG_4M` visible `<audio|>` leak is fixed in the Gemma4 parser; the
  final server response guard in `eb511e92` is defense-in-depth only.

## Release Is Still Blocked

- Responses raw SSE parity is not green. Gemma4 E2B direct/gateway preserve
  args, but tunnel does not advertise the same model. Qwen35 tunnel preserves
  args and reasoning events, but reuses `output_index=0` for both message and
  function_call.
- Gemma full release still needs media/video/audio E2E, post-media text
  recovery, Responses content/tool-arg streaming, UI/CLI settings parity, and
  installed-app parity.
- MiMo still needs artifact/logit/quant or decode fix for exactness, plus
  JANGTQ/JANG_2L tool/cache/media/UI proof. Do not chase cache/L2/sampling as
  the primary exactness cause without new contradictory logits.
- N2 Pro 397B `JANG_1L` remains memory-gated. Do not lower the guard; rerun
  only when available RAM is at or above the preflight requirement or a real
  smaller-runtime strategy exists.
- DSV4 still needs memory-gated default-cache tool-loop proof.
- MiniMax Chinese/planning leak still needs cache/TQ KV/L2/parser/template and
  generation-config isolation.
- No package, sign, notarize, tag, appcast, or public download update until
  runtime/model/UI/cache blockers are green or Eric explicitly overrides.

## Best Parallel Work Items

1. Capture same-model Responses raw SSE for direct local, panel gateway, and
   tunnel. Required proof: reasoning enabled without workaround, visible stream,
   `response.function_call_arguments.delta`, `.done`, final object consistency,
   valid output indices, and tool-result continuation.
2. Fix or recapture the Qwen35 tunnel output-index path. Current failing proof:
   `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-20260609.json`.
   Follow-up source recheck `build/current-noheavy-api-cache-contract-after-qwen35-output-index-recheck-20260609.json`
   is `status=pass`, including source Responses streaming tool args/indexes and
   gateway argument passthrough. Treat the next Qwen35 action as live recapture
   or deployed/tunnel freshness unless a same-model current-source capture
   contradicts that.
3. Run Gemma4 media/UI/installed-app parity from current source only after
   verifying bundled Python includes the parser, response guard, cache, and
   model-register changes.
4. Continue MiMo on artifact/logit/quant contract or runtime decode. Keep media
   runtime truth separate from preserved-but-unwired media weights.
5. Recheck N2 `JANG_1L` memory preflight before any launch. If still below the
   guard, update status with the skip artifact rather than forcing a load.

## Current Dirty Files To Avoid Unless Owning That Lane

- `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json`
- `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
- `uv.lock`
- `node_modules/`

These are not part of the Responses checklist/output-index slice unless the
agent is explicitly working installed-app/package parity.
