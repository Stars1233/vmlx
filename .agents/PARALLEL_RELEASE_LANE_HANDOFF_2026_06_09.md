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
- `/Applications/vMLX.app` was rebuilt and installed from current source with
  `panel/scripts/build-and-install.sh`. The installed-app runtime parity audit
  now passes at
  `build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json`.
- Packaged integrity has green source/unit and bundled verifier checks in
  `build/current-packaged-integrity-contract-after-installed-app-rebuild-20260606.json`.
  Its older keychain-blocked wording is superseded by the signed-checkpoint audit below.
- Signed-checkpoint DMG readiness is now captured in
  `build/current-signed-checkpoint-dmg-readiness-20260609.json`. Existing local
  June 5 Sequoia/Tahoe DMGs are signed, stapled, and Gatekeeper-accepted, but
  they are not current-source proof for HEAD `d1054f41`. After following
  `/Users/eric/wiki/infra/apple-notarization.md`, fresh Developer ID signing is
  `pass`, notary history access is `pass` with
  `--keychain ~/Library/Keychains/vmlx-build.keychain-db --keychain-profile vmlx-notary`,
  and the keychain reports `no-timeout`.
- `panel/scripts/bundle-python.sh` now restores the Python standalone launcher
  immediately after extraction and again after MLX wheel installation, avoiding
  the intermittent missing `python3` / bootstrap `cp437` failure during app
  rebuild.

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
- MiniMax current-source Small JANGTQ tool/reasoning/cache/L2/TQ smoke is now
  audit-consumed and green, but #179 remains open for reporter K artifact,
  reporter generation-config/sampling, reporter bundle hash drift, and
  reporter-machine same-prompt raw SSE/visible/reasoning capture.
- No package, sign, notarize, tag, appcast, or public download update until
  runtime/model/UI/cache blockers are green or Eric explicitly overrides.
- The locally installed app is ad-hoc signed and valid on disk; do not call it a
  Developer ID signed or notarized checkpoint DMG.
- To produce a real current-source signed checkpoint DMG from this state,
  rebuild Sequoia/Tahoe DMGs, then run notarization with
  `VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db` and
  verify with `panel/scripts/verify-release-dmgs.sh`. Only redo the documented
  unlock plus `codesign` partition-list sequence if a fresh signing/notary probe
  regresses.
- Current official DMG build attempt:
  `VMLINUX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-pre-dmg-release-build-after-keychain-unlock-20260609.json panel/scripts/build-release-dmgs.sh sequoia`
  stopped at `--require-prepackage-ready` before packaging. The manifest reports
  `status=fail`, `prepackage_ready=false`, `release_ready=false`, while
  `packaged_app_developer_id_signing=true`. Treat the next blocker as release
  proof scope/model/API/UI rows, not signing access.
- Proper release mechanics are the documented path in
  `/Users/eric/wiki/infra/apple-notarization.md`; do not invent a GUI-only,
  ad-hoc-signing, cert-reimport, or verifier-weakening workaround. If signing
  returns `errSecInternalComponent`, rerun:

```sh
security unlock-keychain -p vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
security set-keychain-settings ~/Library/Keychains/vmlx-build.keychain-db
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k vmlx-release ~/Library/Keychains/vmlx-build.keychain-db
```

- The canonical current-source checkpoint sequence, once the prepackage ledger
  is green or Eric explicitly overrides it with listed open rows, is:

```sh
VMLINUX_PREPACKAGE_READY_MANIFEST_OUT=build/current-release-regression-manifest-pre-dmg-release-build-<scope>.json panel/scripts/build-release-dmgs.sh all
VMLINUX_NOTARY_KEYCHAIN=$HOME/Library/Keychains/vmlx-build.keychain-db panel/scripts/notarize-release-dmgs.sh
panel/scripts/verify-release-dmgs.sh
```

- `build-release-dmgs.sh` creates both Sequoia and Tahoe flavors from the
  current checkout and stops up front on `--require-prepackage-ready`;
  `notarize-release-dmgs.sh` submits and staples both final DMGs and regenerates
  blockmaps; `verify-release-dmgs.sh` is the final post-staple verification.

## Best Parallel Work Items

1. Capture same-model Responses raw SSE for direct local, panel gateway, and
   tunnel. Required proof: reasoning enabled without workaround, visible stream,
   `response.function_call_arguments.delta`, `.done`, final object consistency,
   valid output indices, and tool-result continuation.
2. Fix or recapture the Qwen35 tunnel output-index path. Current failing proof:
   `build/current-responses-raw-sse-parity-qwen35-tunnel-output-index-recapture-20260609.json`.
   Fresh tunnel raw capture:
   `build/responses-sse-captures-20260609/tunnel-qwen35-mxfp8-mtp-tool-recapture-max512-20260609.sse`.
   It preserves `record_fact` args `{"value": "blue-cat"}`, has reasoning
   events, and matches model `models/Qwen3.6-35B-A3B-MXFP8-CRACK-MTP`, but
   still emits both `message` and `function_call` at `output_index=0`.
   Follow-up source recheck `build/current-noheavy-api-cache-contract-after-qwen35-output-index-recheck-20260609.json`
   is `status=pass`, including source Responses streaming tool args/indexes and
   gateway argument passthrough. Treat the next Qwen35 action as live recapture
   or deployed/tunnel freshness unless a same-model current-source capture
   contradicts that.
   Current source audit: `stream_responses_api()` closes the message at
   `output_index=0`, increments, then emits function calls at the next index.
   Do not add a fake parser fallback or disable reasoning for this row. The
   next useful proof is Qwen35 current-source direct plus panel-gateway raw SSE
   with the exact tunnel request/model. If direct/gateway emit function calls at
   `output_index=1`, rebuild/redeploy the tunnel backend from current source and
   recapture. If direct/gateway also reuse `0`, reopen the source streaming path.
3. Run Gemma4 media/UI proof from current source or the rebuilt installed app.
   Installed-app runtime parity is now green, so the next useful Gemma work is
   real model/media/API/UI behavior, not another parity hash audit.
4. Continue MiMo on artifact/logit/quant contract or runtime decode. Keep media
   runtime truth separate from preserved-but-unwired media weights.
5. Recheck N2 `JANG_1L` memory preflight before any launch. If still below the
   guard, update status with the skip artifact rather than forcing a load.
   Latest continuation refresh:
   `build/current-n2-pro-jang1l-local-memory-preflight-continuation-20260609.json`
   is `decision=do_not_launch`, indexed payload `110.57 GiB`, required
   available `118.57 GiB`, current available `112.17 GiB`, gap `6.40 GiB`.
   It proves the model/index/config are present and identifies
   `qwen3_5_moe`/`JANG_1L`/`jang`, but no weights were loaded.
   Latest refresh after the DMG gate:
   `build/current-n2-pro-jang1l-local-memory-preflight-after-release-gate-20260609.json`
   is `do_not_launch` with payload `110.57 GiB`, required available `118.57 GiB`,
   available `112.56 GiB`, gap `6.01 GiB`.
   `build/current-n2-jang1l-chat-cache-proof-after-release-gate-20260609.json`
   is `status=skipped`, `reason=n2_jang1l_insufficient_available_memory`.
6. For MiniMax #179, do not rerun the already-green MiniMax Small source smoke
   unless source changes. Best help is reporter-machine K parity: model file
   manifest/hash, rendered prompt/template flags, resolved sampling kwargs, raw
   SSE visible/reasoning capture, and cancel lifecycle for the same response id.

## Current Dirty Files To Avoid Unless Owning That Lane

- `build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json`
- `uv.lock`
- `node_modules/`

The installed-app and packaged-integrity artifacts are now owned by the
installed-app/package parity slice and should be committed with the bundler fix.
