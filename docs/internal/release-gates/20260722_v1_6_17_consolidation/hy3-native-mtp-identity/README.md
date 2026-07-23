# HY3 native-MTP identity and safe autoregressive checkpoint

Date: 2026-07-23

Source head: `adf92b5d431b41cb88b3f82cf891c5e0e36a71e7`

Host: `erics-m5-max.local`

Bundle:
`/Volumes/EricsLLMDrive/JANGQ-AI/Hy3-JANG_2K-MTP`

Status:
`SAFE AR PASS / MTP ACCELERATION VALIDATION-BLOCKED / UI EXACT-PUNCTUATION PARTIAL / OVERALL PARTIAL`

## Scope and artifact identity

This row covers HY3 `JANG_2K` affine weights. It is not JANGTQ/MXTQ
Hadamard-codebook quantization and it is not base MLX MXFP. The bundle is
text-only, declares 80 layers plus one MTP layer, and retains 42 MTP tensors.
Its model-derived generation defaults are temperature `0.9`, top-p `0.9`,
top-k `-1`, and min-p `0.05`. The current Electron process launched the
`hunyuan` tool parser and `qwen3` reasoning parser.

The ordinary attention cache is `plain_kv_v1`. Live health reports q4
TurboQuant block storage (`turboquant-storage`, four bits,
`hy3_full_kv_storage_tq4`) with block-disk L2. MTP artifact preservation and
MTP acceleration are separate facts: the MTP tensors remain detected, while
the unsafe accelerated verifier is blocked.

## Failure and root cause

With native MTP active, an exact direct Ollama two-tool flow produced
`AGENT斯-OLLAMA-STREAM-DONE ...`. With MTP Off, the byte-identical flow
produced `AGENTIC-OLLAMA-STREAM-DONE ...`.

The first two reasoning payloads were identical between those controlled
runs. Token inspection reduced the visible divergence to one substituted
token (`IC` versus `斯`), so it was not a reasoning parser, tool parser, API
adapter, or renderer rewrite.

The verifier performs one multi-row backbone forward over
`[next_main, draft]` and uses the second row as the full-accept bonus:

- `vmlx_engine/patches/mlx_lm_mtp/batch_generator.py:803-851`;
- `vmlx_engine/patches/mlx_lm_mtp/batch_generator.py:880-935`.

The shipped tuning sidecar declared a validated best depth, but did not attest
that two-row greedy affine verification is token-identical to ordinary one-row
autoregressive decode. A real bundle tensor probe confirmed that the two
shapes are not numerically identical:

- bf16: 13,604 unequal values, max absolute difference
  `9.5367431640625e-07`;
- fp16: 33 unequal values, max absolute difference `0.0009765625`;
- fp32: 14,822 unequal values, max absolute difference
  `2.384185791015625e-06`.

The numeric artifact is `hy3-affine-shape-identity.json`. It supports the
live token A/B; it does not by itself certify output quality.

## Fix

Commit `adf92b5d4` preserves the HY3 MTP artifact but blocks acceleration
unless `vmlx_mtp_tuning.json::native_mtp.output_equivalent` is explicitly
`true`:

- backend gate: `vmlx_engine/native_mtp.py:471-533`;
- backend status: `vmlx_engine/native_mtp.py:783-829`;
- Electron bundle detection and blocked reason:
  `panel/src/main/model-config-registry.ts:902-958`;
- visible settings warning and hidden unsupported controls:
  `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx:1597-1639`.

`VMLX_NATIVE_MTP_FORCE=1` remains a measurement-only override. No sampler
clamp, prompt rewrite, output repair, or synthetic fallback was added.

Focused current-head verification on the M5 Max:

- Python HY3 MTP and autodetection: `97 passed`;
- panel model-config registry: `94 passed`;
- panel typecheck: pass;
- `git diff --check`: pass.

## Real Electron proof

The isolated current-source Electron app used CDP `9335`. The real visible
Start button loaded HY3 as PID `33746` on port `8010`. Load completed healthy
without an error toast.

The launched argv included the exact bundle, `hunyuan` and `qwen3` parsers,
Paged RAM, q4 TurboQuant block storage, block-disk L2, and the model-derived
sampling defaults. It did not contain a hidden native-MTP enable or disable
flag. Backend health reported:

- `mtp.artifact_available=true`;
- `mtp.runtime_active=false`;
- `mtp.runtime_available=false`;
- `mtp.runtime_validation_blocked=true`;
- `mtp.status=runtime_validation_blocked`;
- `model_loaded=true`;
- `status=healthy`.

The Server settings UI visibly retained the MTP section and displayed:

> Native MTP weights were detected, but this HY3 affine bundle has not proven
> token-identical greedy output for its two-token verifier. Autoregressive
> decode remains active.

Retained screenshots:

- `r17-hy3-mtp-block-settings.png`;
- `r17-hy3-mtp-block-warning.png`.

Current-head UI turns then proved non-empty visible answers and separate
reasoning rails. A markup turn displayed literal currency `$43` and
`9×6=54` without raw control markup. A real built-in
`file_info(panel/package.json)` call executed once, consumed the real
`5.2 KB` result, and continued to a non-empty final. The model added a trailing
period to the requested exact final, so exact-copy compliance for that UI turn
is honestly `PARTIAL`; tool execution and continuation are `PASS`.

Retained screenshots:

- `r17-hy3-safe-ar-ui-markup.png`;
- `r17-hy3-safe-ar-ui-tool.png`.

## Direct and gateway streaming proof

The current-head allowlisted harness exercised four live streaming lanes:

- direct Anthropic;
- direct Ollama;
- Electron gateway Anthropic;
- Electron gateway Ollama.

Every lane completed three rounds:

1. separate reasoning, exact `file_info(panel/package.json)`, no visible prose;
2. separate reasoning, exact `run_command(pwd)`, no visible prose;
3. separate reasoning, progressive exact final, no tool call.

Every lane passed status, exact tool arguments, real result consumption,
reasoning/content separation, no control-marker leak, distinct/non-stale
reasoning, monotonic stream timing, progressive final output, and truthful
terminal-event classification. Anthropic ended the tool rounds as
`tool_use/message_stop` and the final as `end_turn/message_stop`. Ollama ended
tool rounds as `tool_calls` and the final as `stop`.

Evidence:
`r17-hy3-safe-ar-direct-gateway-anthropic-ollama.json`.

## Paged-On and Paged-Off SSD partial-prefix proof

The same exact bundle had already completed this current campaign's cache
axis:

- cold A: 7,183 prompt tokens, no hit;
- changed-tail B with Paged RAM On: restored 7,104/7,182 tokens;
- restart C with Paged RAM On: restored 7,104/7,186 tokens from SSD as
  `paged+disk+tq-native`;
- Paged RAM Off D: restored 7,104/7,188 tokens from SSD as
  `block-disk+tq-native`, with zero RAM cache;
- second Paged-Off restart E: restored the same 7,104/7,188 prefix from SSD.

All changed-tail turns prefilling only the uncached suffix reached the exact
requested final. The UI then restored Paged RAM On. These rows prove
longest-prefix partial block reuse, not arbitrary later-substring/suffix
splicing.

Evidence:

- `r17-hy3-l2-paged-on-a-cold.json`;
- `r17-hy3-l2-paged-on-b-partial.json`;
- `r17-hy3-l2-paged-on-c-restart-disk.json`;
- `r17-hy3-l2-paged-off-d-disk.json`;
- `r17-hy3-l2-paged-off-e-restart.json`.

## Boundaries that remain open

- The HY3 accelerated native-MTP implementation remains
  `VALIDATION-BLOCKED`; this change establishes safe AR behavior, not a fixed
  MTP fast path.
- The current source does not have a model-side
  `output_equivalent:true` attestation.
- Disk-cap eviction/refault was not exercised in this row.
- This row does not close other model families, media, full suites, signed
  installed-app repetition, package/sign/notarize, tagging, or publication.
- The dev log proves the exact project-venv Python path and imported
  `vmlx_engine 1.6.16`, but it did not emit the hook's exact
  `[Engine Manager] Found in PATH: .../.venv/bin/vmlx-engine` line. That exact
  provenance-log sub-requirement remains `FAIL`, even though PID argv, source
  head, bundle, UI warning, and live health were retained.

Overall v1.6.17 remains `PARTIAL / NOT RELEASE-READY`.
