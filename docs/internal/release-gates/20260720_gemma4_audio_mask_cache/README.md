# Gemma 4 Unified direct-audio mask and cache gate

Date: 2026-07-20
Host: `erics-m5-max.local`
Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Branch: `codex/postrelease-ui-drawers-20260720`
Starting source: `912df1c633f0619f3f4b10c06d54ce7db634de38`

## Verdict

| Axis | Verdict | Current source and live evidence |
|---|---|---|
| Bundle classification | PASS-SOURCE | The retained bundle files identify `gemma4_unified`, affine `JANG_4M`, encoder-free audio projection weights, 40 rotating-SWA layers, eight full-attention layers, audio+vision, and no advertised video. This artifact is not JANGTQ/MXTQ and not base MLX MXFP. |
| Audio capability advertisement | PASS-SOURCE+LIVE | `server.py::_bundle_declares_native_audio` now requires both artifact-owned `capabilities.modalities.audio=true` and real `embed_audio.*` weights for the direct projection route. The real engine advertised and ingested the WAV. Seven focused capability tests pass. |
| Continuous-batched audio math | PASS-SOURCE+LIVE | The pre-fix wrapper forwarded the processor's 2-D padding mask into causal language attention. A same-artifact logits A/B measured max absolute logit error `18.86328125`; omitting that mask was bit-identical to the direct mlx-vlm reference (`0.0`). `_language_model_mask` now drops only 2-D-or-lower processor padding masks and preserves explicit higher-rank masks. Nine focused scheduler/media tests pass. |
| Real Electron attachment, Thinking Off | PASS-LIVE | The real Electron composer at CDP 9335 visibly attached the WAV. The saved chat used Thinking Off and returned `The secret audio marker is Cobalt 7429. Repeat exactly Cobalt-7429.` as non-empty visible content: 26 tokens, 219 prompt tokens, 0.48 s TTFT, 1.0 s total. No reasoning rail, numeric loop, tool markup, warning, or truncation appeared. |
| Raw Responses streaming | PASS-LIVE | With both explicit TQ None and restored Auto q4, the same artifact emitted 21 progressive content deltas, `response.output_text.done`, and `response.completed`, with exact lowercase transcript and no warning. |
| Resident prefix/TQ reuse | PASS-LIVE | The identical warm request restored 218/219 tokens as `paged+mixed_swa+tq-native`, emitted the exact transcript, and completed in 0.747 s. |
| Process-restart L2/TQ restore | PASS-LIVE | After real Electron Save & Restart, L1 began empty while L2 retained eight blocks. The first identical request restored 218/219 tokens as `paged+mixed_swa+disk+tq-native`, emitted the exact transcript in 21 progressive deltas, and completed in 1.072 s. Health recorded four disk hits and four native-TQ hits with zero writes. |
| Auto-thinking quality/economy | PARTIAL | Auto no longer enters the pre-fix repeated `0.02e+19` collapse, but two real UI Auto turns overthought for 2,896 and 2,271 tokens; the second claimed the audio was absent. This is retained as a reasoning/default-quality issue, not promoted by the Thinking Off transport proof. |
| Generation-setting slider parity | PASS-LIVE scoped / PARTIAL family breadth | The real affine `JANG_4M` Gemma 4 26B bundle declares temperature 1.0, top-p .95, and top-k 64; it declares no min-p or repetition penalty. Persisted session detection, the real Electron drawer, omitted-override payload, and engine resolution agree on 1.00/.95/64/0.00/1.00. The decisive saved Thinking-Off/tools-Off turn painted partial content before exact completion. JANGTQ/MXTQ, base MLX/MXFP, typed routes, and a non-neutral repetition-penalty artifact remain separate. Evidence: `../20260721_gemma4_26b_sampling_parity/`. |

## Root cause

The direct mlx-vlm generation path uses the processor's 2-D attention mask only
while building media-conditioned embeddings and invokes the Gemma language
model without that mask. The continuous-batching wrapper incorrectly reused the
same 2-D padding mask as the language model's attention mask. That changed
first-step logits and collapsed audio decode into repeated `thought`/numeric
tokens regardless of TQ, prefix cache, or parser settings.

The fix is contract-level and not an output rewrite: ordinary processor padding
masks are not forwarded to causal language attention; explicitly constructed
higher-rank masks remain supported.

## Validation

- `tests/test_engine_audit.py -k "gemma4_unified and audio"`: 7 passed.
- `tests/test_mllm_scheduler_cache.py -k "gemma or audio"`: 9 passed, 99
  deselected; two unrelated librosa deprecation warnings.
- `git diff --check`: passed before staging.
- Same-artifact direct/wrapper logits A/B: bad wrapper max error
  `18.86328125`; fixed wrapper/direct error `0.0`.
- Real Electron UI, raw Responses explicit None, raw Responses Auto, resident
  warm cache, and process-restart L2 rows all ran on current source.

## Evidence files

- `g4-audio-maskfix-final.png` — real Electron WAV attachment and Thinking Off
  visible result.
- `g4-ui-audio-maskfix-off-cold-observe.json` — UI progressive/terminal timing.
- `g4-api-maskfix-notq.jsonl` and `g4-api-maskfix-auto.jsonl` — raw streaming
  negative-control and restored-Auto rows.
- `g4-audio-maskfix-warm.jsonl` — resident paged/mixed-SWA/native-TQ hit.
- `g4-health-maskfix-restart-before.json`, `g4-audio-maskfix-disk.jsonl`, and
  `g4-health-maskfix-disk-after.json` — process-restart L2 promotion proof.
- `gemma-audio-logits-ab.json` and `gemma-audio-direct-match.json` — numeric
  root-cause A/B.
- `g4-audio-ui-final.png` — retained pre-fix repeated-output failure.
- `bundle-*.json` — exact bundle-owned configuration evidence.
- `test-engine-audit.xml` and `test-mllm-cache.xml` — focused test reports.

## Remaining boundaries

- Retain model-derived sampling breadth for JANGTQ/MXTQ, base MLX/MXFP,
  DSV4/M3 typed routes, and a non-neutral repetition-penalty artifact. The
  affine Gemma UI/payload/runtime chain is closed at
  `../20260721_gemma4_26b_sampling_parity/`.
- Investigate Gemma Auto-thinking economy/attachment awareness without hidden
  sampler clamps, prompt coercion, output rewriting, or forced thinking modes.
- Keep exact OCR/strict-format, bounded eviction, non-advertised-video
  rejection, broader protocols, and signed-app repetition in their existing
  rows.
