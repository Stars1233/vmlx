# N2 Pro JANGTQ2 Proof Intake - 2026-06-12

## Source Proof

Imported from JANG agent proof:

- `/Users/eric/jang/build/current-n2-jangtq2-self-runtime-20260612/SUMMARY.md`
- `/Users/eric/jang/build/current-n2-jangtq2-self-runtime-20260612/coherence_probes.json`
- `/Users/eric/jang/build/current-n2-jangtq2-self-runtime-20260612/server.log`

## Working Artifact

`/Volumes/EricsLLMDrive/jangq-ai/Nex-N2-Pro-JANGTQ2-20260611`

Size: `101G`

Classification: coherent working N2 Pro source-runtime lane.

## Proven

- Source vMLX runtime on port `8140`.
- MLLM/VLM JANGTQ native TurboQuant fast path.
- No quant-shape repair warning.
- Exact text: `blue cat`.
- Arithmetic: `45`.
- Exact JSON: `{"ok":true,"n":45}`.
- Responses no-tool sentinel: `N2_NO_TOOL_OK`.
- Responses arithmetic: `54`.
- Responses auto-tool function call: `lookup` with args `{"query":"n2-self-742"}`.
- Responses `previous_response_id` continuation: `N2_TOOL_OK_742`.
- Hybrid paged+SSM cache reuse: `209` cached continuation tokens,
  `219` total cache-hit tokens.
- Final health: active `103802.3 MB`, peak `104874.8 MB`,
  generation throughput `34.27 tok/s`.

## Not Proven By This Intake

- Chat `logprobs/top_logprobs` diagnostics on this MLLM/JANGTQ path; the proof
  recorded HTTP 400 for that request shape.
- UI/installed-app parity.
- Public tunnel/gateway parity.
- Release packaging, signing, notarization, upload, or updater/download state.
- MTP runtime; proof says MTP metadata is present but weights are missing and
  runtime skipped it.

## Release Boundary

Use this JANGTQ2 artifact as the current coherent N2 Pro model lane. Do not use
the failed affine JANG_1L row-L2 prune15 path as a production lane. This source
proof can feed release planning, but it is not by itself a signed/notarized
release gate.
