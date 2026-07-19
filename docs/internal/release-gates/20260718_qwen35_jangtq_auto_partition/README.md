# Qwen 3.6 35B JANGTQ Auto reasoning/output partition

Status: `PASS-LIVE` for ordinary Auto reasoning/content streaming on the
current post-1.6.11 source. Explicit native-tool reliability remains
`PARTIAL-STOCHASTIC`; malformed calls are rejected rather than repaired with
invented arguments.

## Artifact identity

- Bundle: `dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- Live path: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- Weight format: `mxtq`
- Profile: `JANGTQ2`
- Routed expert bits: 2
- Runtime weight codec: `turboquant_codebook`

This is a JANGTQ/MXTQ Hadamard-rotation/codebook artifact. It is not affine
JANG and it is not a base MLX MXFP artifact. Cache quantization is a separate
axis: the live session stored the hybrid model's attention-KV component as
TurboQuant q4 while retaining native SSM companion state.

## Root cause and source trace

An attached `tool_choice=auto` catalog disabled the Qwen Auto reasoning/output
partition even when the user explicitly requested no tool. The native
reasoning rail could consume the full output cap, leaving little or no visible
answer. The shared server now uses
`_auto_thinking_partition_allowed(...)` for Qwen Auto families:

- ordinary Auto requests may reserve a bounded visible-answer share;
- required, named, or explicitly requested tool turns remain unpartitioned;
- an ordinary no-call pass may use the established tools-free visible-answer
  fallback only after final parsing found no schema-valid call;
- explicit tool turns remain fail-closed and never become a fabricated
  tools-free answer.

The policy is applied to streaming and non-streaming Chat Completions and
Responses. Regression coverage is in
`tests/test_qwen3_answer_pass_policy.py`; the affected focused matrix passed
101 tests with 105 deselected, plus `py_compile`.

## Live Electron proof

The source was installed into the live dev tree and the real Electron Server
panel performed `Save & Restart`, replacing PID 61979 with PID 63899 on port
8029.

- Row 126, ordinary Auto: 3,773 reasoning characters remained separate, then
  visible content completed exactly as three non-empty lines:
  `REASONING-SEPARATE`, `CONTENT-PROGRESSIVE`, `TERMINAL-COMPLETE`.
  Metrics reported 959 output tokens and no warning.
- Same-chat row 129, explicit tool: one real
  `file_info({"path":"panel/package.json"})` call and one matching 5.2 KB
  result were persisted. The UI progressively painted the exact final
  `Q35-JT-FINAL-TOOL-DONE SIZE=5.2 KB`, with no warning. It restored 325
  prompt tokens as `paged+ssm+disk`.

Screenshots:

- `q35-jt-final-auto.png`
- `q35-jt-final-tool.png`

## Raw Responses streaming proof

On the same PID, an ordinary Auto request with a real `file_info` schema but
an explicit no-tool instruction emitted 237 separate reasoning deltas, then
14 progressive content deltas over about 0.15 seconds, and exactly one
`response.completed` terminal. The visible marker completed without a tool.

The exact explicit-tool prompt was then repeated 12 times: four at temperature
0 and eight at the UI-like temperature 1.0/top-p 0.95/top-k 20. Every sample
emitted exactly one schema-valid `file_info` call with
`{"path":"panel/package.json"}` and one completed terminal.

Raw Chat Completions on the same PID also stayed incremental. Its ordinary
Auto control emitted 1,024 reasoning deltas followed by 353 content deltas,
`finish_reason=stop`, and one `[DONE]`; the model added a long explanation
before the requested marker, retained as a strict-format miss. The explicit
tool pass emitted 58 reasoning deltas, one valid `file_info(path)`,
`finish_reason=tool_calls`, and `[DONE]`. Supplying the real tool result in
Chat history then emitted 152 new reasoning deltas and 17 content deltas,
exact-finaled `Q35-JT-CHAT-CONT-DONE SIZE=5.2 KB`, and ended with `stop` plus
one `[DONE]` without calling the tool again.

## Retained negative controls

The release evidence must not erase the native variability:

- Electron row 63 produced 581 reasoning characters, no visible answer, and a
  `file_info` candidate missing required `path`; the validator dropped it.
- A distinct raw API wording reproduced the same missing-`path` native
  candidate on final source. No function executed and the response retained
  explicit warnings.
- Repeated same-chat row 69 reused the prior result without a new call even
  though the prompt requested a call.
- Repeated same-chat row 72 eventually called the tool and exact-finaled, but
  only after 52,343 reasoning characters / 16,404 tokens / 214.5 seconds.

Those controls, alongside 11/11 fresh Electron tool passes, 3/3 fresh
Auto-then-tool pairs, the final-source Electron pair, and 12/12 exact raw API
calls, classify the missing argument as prompt/history/sampling-sensitive
native model emission rather than a deterministic parser or transport defect.
The application behavior is still only partial for repeated-tool soak: it
fails closed safely, but a coding harness can still stop when the native model
does not emit a valid call. No hidden thinking-off retry, sampler clamp,
synthetic call, or guessed argument was added.

## Evidence files

- `electron-final-and-controls.json`
- `responses-stream-and-tool-sampling.json`
- `health-quant-cache-summary.json`
- `q35-jt-final-auto.png`
- `q35-jt-final-tool.png`
