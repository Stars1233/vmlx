# Bonsai 1-bit post-dispatch streaming and q8 hybrid-cache proof

Status: `PASS-LIVE` for current-head raw Chat/Responses streaming, Electron
multi-turn tool continuation, q8 attention-KV plus native SSM RAM reuse, and
process-restart Block Disk L2 restore. Status remains `PARTIAL` for strict
sampled byte formatting and long native reasoning reliability; this is not a
release claim.

## Artifact and source trace

- Artifact: `jangq-ai/Bonsai-27b-1bit-JANG`, visibly loaded through Electron on
  port 8030. Single-model mode stopped the prior Qwen process and left Bonsai
  PID 94843 as the only engine. A later visible Save & Restart produced PID
  95400 without clearing L2.
- `/health` derives `qwen3_5` hybrid state from the real runtime graph: 16
  attention-KV layers and 48 native companion layers. Bonsai's explicit Auto
  exception stores only eligible attention KV as TurboQuant q8; companion state
  remains native with clean async prefill rederive or a matching SSM checkpoint.
- Commit `aa6a3d2ef` is the shared async terminal-dispatch repair. It dispatches
  and yields terminal output before cache/TQ/SSM persistence in both LLM and
  MLLM schedulers, without a Bonsai/model/parser special case.

## Raw Chat and Responses stream

- `bonsai-postdispatch1.json`: Chat turn 1 emitted 363 reasoning and 14 content
  deltas; turn 2 in the same conversation emitted 512 reasoning and 22 content
  deltas, recalled its exact codeword, and reused 46 `paged+ssm` tokens.
  Responses emitted 512 reasoning and 11 `response.output_text.delta` events,
  matching `output_text.done`, and one `response.completed`.
- Turn 1's assembled Chat bytes contain two leading newlines before the exact
  marker; turn 2 and Responses are exact bytes. The harness's trimmed semantic
  checks passed all three, but the retained leading whitespace keeps strict
  sampled byte-format reliability `PARTIAL`.
- The visible answer spans were 0.250s, 0.397s, and 0.200s respectively. At
  Bonsai's 40-50 tok/s decode rate this can look nearly instantaneous, but the
  raw timestamp arrays prove distinct SSE deltas rather than one terminal blob.

## Electron agent loop

- `bonsai-postdispatch-electron1.json` captured a real turn in an existing
  Bonsai chat: 51 reasoning paints, 12 visible-content paints, exactly one
  `file_info(panel/package.json)`, one result, and exact
  `B1-POSTDISPATCH-ELECTRON1-DONE` with no warning.
- `bonsai-postdispatch-electron1.png` visibly shows the reasoning panel, real
  Info card, exact final answer, and metrics. The SQLite extract persists the
  same OpenAI function call and matching result.
- A prior `how are u` turn in that chat restored only 320 of 3,278 prompt
  tokens, so its 9.25s TTFT includes roughly 2,958 uncached tokens. It is not
  used as a fair full-prefix performance comparison.

## Identical long-prefix RAM and restart-L2 proof

- `bonsai-postdispatch-tqfair1.json` uses one exact 4,631-token prompt with
  thinking off. Cold first content was 40.3291s. The identical resident request
  restored 4,630 `paged+ssm` tokens, first content in 0.6969s (57.869x), and
  streamed all 15 visible deltas through 0.9662s. Worker reconstruction was
  0.281451s.
- After visible Electron Save & Restart from PID 94843 to 95400,
  `bonsai-postdispatch-tqfair1-l2.json` restored all 4,630 tokens as
  `paged+ssm+disk`, first content in 1.5169s, and streamed 15 deltas through
  1.7855s. The following RAM restore reached first content in 0.6716s and last
  content in 0.9390s.
- `bonsai-postdispatch-health-after-l2.json` records two hits / 9,260 tokens
  saved, 292 native-TQ q8 block hits, one real SSM disk hit, attention-only q8,
  native companion policy, and zero unsafe KV-without-SSM reuse.

## Release boundary

This is a current-head regression proof for the exact family that originally
showed reasoning-then-batched-answer behavior. It closes progressive streaming
and q8 RAM/L2 ownership for this artifact. It does not erase the retained
leading-newline sample, historical long/variable reasoning, broader Bonsai
variant matrix, cancellation/soak, or final aggregate release gates.
