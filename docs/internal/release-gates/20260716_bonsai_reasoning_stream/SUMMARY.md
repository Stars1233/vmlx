# Bonsai reasoning and Responses stream gate - 2026-07-16

Scope: `jangq-ai/Bonsai-27b-1bit-JANG` through the current Python engine and
live Electron dev app on `erics-m5-max.local` (CDP `127.0.0.1:9335`, engine
`127.0.0.1:8030`). This is a scoped parser/UI gate, not release clearance.

## Verdict

| Surface | Verdict | Current evidence |
|---|---|---|
| Responses speculative text buffer | PASS | Source recovery in `panel/src/main/ipc/chat.ts`; repeated zero-tool soak rows restored exact final text; screenshot and screen logs in this directory. |
| Tools-enabled reasoning-only fallback | PASS | Source gate in `vmlx_engine/server.py`; engine audit regression and live exact low-cap rows 1776/1779/1782/1785. |
| Exact-once built-in tool continuation | PASS | Electron rows 1791, 1794, and 1797 each persisted one `file_info`, one result, and an exact final marker. |
| Terminal reasoning completeness | PASS | Row 1797 persisted/rendered the complete 158-character terminal summary; raw tool-control markup is rejected by the panel helper and tests. |
| General Bonsai pre-tool latency | PARTIAL | Row 1788 required 2,128 output tokens and 62.5s. Repeats on fresh and reused chats took 327, 316, and 87 total tokens; a direct one-tool Responses trace took 111. The outlier is stochastic and not closed as normal behavior. |
| TQ/cache health truthfulness | PARTIAL | Health shows 82 native-TQ block writes and 393 native-TQ hits, but top-level `kv_cache_quantization.enabled=false`. SSM disk restore is suppressed, so no restart-SSM reuse claim is made. |
| Broad release | BLOCKED | Remaining family, cache, protocol, media, UI-parity, performance, package, signing, and notarization gates remain open. |

## Source trace

- `panel/src/shared/responsesStreamRecovery.ts` and its call sites in
  `panel/src/main/ipc/chat.ts` clear only a speculative zero-tool buffer and
  restore authoritative final Responses text.
- `panel/src/shared/interleavedReasoning.ts` adopts a longer terminal reasoning
  summary only when it extends the streamed prefix and contains no Qwen,
  MiniMax, or Harmony tool-control syntax.
- `vmlx_engine/server.py` preserves the tool-capable first pass, then re-arms a
  bounded tools-free visible-answer pass only when final parsing finds no call
  and no visible answer.
- `tests/test_engine_audit.py`,
  `panel/tests/responses-stream-recovery.test.ts`,
  `panel/tests/interleaved-reasoning-segments.test.ts`, and
  `panel/tests/tool-status-responsiveness.test.ts` cover the changed contracts.

## Live verification

- Row 1797 (`B1-REASON-DONE-LIVE1`): 2,535 prompt tokens, 87 total output
  tokens, 48.8 tok/s, 9.2s, full 158-character reasoning, one call/result,
  exact final.
- Row 1794 (`B1-LONGCHAT-UI-TOOL2`): reused long chat, 316 total output
  tokens, 13.3s, one call/result, exact final.
- Row 1791 (`B1-FRESH-UI-TOOL1`): fresh chat, 327 total output tokens, 9.1s,
  one call/result, exact final.
- Row 1788 is retained as the latency outlier: 2,128 tokens and 62.5s despite
  one valid call/result and exact final.
- `b1-raw-sse1.txt`: raw direct Responses stream emitted reasoning deltas,
  tool-buffer heartbeats, one valid function call, and completed at 111 output
  tokens.

## Tests

- `pytest-engine-audit.txt`: 580 passed.
- `panel-focused-tests.txt`: 24 passed.
- `panel-typecheck.txt`: TypeScript typecheck passed.
