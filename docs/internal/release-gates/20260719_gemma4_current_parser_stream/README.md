# Gemma 4 current-head parser, stream, and cache recheck

Date: 2026-07-19

Source head under test: `aa81d2170` plus the already-pushed source history ending in
`359ce6b2b`. The dev Electron app used the source checkout, not the stale packaged
Python bundle.

## Verdict

- `PASS-LIVE` for real Electron Start-button load, current config-derived parser and
  cache settings, Electron no-tool and tool-result turns, raw Responses and Chat
  reasoning/content streaming with adequate output headroom, required tool calls,
  clean continuation, and current focused tests.
- `PARTIAL` for the default stochastic reasoning-efficiency row: the short Electron
  no-tool prompt produced 3,322 output tokens and 15,629 reasoning characters before
  a coherent two-sentence answer. This was progressive and did not replay or leak,
  but it is too verbose to call efficient or reliability-closed.
- `EXPECTED LIMIT`, not a streamer pass, for the 512-token raw no-tool controls:
  Responses truthfully ended `response.incomplete` with
  `incomplete_details.reason=max_output_tokens`; Chat ended `finish_reason=length`
  with a truncated visible sentence. The otherwise-identical 4,096-token controls
  completed on both APIs.
- Overall release remains `PARTIAL/BLOCKED`: bundled `server.py` differs from current
  source and must be refreshed by `panel/scripts/bundle-python.sh` before the next
  packaging verifier/build/sign/notarize chain.

## Bundle and launch truth

Bundle: `/Volumes/EricsLLMDrive/jangq-ai/gemma-4-26B-A4B-it-qat-JANG_4M`

- `model_type=gemma4`, 30 text layers: 25 sliding-attention and 5 full-attention;
  sliding window 1,024; advertised context 262,144; vision config present.
- Weight runtime is affine `JANG_4M`, not JANGTQ/MXTQ and not base MLX MXFP.
- Electron Sessions-card Start loaded PID 4530 on port 8000 and stopped the prior M3
  process. Actual argv contains `--is-mllm`, `--tool-call-parser gemma4`,
  `--reasoning-parser gemma4`, prefix/paged cache, 64-token blocks, 1,000 blocks,
  block-disk L2, and stream interval 1.
- The current Electron main log records
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`
  before the UI-started session.
- Chat Settings visually showed Auto reasoning, Responses wire, built-in tools On,
  blank Max Tokens, temperature 1.0, and top-p 0.95.

## Cache boundary

Current health reports `mixed_swa_kv_v1` with full-attention KV, sliding-window KV,
and rotating-window metadata. It distinguishes two mechanisms:

1. Live flat generic TQ is disabled (`reason=storage_only`), so native rotating cache
   objects and their window metadata stay model-owned.
2. Stored prefix/block payloads use q4 TurboQuant at the storage boundary for both
   full and sliding KV, with `metadata_policy=preserve_rotating_window_metadata`.

After the live turns, scheduler health recorded 10 prefix hits, 7,950 tokens saved,
56 scheduler disk hits, and 239 native-TQ L2 hits. Electron row 397 restored 7,168
tokens as `paged+mixed_swa+disk` and exact-finaled the tool result.

Source trace:

- `vmlx_engine/utils/hybrid_tq_cache.py:139-168` validates the mixed layout and keeps
  sliding slots native for live cache operation.
- `vmlx_engine/utils/hybrid_tq_cache.py:180-234` wraps only compatible live
  full-attention slots and preserves companion/native cache classes.
- `vmlx_engine/server.py:8611-8673` reports the current mixed-SWA storage contract,
  including q4 storage for full/sliding KV and rotating metadata preservation.
- `vmlx_engine/tool_parsers/gemma4_tool_parser.py:124-313` owns native Gemma tool
  extraction and streaming state.
- `vmlx_engine/reasoning/gemma4_parser.py` owns the separate reasoning/content rail.

## Electron proof

- Row 394 no-tool: coherent non-empty visible answer, 15,629 distinct reasoning
  characters, no tool and no warning. CDP observed progressive DOM mutations and the
  final visible answer remained separate from the reasoning rail.
- Row 397 same-chat tool: exactly one real
  `file_info({"path":"panel/package.json"})`, real result `Size: 5.2 KB`, final
  `The file size is 5.2 KB.`, 211 reasoning characters, no warning, and 7,168
  `paged+mixed_swa+disk` cached tokens. CDP observed the answer advance through
  partial visible text before `5.2 KB.`.

Screenshots and persisted rows are included in this directory.

## Raw API proof

The six-case probe exercised Responses and Chat with Auto no-tool, required
`file_info`, and a real tool-result continuation.

At 4,096 tokens:

- Responses no-tool: 356 reasoning deltas, 44 content deltas, one
  `response.completed`.
- Responses tool: 116 reasoning deltas, one schema-valid call, completed terminal.
- Responses follow: 29 reasoning and 14 content deltas, completed terminal.
- Chat no-tool: 428 reasoning and 30 content deltas, `finish_reason=stop`, `[DONE]`.
- Chat tool: 116 reasoning deltas and two assembled tool fragments,
  `finish_reason=tool_calls`, `[DONE]`.
- Chat follow: 17 content deltas, `finish_reason=stop`, `[DONE]`.

The 512-token capture is retained to prove truthful limit handling rather than hiding
it: Responses emitted a complete-looking two-sentence fallback but correctly kept the
overall response incomplete because total output exceeded the requested cap; Chat's
bounded answer was visibly truncated and ended `length`.

## Tests

Current-source focused selection: 361 passed, with two unrelated librosa future
warnings. The selection covers Gemma tool/reasoning parsers, general reasoning and
streaming, answer-pass streaming, cache architecture/TQ contracts, VL media cache,
MLLM scheduler cache, and MLLM stream lifecycle.

## Retained open rows

- Classify/accept or improve Gemma's default-temperature long-reasoning reliability
  without hidden sampler clamps, prompt coercion, or fabricated output.
- Current-head alternate-video, larger-media, and advertised audio-family rows remain
  separate from this text/parser gate.
- Signed/notarized packaged-app repeat remains blocked on bundled-source refresh.
