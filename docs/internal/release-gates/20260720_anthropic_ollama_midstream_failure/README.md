# Anthropic and Ollama mid-stream failure/recovery — 2026-07-20

Status: `FIXED_SOURCE_VERIFIED_LIVE_SCOPED` at pushed source commit
`d811270ad`. Public v1.6.12 remains sealed and does not contain this
post-release fix.

## Blocker and source trace

- Blocker class: `api/ui` protocol finalization.
- The shared OpenAI-compatible streams already emitted a structured error and
  then `[DONE]`. `AnthropicStreamAdapter` translated that error to a native
  `event: error` and suppressed normal finalization.
- All three Ollama converters dropped the structured error. Their route
  wrappers then interpreted the later `[DONE]` as success and synthesized a
  false `done:true` terminal.
- `vmlx_engine/api/ollama_adapter.py` now maps an upstream structured error to
  Ollama's native terminal NDJSON shape, `{"error":"..."}`, for chat,
  templated generate, and raw generate.
- `vmlx_engine/server.py` now marks that row terminal, clears deferred success
  and tool state, and suppresses all later usage/`[DONE]` success synthesis on
  `/api/chat` and both `/api/generate` rails.
- Official Ollama contract:
  <https://docs.ollama.com/api/errors>.

## Failing proof before the fix

The new converter regressions first failed 3/3:

```text
openai_chat_chunk_to_ollama_ndjson(error) -> None
openai_chat_chunk_to_ollama_generate_ndjson(error) -> None
openai_completion_chunk_to_ollama_ndjson(error)
  -> {"response":"","done":false,...}
```

This reproduced the false-success root cause without model-family inference.

## Live raw HTTP proof

The localhost proof server used the production Anthropic and Ollama handlers,
production adapters, and production Chat/Completions stream generators. Only
model inference was replaced by `FailureProofEngine`, which emits two delayed
visible chunks and then raises. Every failure call used literal
`curl -sS -N --no-buffer`; each was followed immediately by a recovery call.

Requests retained in `/tmp/vmlx-anthropic-ollama-midstream-requests.jsonl` at
run time:

```text
/v1/messages       ANTHROPIC FAIL -> ANTHROPIC RECOVER
/api/chat           OLLAMA CHAT FAIL -> OLLAMA CHAT RECOVER
/api/generate       OLLAMA TEMPLATED FAIL -> OLLAMA TEMPLATED RECOVER
/api/generate raw   OLLAMA RAW FAIL -> OLLAMA RAW RECOVER
```

Observed terminal excerpts:

```text
Anthropic failure:
  text_delta "ANTHROPIC-PARTIAL-"
  text_delta "VISIBLE"
  event: error / ANTHROPIC-MIDSTREAM-PROBE-FAILURE
  no message_delta or message_stop

Anthropic recovery:
  text_delta "ANTHROPIC-RECOVERY-"
  text_delta "OK"
  message_delta end_turn with 6 input / 2 output tokens
  message_stop

Ollama chat failure:
  {"message":{"content":"OLLAMA-PARTIAL-"},"done":false}
  {"message":{"content":"VISIBLE"},"done":false}
  {"error":"Stream generation failed: RuntimeError: OLLAMA-MIDSTREAM-PROBE-FAILURE"}

Ollama templated generate failure:
  {"response":"OLLAMA-PARTIAL-","done":false}
  {"response":"VISIBLE","done":false}
  {"error":"Stream generation failed: RuntimeError: OLLAMA-MIDSTREAM-PROBE-FAILURE"}

Ollama raw generate failure:
  {"response":"OLLAMA-RAW-PARTIAL-","done":false}
  {"response":"VISIBLE","done":false}
  {"error":"Stream generation failed: OLLAMA-RAW-MIDSTREAM-PROBE-FAILURE"}
```

No failing Ollama stream contained `done:true` after its error. Immediate
recoveries returned `OLLAMA-RECOVERY-OK` or `OLLAMA-RAW-RECOVERY-OK` followed
by exactly one `done:true` terminal with nonzero prompt/eval counts.

## Validation

```text
tests/test_streaming_adapter_failures.py
tests/test_ollama_adapter.py
tests/test_anthropic_stream_fixes.py
  30 passed

tests/test_server.py -k 'midstream_exception or streaming_responses_midstream'
  2 passed, 120 deselected

tests/test_engine_audit.py -k ollama_streaming_num_predict
  1 passed, 588 deselected

py_compile changed Python/harness files
  PASS

git diff --check
  PASS
```

## Scope boundary

This closes injected mid-stream engine failure and immediate recovery for the
production Anthropic adapter and all production Ollama streaming routes. It
does not claim gateway network loss, signed-app repetition, parser-family
agentic continuity, non-stream pre-header failures, or model/cache/media rows.
Those remain governed by the master matrix.
