# Reasoning / streaming / sampler proofread

Date: 2026-07-11

Source reviewed: `edb0fac4246f176138250d49983da6eac8603bdc` (`origin/main` at intake), including `edb0fac42`, `bdaee1d4f`, `dc8e1600d`, and the H1 changes in `5520b1759`.

## Findings, ranked

### Critical — answer-pass starvation remains open

The strict C1 draw-down respects the client cap, but it also makes the answer pass unreachable in the exact runaway case it is supposed to recover. `_remaining_answer_pass_budget(cap, used)` returns zero when the first reasoning pass consumes the cap. Every non-stream and stream answer-pass guard requires the result to be positive before it launches the retry. A MiniMax-M2/M3, Gemma, Qwen3.5/3.6, or openPangu turn that spends all `max_tokens` in reasoning therefore remains empty; adding `minimax_m2` at `edb0fac42` does not recover a cap-exhausted turn.

Evidence:

- `vmlx_engine/server.py:1061-1071` clamps remaining budget to zero.
- Chat non-stream: `server.py:12497-12519`.
- Responses non-stream: `server.py:14644-14666`.
- Chat stream: `server.py:16366-16409`.
- Responses stream: `server.py:17854-17895`.
- `tests/test_output_budget_cap.py` explicitly asserts `budget(64, 64) == 0`.

The previous 32-token floor violated the client cap; the current behavior avoids that violation but does not reserve answer capacity. A real resolution needs an explicit within-cap reservation policy, not a hidden overage.

### High — Ollama streaming skipped Hy3's reasoning dialect (fixed in source)

Chat, Responses, and Anthropic applied `_apply_hy3_reasoning_policy`; Ollama's locally built streaming kwargs did not. Hy3 ignores `enable_thinking` and needs `reasoning_effort=low|high`, so `think:true` caused parser/prompt disagreement. Live pre-fix results put the short final answer in `message.thinking`, emitted a one-delta diagnostic notice as visible content, and omitted terminal usage.

Fix: apply the same Hy3 normalizer in the Ollama streaming handler before family-specific DSV4 processing (`server.py:10046-10056`). This is source/unit proven only; the already-running 105GB process was deliberately not restarted.

### High — H1 moved repetition processing after normalization (fixed)

H1 correctly normalized generic stochastic sampler input once, but the request wrapper then applied logits processors to normalized log-probabilities. MLX-LM's contract is raw logits → logits processors → one log-softmax → generic sampler. Repetition penalty is sign-sensitive and is not invariant to subtracting log-sum-exp, so this changed token selection.

Fix: the wrapper now owns the raw-logit transition, applies processors first, normalizes exactly once only for logprob samplers, and advertises raw-logit acceptance to every prefill/decode/MTP caller (`mllm_batch_generator.py:7053-7074`). Greedy remains exact argmax of processed logits.

### High — seeded requests shared the first request's PRNG state (fixed)

`_batch_shares_sampler_params` did not include `seed`. Two request-local seeded rows with otherwise equal sampler parameters could use the first request's mutable sampler state. Seeded requests now bypass the shared sampler fast path (`mllm_batch_generator.py:726-746`).

### High — chat streaming could emit two visible fallbacks (fixed)

The legacy reasoning-as-content fallback ran immediately before the bounded answer pass and did not set `content_was_emitted`. When reasoning was accumulated but not incrementally streamed, both blocks could emit. The legacy fallback is now disabled whenever either answer-pass policy is armed (`server.py:16337-16350`).

### Medium — suppressed-reasoning cleanup dropped legitimate visible prefixes (fixed)

`_strip_residual_think_markup_for_display("Visible</think> tail")` returned `"tail"`. The display path no longer assumes everything before an orphan close marker is hidden reasoning; it strips markers while preserving visible text (`server.py:815-832`).

### Medium — Ollama greedy warm determinism failed live (open)

The two measured warm, `temperature=0`, seeded Ollama requests returned `"DET-731"` (4 tokens) and `"DET-731."` (5 tokens). Chat, Responses, and Anthropic returned byte-identical `"DET-731"` twice. This is still open; it was observed on the pre-fix running process and is distinct from the Hy3 `think:true` streaming mapping defect.

## Requested contract conclusions

- `reasoning_effort → enable_thinking`: precedence is correct in the shared resolver. `supports_thinking=False` fails closed; explicit top-level or template `enable_thinking=False` wins; only a non-off effort on a supported family maps to `True`. Mistral retains native `none/high`; DSV4 retains its `high/max` normalizer; Hy3 retains `low/high`; MiniMax-M3 retains `disabled/enabled/adaptive`. Ollama now reaches Hy3's normalizer too.
- Answer-pass cap: all four call sites use remaining budget and never allocate a negative value or an above-cap retry. The open defect is starvation at zero remaining budget.
- Streaming answer pass: the incremental helper preserves monotonic text and marker holdback. The adjacent legacy double-emission path is now gated off.
- H1 sampler: generic stochastic paths normalize once; compact top-k and greedy consume raw logits; processors run on raw logits; greedy is exact processed-logit argmax; seeded samplers remain request-local.
- `compress_after`: default is zero, no positive family default was found, and activation requires explicit environment or model-owned `jang_config`. Live health reported `objects_active=true`, `live_encode_enabled=false`, `compress_after=0`, and `resident_memory_reduction_claimed=false`; the capability wording is truthful.

## Verification

- Focused/broad unit selection: `432 passed`.
- Python compile: `server.py`, `mllm_batch_generator.py`, and the stress harness passed.
- `git diff --check`: passed.
- Live engine was reused on `127.0.0.1:8010`; no model was started or restarted.
