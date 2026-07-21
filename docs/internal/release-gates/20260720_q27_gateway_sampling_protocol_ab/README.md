# Qwen3.6 MXFP4-MTP gateway sampling and terminal parity

Date: 2026-07-20
Host: `erics-m5-max.local`
Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Branch: `codex/postrelease-ui-drawers-20260720`
Source: `943f28b96`

## Scope

This gate closes the explicit-versus-omitted sampler translation row for the
already Electron-loaded `dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP` backend through
the current dev Electron gateway at `127.0.0.1:8088`. It does not generalize
the result to every model/parser family.

The artifact is base MLX MXFP4 with a native MTP head. It is not JANG affine
and it is not JANGTQ/MXTQ.

## Source trace

- `panel/src/main/api-gateway.ts:1419-1465` resolves the requested backend,
  enforces the single-model transition, and forwards Ollama temperature,
  top-p, non-negative top-k (including zero), min-p, and repeat penalty.
- `vmlx_engine/api/anthropic_adapter.py:61-92` accepts Anthropic temperature,
  top-p, and top-k; `:227-255` forwards them into the shared Chat request.
- `vmlx_engine/api/ollama_adapter.py:161-185` maps Ollama chat sampling to the
  shared OpenAI-compatible request.
- `vmlx_engine/server.py:1567-1698` implements request-over-startup-over-bundle
  resolution and preserves explicit neutral disables.
- `vmlx_engine/server.py:21743-21767` installs greedy omitted-request defaults
  only for the session's `deterministic-defaults` native-MTP policy.

## Live protocol matrix

`q27-gateway-sampling-protocol-ab.json` contains the full request bodies,
timestamped deltas, terminals, non-stream bodies, and checks for 16 live calls:
four protocols x omitted/explicit sampling x stream/non-stream.

| Protocol | Omitted stream | Explicit stream | Non-stream | Resolved values |
|---|---:|---:|---:|---|
| Chat Completions | 10 content deltas; stop + DONE | 16 content deltas; stop + DONE | Exact in both modes | omitted `0/1`; explicit `1/.95/top-k 20/repetition 1.05` |
| Responses | 10 deltas; `response.completed` | 18 deltas; `response.completed` | Exact in both modes | omitted `0/1`; explicit `1/.95/top-k 20/repetition 1.05` |
| Anthropic Messages | 12 deltas; `message_stop` | 19 deltas; `message_stop` | Exact in both modes | converted Chat logs show omitted `0/1`; explicit `1/.95/top-k 20` |
| Ollama chat | 10 deltas; done/stop | 17 deltas; done/stop | Exact in both modes | converted Chat logs show omitted `0/1`; explicit `1/.95/top-k 20/repetition 1.05` |

All 16 requests returned HTTP 200 and exact non-empty visible markers. Every
stream was progressive in wall-clock observation, reasoning stayed empty and
separate under the explicit thinking-off request, and every adapter emitted its
native terminal. `q27-gateway-sampling-session-logs.json` independently records
the engine's resolved kwargs rather than trusting request construction.

Explicit temperature 1 correctly disabled deterministic native-MTP execution;
the logs say MTP was skipped because the request was stochastic. Omitted
temperature 0 exercised native MTP D3. This is intended request-policy behavior,
not a silent sampler rewrite.

## Ownership and cache observations

`live-health-and-process.txt` records `single_model_mode=true`, Qwen as the only
running backend, all other saved backends stopped, and exactly one
`vmlx-engine serve` process. Backend health after the matrix had zero active
requests, 444 L1-indexed tokens, 444 block-L2 tokens, 572 SSM-companion L2
tokens, 11 q4-native TQ block writes, and no error terminal.

The A/B also exercised same-prompt stream/non-stream prefix reuse, but it did
not force RAM eviction or process-restart L2 restore; those are separate cache
gates and are not promoted here.

## Focused validation

- Python Anthropic/Ollama adapters: 92 passed.
- Panel gateway/Ollama/single-model selections: 82 passed.
- Panel TypeScript typecheck: passed.

See `focused-tests.txt` for exact commands and counts.

## Honest boundary

This is a current-source live closure for Qwen's four gateway protocol adapters
and sampler precedence. Cancellation/network-loss/failure injection was not
rerun in this gate; media-bearing requests, tool-result continuation under the
explicit stochastic sampler, concurrent-client soak, LAN/port rollback,
signed-app repetition, and other model/parser families remain PARTIAL/OPEN in
the master matrix.
