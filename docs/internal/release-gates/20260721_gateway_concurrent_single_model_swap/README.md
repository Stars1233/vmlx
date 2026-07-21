# Concurrent Single Model stream displacement and swap-back — current source

Date: 2026-07-21

Status: `VERIFIED-LIVE_SCOPED`

Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Cutoff: `6596122bd`

Models:

- `dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP` — base MLX MXFP4 plus native MTP
  depth 3; not affine JANG and not JANGTQ/MXTQ.
- `JANGQ-AI/Laguna-XS.2-JANGTQ` — JANGTQ2/MXTQ routed artifact.

## Scope and source trace

This is the unrepeated concurrent boundary left open by the earlier sequential
one-model swap and four-protocol backend-loss gates. With the Electron API
dashboard's Single Model toggle already On, a Qwen Chat Completions stream was
kept active while a second gateway request selected Laguna. The proof then
requested Qwen again to exercise swap-back.

The active production owners are:

- `panel/src/main/api-gateway.ts::withSingleModelTransition` serializes gateway
  target transitions.
- `panel/src/main/api-gateway.ts::enforceSingleModelMode` stops every other
  local running/loading/standby session before routing to the requested target.
- `panel/src/main/api-gateway.ts::guardProxyResponseLifecycle` emits the native
  failure boundary when that intentional stop aborts an in-flight backend
  response.
- `panel/src/main/sessions.ts::startSession` independently enforces the same
  stop-before-start ownership contract for manual Electron Start operations.

All four helpers have production call sites. No dead compatibility branch or
model-specific output rewrite was added for this proof, and no source edit was
required.

## Current-source live Electron and gateway proof

The Electron main process was already fully relaunched from this checkout and
had printed:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The real Server-page Start control had eagerly loaded Qwen before this row.
The concurrent run then observed:

1. Qwen emitted 262 progressive visible content deltas before its process was
   displaced. The gateway emitted one Chat-native
   `backend_connection_closed` server error at 14,738.20 ms, closed promptly,
   raised no client exception, emitted no false `[DONE]`, and leaked no
   reasoning or tool data.
2. Laguna JIT-loaded through the same gateway request and exact-finaled
   `CONCURRENT-SWAP-LAGUNA-DONE` over 12 content deltas in 16,579.74 ms with
   `stop` plus `[DONE]`, no reasoning/tool/error payload, and no truncation.
3. The captured real Electron Server page visibly showed `ACTIVE (1)` with
   Laguna running as PID 66989 and Qwen stopped. Process inspection found
   exactly one `vmlx-engine serve` process. Gateway health and SQLite agreed.
4. A gateway request for Qwen then stopped Laguna, JIT-loaded Qwen, and
   exact-finaled `CONCURRENT-SWAP-QWEN-RETURN-DONE` over eight content deltas in
   6,736.93 ms with `stop` plus `[DONE]` and no reasoning/tool/error payload.
5. The second captured Server page visibly showed `ACTIVE (1)` with Qwen
   running as PID 67116 and Laguna stopped. Process inspection again found one
   engine; health and SQLite agreed.

The final Qwen argv preserves the exact model contract used in this row:
`--is-mllm`, Qwen tool and reasoning parsers, paged/block-disk cache, native
MTP depth 3, and deterministic-defaults native-MTP sampling. The Laguna argv
shows its distinct JANGTQ path, GLM47 tool parser, Qwen3 reasoning parser, and
paged/block-disk cache. These argv observations are identity and ownership
evidence only; this row does not re-claim either family's already-proved cache
matrix.

Artifacts:

- `q27-laguna-concurrent-swap-proof.json`
- `concurrent-swap-laguna.png`
- `concurrent-swap-qwen-return.png`
- `concurrent_swap_probe.py`

Both screenshots were visually inspected, not accepted from DOM text alone.

## Boundary

This closes one real concurrent Chat displacement, target load, and swap-back
under Single Model mode on current dev Electron source. It does not close a
multi-client stress soak, simultaneous host/port mutation plus swap, repeated
failure during target load, every parser family, non-stream partial-body loss,
full suites, or a signed/notarized app repetition. One unread `Stopped` then
`Listening` pair appeared once in accumulated dev stdout between the Laguna
and Qwen logs; a direct API-to-Server navigation repetition produced no new
listener restart. It is retained as un-attributed observation rather than
rationalized as a product fix or included in this scoped pass.
