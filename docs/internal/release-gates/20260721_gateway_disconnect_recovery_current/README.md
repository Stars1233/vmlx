# Electron gateway client-disconnect recovery

Date: 2026-07-21

Worktree: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

## Verdict

`VERIFIED-LIVE_SCOPED` for downstream client disconnect cleanup and immediate
recovery through the current development Electron-owned gateway on port 8088.
No source fix was required and no already-green model/cache matrix was rerun.

## Live gateway proof

The Electron API gateway was running with Single Model On and the exact
`dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP` backend. For each protocol, the probe
started a 1,024-token stream, received three visible content deltas, and closed
the downstream HTTP response before a terminal:

| Protocol | Client close | Backend idle after close | Recovery content deltas | Recovery terminal |
|---|---:|---:|---:|---|
| Chat Completions | 487.35 ms | 24.88 ms | 8 | `stop`, `[DONE]` |
| Responses | 512.53 ms | 27.95 ms | 9 | `response.completed` |
| Anthropic | 518.26 ms | 28.08 ms | 10 | `message_stop` |
| Ollama | 477.97 ms | 27.97 ms | 9 | `stop` |

Every immediate recovery request returned HTTP 200, exact protocol-specific
content, multiple progressive content deltas, its native terminal, and an idle
backend afterward. Final gateway health retained one running Qwen backend and
all other configured sessions stopped. Final backend health reported zero
running/waiting/active requests. See `q27-gateway-disconnect-proof.json` and
`gateway_disconnect_probe.py`.

## Visible Electron recovery

After all four forced disconnects, the real Chat UI sent a new prompt in the
same loaded session. The DOM observer recorded 88 changing states. The final
screen visibly shows a separate 891-character Reasoning rail and exact visible
content:

`GATEWAY-UI-DISCONNECT-RECOVERY-DONE`

SQLite row 257 has that exact content, no tool payload, no warning, and metrics
for 130 `paged+ssm+disk+tq-native` cached tokens, 0.38-second TTFT, and
9.3-second total time. The final health snapshot is idle and records a real
three-block disk promotion for that UI recovery. See the UI screenshots,
`q27-gateway-disconnect-ui-trace.json`, and `ui-recovery-row.json`.

## Source and dead-code trace

- `panel/src/main/api-gateway.ts:430-449` owns downstream-close cleanup.
- `abortProxyResponseOnClientClose` has four production route call sites at
  lines 998, 1528, 1854, and 2144.
- `abortProxyRequestOnClientClose` has four production route call sites at
  lines 1011, 1731, 2005, and 2173.
- Guarded response writers at lines 338-427 prevent late EPIPE writes.

These helpers are active shared gateway code, not test-only or zombie
compatibility functions. No dead replacement branch was added in this proof.

## Honest remaining boundary

- This closes downstream client disconnect cleanup through the four protocol
  routes. It does not close an explicit request-ID `/cancel` request.
- Backend connection loss, injected upstream failure through the gateway,
  port/LAN failure while a request is active, and concurrent disconnect/swap
  soak remain open.
- This is current development Electron proof, not signed-app repetition.
- The Qwen model was used as the loaded transport producer; this gate does not
  repeat or promote its already-proven model, MTP, media, or cache rows.
