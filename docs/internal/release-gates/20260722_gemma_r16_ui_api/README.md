# 2026-07-22 Gemma 4 E2B UI/API reasoning and tool proof

Status: `VERIFIED-LIVE_SCOPED / GLOBAL GEMMA MATRIX PARTIAL`.

Source checkpoint: `78be62bf3d186cd8b045d8e069610f2ff5fb5ef8`.

Live app: Electron dev app on `erics-m5-max.local`, CDP `9335`, gateway
`127.0.0.1:8088`.

Model:
`/Volumes/EricsLLMDrive/jangq-ai/gemma-4-E2B-it-qat-JANG_4M`.

## Bundle and session defaults

The Gemma session was created through Electron `sessions.create` with an empty
config so current source applied bundle-derived startup defaults. The visible
Chat Settings matched the bundle/app-derived defaults:

- Enable Thinking: `Auto`
- Temperature: `1.00`
- Top P: `0.95`
- Top K: `64`
- Min P: `0.00`
- Repetition Penalty: `1.00`
- API wire format: `Responses`
- Built-in coding tools: enabled

Health/session facts:

- PID `56353` for the retained UI/API proof
- `toolCallParser=gemma4`
- `reasoningParser=gemma4`
- native cache `mixed_swa_kv_v1`
- Paged RAM On and Block Disk L2 On after the cache proof was restored

One retained settings issue remains: the visible Working Directory was still
`/Users/eric/mlx/vllm-mlx-release-1.6.13`. The tool used here still resolved
`panel/package.json`, but release cleanup should reset or validate stale
working-directory paths.

## Electron UI proof

All three UI turns were sent through the visible Gemma chat.

| Turn | Result |
|---|---|
| `GEMMA-R16-UI-T1` | Separate Reasoning rail `1106 chars`; exact visible `GEMMA-R16-UI-T1-DONE`; `395 tokens`, `144.6 t/s`, `0.13s TTFT`, `2.9s total` |
| `GEMMA-R16-UI-T2` | Separate Reasoning rail `733 chars`; exact visible `GEMMA-R16-UI-T2-DONE PREV=GEMMA-R16-UI-T1-DONE`; prompt showed `68 paged+mixed_swa cached` |
| `GEMMA-R16-UI-T3` | Separate Reasoning rail `271 chars`; exactly one visible `Info panel/package.json` tool card; exact visible `GEMMA-R16-UI-T3-DONE SIZE=5.2 KB` |

No native thinking markers, markdown/LaTeX corruption, or random dollar-number
display issue was observed in these retained Gemma UI turns.

## Gateway API proof

Artifact: `gemma-api-gateway-proof.json`.

Gateway health routed `jangq-ai/gemma-4-E2B-it-qat-JANG_4M` as the running
single-model backend.

| Case | Result |
|---|---|
| Chat Completions stream | `reasoning_content` `621 chars`; exact visible `GEMMA-R16-API-CHAT-DONE`; finish `stop` plus `[DONE]`; no native marker leak |
| Responses stream | reasoning `579 chars`; exact visible `GEMMA-R16-API-RESP-DONE`; terminal `response.completed`; no native marker leak |
| Chat required tool | reasoning `261 chars`; tool-call chunks emitted; finish `tool_calls` plus `[DONE]`; no visible-content marker leak |
| Chat tool-result continuation | reasoning `549 chars`; exact visible `GEMMA-R16-API-TOOL-DONE SIZE=5.2 KB`; finish `stop` plus `[DONE]`; no native marker leak |

## Remaining Gemma work

- Anthropic and Ollama protocol rows for this current source checkpoint.
- Media proof for image/video/audio only where the exact artifact and runtime
  advertise support.
- Low-limit eviction/refault for `Max Cache Blocks` and `Block Cache Max (GB)`.
- Signed-app markdown/math rendering proof.
- Stale working-directory reset/repoint validation.
