# 2026-07-22 settings/defaults/parser parity checkpoint

Status: PARTIAL-LIVE.

This checkpoint covers the source/UI default-detection issue where Chat
Settings did not visually match model-owned generation and chat defaults.
It does not close the broader release matrix or the raw TPS accuracy gate.

## Source changes

- `panel/src/main/model-config-registry.ts`
  - mirrors `vmlx_engine.model_config_registry._jang_stamp_default_enable_thinking`;
  - reads `jang_config.chat.template_kwargs_defaults.enable_thinking` before
    `jang_config.chat.reasoning.default_enabled`;
  - keeps runtime-policy exceptions locked for ZAYA1-VL, Hy3, MiMo-V2, Ling;
  - aligns text ZAYA with current engine policy: Auto reasoning defaults on.
- `panel/src/main/chat-override-policy.ts`
  - raises the chat top-k override hard cap from `1000` to `1_000_000`.
- `panel/src/renderer/src/components/chat/ChatSettings.tsx`
  - expands the Top K slider maximum to at least the displayed model default.
- `panel/src/renderer/src/components/chat/chat-utils.ts`
  - labels `t/s` as decode speed after first token, including reasoning and
    tool-loop completion tokens. This avoids implying end-to-end wall-clock
    throughput.

## Focused source gates

Command:

```sh
cd /Users/eric/mlx/vllm-mlx-release-1.6.13/panel
export PATH=/Users/eric/.local/node/bin:$PATH
npm test -- --run tests/chat-ui.test.ts tests/chat-metrics.test.ts tests/model-config-registry.test.ts tests/chat-override-policy.test.ts tests/generation-defaults.test.ts
npm run typecheck
```

Result:

- 5 test files passed.
- 278 tests passed.
- `tsc --noEmit` passed.

## Live Electron evidence

App state:

- Dev Electron relaunched with
  `VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1613-responsive-dev` and CDP
  `127.0.0.1:9335`.
- Startup log showed:
  - `[STARTUP] Using vMLX userData override: /Users/eric/.vmlx-v1613-responsive-dev`
  - `[STARTUP] No existing vmlx-engine processes found`
  - `[gateway] Listening on 127.0.0.1:8088`
  - `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`
  - `[Engine Manager] Version: 1.6.14`

Artifacts:

- `laguna-chat-settings-auto-defaults.png`
- `openpangu-chat-settings-topk-151552.png`
- `live-ipc-and-dom.json`

Observed live IPC/DOM values:

- Laguna S-2.1 JANG_2L:
  - `toolParser:"glm47"`
  - `reasoningParser:"deepseek_r1"`
  - `supportsThinking:true`
  - `thinkInTemplate:true`
  - `defaultEnableThinking:true`
  - UI Chat Settings showed Auto selected, Temperature `1.00`, Top P `1.00`,
    Top K `20`, Min P `0.00`, Repetition Penalty `1.00`.
- openPangu-2.0-Flash JANG_3M:
  - `family:"openpangu_v2"`
  - `cacheSubtype:"openpangu_v2_composite"`
  - `usePagedCache:false`
  - generation defaults from `generation_config`: Temperature `1.0`, Top P
    `0.8`, Top K `151552`.
  - UI Chat Settings visibly showed Top P `0.80`, Top K `151552`.
  - DOM range input for Top K had `value:"151552"`, `max:"151552"`, `step:"1"`.

## Still open

- Raw metric truth: compare Electron `tokensPerSecond` for the same generation
  against raw SSE token arrival timestamps and server usage. Current source
  labels it as decode speed after first token, but the exact 90+ t/s historical
  rows remain PARTIAL until that live timing comparison is captured.
- Broad family settings parity: repeat bundle-grounded detector + visible Chat
  Settings + server argv/health for remaining families that are still in the
  active matrix.
- Release readiness: BLOCKED until full current-source gates, package build,
  signing/notarization, install smoke, and version/public truth are complete.
