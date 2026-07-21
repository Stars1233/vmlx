# JANGTQ Sampling and Session-Chat Lifecycle — Current Source

Date: 2026-07-21

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Artifact: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`

## Scope and artifact identity

This is a scoped JANGTQ/MXTQ row. It is not affine JANG and is not base
MLX/MXFP. The exact bundle declares `model_type=qwen3_5_moe`,
`weight_format=mxtq`, `JANGTQ2`, and `method=affine+mxtq`. Its routed experts
are two-bit MXTQ while attention, shared experts, embeddings, and LM head are
eight-bit. Its `generation_config.json` declares sampling on with temperature
`1.0`, top-p `0.95`, and top-k `20`.

## Source trace

- `panel/src/main/ipc/models.ts` owns both generation-default reading and the
  new bundle-grounded `hasUsableChatTemplate()` check. A JANG/JANGTQ label no
  longer implies that a current bundle is stale.
- `panel/src/renderer/src/components/sessions/SessionView.tsx` displays the
  redownload notice only when the scanner explicitly reports a missing usable
  template. It now calls one `chat.ensureForModel` operation instead of racing
  a renderer-side `getByModel` and `create` pair.
- `panel/src/main/ipc/chat.ts` owns the synchronous lookup-or-create operation
  and reuses one `createChatRecord()` path for ordinary and ensured chats.
  This removes the duplicate initial-chat branch exposed by development
  effect replay.
- `panel/src/preload/index.ts` and `panel/src/env.d.ts` expose the single IPC
  contract to the renderer.

## Live Electron proof

All UI observations used the real current-source Electron app through CDP
`127.0.0.1:9335`, with user data
`/Users/eric/.vmlx-v1613-responsive-dev`. The main log contained
`[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.

- Real Start eagerly materialized the model before the first request. The
  session settings visibly showed temperature `1.00`, top-p `0.95`, top-k
  `20`, min-p `0.00`, and repetition penalty `1.00`. SQLite retained no chat
  sampler overrides, and Logs resolved the same bundle-owned values.
- The no-tool UI turn finalized exactly `MAGENTA-CEDAR-4821`, with non-empty
  visible content, no reasoning, tools, or warnings, and 0.21-second TTFT.
- An instrumented follow-up visibly painted growing prefixes beginning with
  `ALPHA B` and `ALPHA BRAVO`, then finalized the requested twelve-word line.
  The stored row reported 27 completion tokens, 115 prompt tokens, 48
  `paged+ssm+tq-native` cached tokens, 0.25-second TTFT, and no reasoning or
  warning payload.
- Current renderer IPC scanned the exact artifact as
  `JANGTQ2 (2b)` with `hasChatTemplate=true`. Opening the real running session
  showed no stale JANG redownload banner.
- The retained pre-fix reproduction had created two empty initial chats only
  9 ms apart (`1784668332622` and `1784668332631`). After removing only those
  agent-owned test chats, current source opened the session and produced
  exactly one initial chat, ID `6c1a3209-f015-43f3-b4d2-a797b0c197ce`, at
  `1784669480893`.

Screenshots:

- `jt-chat-settings.png`
- `jt-chat-settings-topk.png`
- `jt-samp-ui-progress-final.png`
- `jt-samp-ui-logs.png`
- `jt-template-banner-live-open.png`
- `initial-chat-dedupe-live.png`

## Raw Responses proof and retained negative control

- With sampler values omitted and thinking disabled, the same
  Electron-loaded model emitted `response.created`, a progressive text delta
  containing `Paris`, `response.output_text.done`,
  `response.output_item.done`, and one `response.completed`. Usage was 33
  input, 2 output, and 35 total tokens.
- A deliberately explicit 64-token cap emitted progressive deltas but ended
  in truthful `response.incomplete`. This is retained as a harness-negative
  control and is not counted as a product pass or hidden by a retry.

## Automated validation

- `panel/tests/chat-override-policy.test.ts`: initial chat lookup and creation
  share the main-process ensure owner.
- `panel/tests/generation-defaults.test.ts`: embedded template, standalone
  template, blank template, and missing include target are covered.
- Focused result: 36/36 passed.
- `npm run typecheck`: passed.
- Full panel/Python suites were not run for this scoped change.

## Verdict and boundaries

- JANGTQ/MXTQ model-derived sampling chain: `VERIFIED-LIVE_SCOPED`.
- Current-template JANG notice suppression: `VERIFIED-LIVE_SCOPED`.
- Development initial-chat deduplication: `VERIFIED-LIVE_SCOPED`.
- Base MLX/MXFP sampling, typed DSV4/M3 routes, and a model with a non-neutral
  repetition-penalty default remain `PARTIAL` and were intentionally not
  repeated or inferred from this Qwen row.
- No prompt coercion, output rewrite, hidden sampler clamp, or model-specific
  fake completion behavior was added.
