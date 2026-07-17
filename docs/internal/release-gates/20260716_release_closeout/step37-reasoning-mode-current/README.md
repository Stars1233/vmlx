# Step 3.7 native reasoning-mode and streaming proof

Status: `PASS-LIVE` for the scoped UI/API capability contract, deterministic
reasoning/content streaming, and cold/RAM/restart-disk cache equivalence at
commit `8b0e23dc1`. Overall Step release status remains `PARTIAL` because one
normal-temperature cached Responses probe exhausted 1,024 reasoning tokens in
a repeated `Isa` loop and emitted no content. The raw leading newline, cold
media-store latency, longer-video coverage, and broader output soak also remain
open.

## Root cause and source trace

The real bundle
`/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K` has a native template
that always appends `<|im_start|>assistant\n<think>\n`. It accepts
`reasoning_effort`, but it has no `enable_thinking` branch and therefore no
truthful direct/instruct mode. The former generic thinking-off sentinel closed
that native rail and exposed planning as assistant content.

- `vmlx_engine/model_configs.py:877-896` now records Step as reasoning-capable,
  instruct-unsupported, with native `low`, `medium`, and `high` efforts.
- `vmlx_engine/server.py:1918-1924` removes Step from the direct answer-pass
  family set. `server.py:3181-3206` rejects request/template Off values, and
  `server.py:9927-9949,10005-10019` exposes the same capability contract.
- `vmlx_engine/utils/chat_template_kwargs.py:87-111` preserves Step's native
  open reasoning rail instead of fabricating an empty thought.
- `panel/src/main/model-config-registry.ts:248-251` and
  `ChatSettings.tsx:82-101,486-568` drive the real UI from those capabilities:
  Auto/On only, native effort buttons, and a truthful native-only explanation.
  `panel/src/main/ipc/chat.ts:1372-1378,1887-1903` rejects stale local Off state
  and refuses to coerce recovery requests.
- The registry regression test explicitly proves ordinary Qwen3-VL was not
  accidentally reclassified while Step was changed.

No output rewriting, hidden sampler clamp, forced fake think tag, or prompt
answer injection was added.

## Focused automated proof

```text
.venv/bin/pytest -q \
  tests/test_model_config_registry.py \
  tests/test_reasoning_modes.py \
  tests/test_thinking_template_render.py \
  tests/test_answer_pass_families_dsv4_step37.py
219 passed, 16 skipped, 6 deselected in 3.30s

npm test -- --run \
  tests/model-config-registry.test.ts \
  tests/chat-settings-compatibility.test.ts
98 passed

npm test -- --run \
  tests/i18n-consistency.test.ts \
  tests/chat-settings-compatibility.test.ts
24 passed

npm run typecheck
passed
```

## Live Electron and API contract

The dev Electron app was rebuilt against the same
`/Users/eric/.vmlx-v1611-cachefix-dev` state, then Step was started and twice
restarted through visible `Save & Restart`. Live PIDs were 54658 and 55815.

- The Chat Settings screenshot shows only Auto/On, Auto/Low/Medium/High, and:
  `This model exposes native reasoning only ... Thinking Off is unavailable.`
- `/v1/models/default/capabilities` reports `family=step3p7`,
  `supports_thinking=true`, `supports_instruct_mode=false`,
  `supported_modes=[reasoning]`, and efforts `low/medium/high`.
- Top-level `enable_thinking=false`, template-level false, and
  `thinking_mode=off` each returned HTTP 400 before generation.
- Electron row 297 persisted 1,062 reasoning characters separately from exact
  content `3973 STEP37-LIVE-RELOAD-DONE`, with no warning.
- Electron video row 300 returned exact `BANANA8426` after restart and restored
  376/377 prompt tokens as `paged+mixed_swa+disk` with 1.19s TTFT.

## Deterministic cache and stream control

The identical temperature-zero Responses prompt was run cold, from resident
RAM cache, and after another visible process restart from block-disk L2:

| Phase | Cache detail | Reasoning deltas | Content deltas | Final |
|---|---|---:|---:|---|
| Cold | none | 234 | 11 | `\n2993 STEP37-DET-CACHE-DONE` |
| RAM | 30 `paged+mixed_swa` | 249 | 11 | same |
| Disk | 30 `paged+mixed_swa+disk` | 249 | 11 | same |

The same visible result and terminal status survived both cache tiers. The
leading newline is retained as a strict raw-format partial rather than hidden.

## Retained red row

The temperature-0.6 Responses replay used a 26-token
`paged+mixed_swa+disk` prefix and streamed 1,024 independently timed reasoning
deltas over 29.9s. It then ended `response.incomplete` with empty content after
repeating `Isa`. This disproves a buffering freeze—the deltas were live—but it
is still a model/runtime reliability failure. Deterministic cold/RAM/disk
equivalence argues against a general cache reconstruction defect; it does not
erase the stochastic failure. The next Step soak must reproduce and trace the
sampler/decode state without adding a hidden repetition penalty or forced
temperature.

## Evidence

- `step37-native-reasoning-help-current.png`
- `step37-live-reload-final.png`
- `step37-video-l2-reasoning-stream.png`
- `capabilities.json` and the three `*-400.json` responses
- `electron-rows.json`
- `deterministic-{cold,ram,disk}.json`
- `stochastic-loop-incomplete.json`
- `health-after-deterministic-disk.json`
