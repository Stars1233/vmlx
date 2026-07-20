# Nemotron Omni audio conversation and media-salt gate

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Base HEAD before this scoped fix: `ae317b60dc1916f03cbb30ad808431172924096e`

## Verdict

- Audio transport and model routing: **PASS-LIVE scoped**.
- Chat Completions and Responses progressive reasoning/content streaming:
  **PASS-LIVE scoped**.
- Same-conversation post-audio recall through the Electron Responses route:
  **PASS-LIVE scoped**.
- Cross-media cache isolation and reset-time media rehydration:
  **PASS-LIVE scoped**.
- Strict formatting reliability: **PARTIAL**. The first stochastic UI control
  added a closing code fence. In the clean three-turn run, turn 2 returned
  `blue6813` instead of the requested hyphenated form; turn 3 returned exact
  `BLUE-6813` after an explicit formatting instruction.
- Omni media-session restart/L2 restore: **OPEN**. The persistent Omni KV+SSM
  session is process-local; the ordinary scheduler's paged/TQ/L2 counters are
  not evidence that media-conditioned Omni state survived process restart.

## Bundle-grounded configuration

`bundle-facts.json` was derived from the live bundle files. The artifact is
`model_type=nemotron_h`, `weight_format=mxtq`, with JANGTQ codebook routing
(`routed_expert=2`, attention/shared/mamba/embed/head at 8 bits). It is not an
affine JANG artifact and not base MLX MXFP. `config_omni.json` advertises the
Omni reasoning architecture, Parakeet sound encoder at 16 kHz, C-RADIO vision,
and video tokens.

## Root cause and source repair

Two independent stale-media paths were reproduced.

1. `vmlx_engine/omni_multimodal.py` hashed only user text when deciding whether
   its persistent KV+SSM conversation prefix matched. Replaying identical text
   with blue audio after orange audio therefore reused orange state; the
   retained pre-fix stream answered `MARKER=ORANGE-4729`.
2. `panel/src/main/ipc/chat.ts` stripped historical media for every local
   follow-up. That is correct for ordinary VL tool replay but wrong for the
   stateful Nemotron Omni dispatcher: a text follow-up became
   `chatIsMultimodal:false` and bypassed Omni entirely.

The engine now includes stable media identities in each user-turn signature.
A prefix mismatch resets the persistent session; if the current follow-up has
no new media, the dispatcher rehydrates the latest prior-turn media before the
new turn. The panel preserves historical media only when the detected family
is `nemotron-h` and the selected bundle actually contains `config_omni.json`.
Other families retain the historical-media stripping policy.

Regression coverage:

- `tests/test_omni_multimodal.py` changes audio bytes under identical text and
  requires reset plus blue-media rehydration.
- `panel/tests/omni-media-history-replay.test.ts` pins the bundle-qualified
  panel exception.

## Current live Electron proof

The old Electron main process was fully stopped and relaunched against the
same isolated profile on CDP 9335. `electron-startup-excerpt.log` contains:

```
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The real UI `Start` button loaded
`dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK` on port 8001 in six seconds. Before
any request, `/health` reported `model_loaded=true`,
`last_request_time=null`, and 9,348.1 MB active memory. This is eager load
evidence, not first-message materialization.

The clean Electron chat ID was
`9a534f36-8675-48a0-92c2-9938d5d675b6`. Neither user prompt contained the
spoken marker:

1. Attached real WAV; visible final `READY`, separate 445-character reasoning.
2. No attachment; visible final `blue6813`, separate 367-character reasoning.
3. No attachment; visible exact final `BLUE-6813`, separate 354-character
   reasoning.

All three reasoning hashes differ; there is no byte-identical stale reasoning
replay. `electron-omni-route-excerpt.log` proves turn 2 sent the historical
`input_audio`, logged `preserveHistoricalMediaForOmni:true`, reached
`[MEDIA_DIAG]`, and the engine logged
`OmniMultimodalDispatcher: continuing conversation (prefix matches)` with
`audio=no` for the new turn. The marker therefore came from the persistent
media-conditioned session, not from prompt text.

The controlled cross-media sequence also retains both sides:

- `media-salt-prefixed-stale.tsv`: pre-fix blue-history replay leaked
  `MARKER=ORANGE-4729`.
- `media-salt-postfix-blue.tsv`: post-fix reset/rehydration progressively
  emitted exact `MARKER=BLUE-6813`, then stop, usage, and `[DONE]`.

## Raw API stream proof

- `chat-audio-stream.tsv`: 146 non-empty `reasoning_content` deltas, then 24
  non-empty content deltas, `finish_reason=stop`, one usage object, and one
  `[DONE]`.
- `responses-audio-stream.tsv`: 160 reasoning-summary deltas, then 26 output
  text deltas, matching reasoning/text done events, one completed output item,
  and exactly one `response.completed` with usage.

Both streams produced the exact requested orange marker/final. Reasoning and
visible content remained on separate protocol rails.

## Validation

- Focused Python Omni/multimodal selection: 25 passed.
- Full panel after the patch: 77 files, 2,346 passed, 3 skipped.
- Panel TypeScript typecheck: passed.
- Full Python execution after source changes: 6,202 passed, 96 skipped, 92
  deselected, with one expected fail-closed bundled-source drift failure.
- `panel/scripts/bundle-python.sh` was then run against the clean detached
  `jang-tools` checkout, not the user's dirty working tree.
- `verify-bundled-python.sh`: all critical source, JANG, import, and
  relocatability checks passed with the release PATH.
- The formerly failing bundled-Python verification test passed 1/1 after the
  bundle refresh.

## Evidence index

- `bundle-facts.json`
- `health-after-electron-start.json`
- `electron-startup-excerpt.log`
- `electron-omni-route-excerpt.log`
- `electron-chat-db.json`
- `electron-start.png`
- `electron-audio-turn1.png`
- `electron-audio-turn2.png`
- `electron-audio-turn3.png`
- `electron-initial-audio-stray-fence.png`
- `chat-audio-stream.tsv`
- `responses-audio-stream.tsv`
- `media-salt-prefixed-stale.tsv`
- `media-salt-postfix-blue.tsv`
