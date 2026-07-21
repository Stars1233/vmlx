# Nemotron Omni artifact Auto detection and launch parity

Date: 2026-07-21

Source cutoff before this checkpoint: `936a10c4cd2423862f790da9e20aa44f187761fa`

Scope: the exact installed
`dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK` bundle and the panel's
multimodal Auto/settings/launch contract. This is not a repeat of the existing
audio/media-salt/protocol campaign in `../20260720_nemotron_omni_audio/` and
`../20260720_nemotron_omni_media_cache_current/`.

## Verdict

`VERIFIED-LIVE_SCOPED`

- Bundle Auto detection: PASS source + focused contract + current Electron.
- Auto override persistence across real Start: PASS current Electron + SQLite.
- Text-decoder Omni argv parity: PASS visible preview + actual process argv.
- Eager load before first request: PASS current Electron Start + health.
- Attachment UI availability: PASS current Electron.
- Audio/image/video generation quality and protocol breadth: N/A in this gate;
  those live rows already have dedicated evidence and were not repeated.

## Retained failures

1. Before the source fix, the settings control visibly said
   `Auto (detect from model)` but the Omni Backend control was absent. The
   panel read only `config.json`; this artifact keeps Parakeet/RADIO media
   declarations in `config_omni.json`.
2. The first repaired Start exposed a second bug: the launch detector wrote
   its effective `isMultimodal=true` back into the session blob. Reopening the
   drawer showed `Force On`, even though the user had selected Auto.

Artifacts:

- `01-auto-before-fix-no-backend.png`
- `02-start-materialized-force-on-prefinal.png`

## Source repair

- `panel/src/main/model-config-registry.ts`
  - Adds one artifact-first Nemotron Omni decision owner.
  - Requires `nemotron_h` identity, a parseable `config_omni.json`, and matching
    indexed encoder/projector tensors.
  - Audio requires `sound_encoder.*` plus `sound_projection.*`; vision/video
    requires `vision_model.*` plus `mlp1.*`.
  - Removes the obsolete blanket branch that forced Nemotron-H text-only when
    `config.json` did not contain media keys.
- `panel/src/main/sessions.ts`
  - Keeps detected Omni media on the engine's text-decoder dispatcher. Auto and
    Force On do not emit generic `--is-mllm`; Force Off still emits
    `--text-only`.
  - Captures whether `isMultimodal` was actually present before detection and
    omits the computed value from persisted config when the user selected Auto.
- `panel/src/renderer/src/components/sessions/SessionSettings.tsx`
  - Mirrors the same command-preview routing contract.

The full patch at proof time is preserved in `source-diff.patch`.

## Exact artifact classification

`bundle-contract.json` records:

- weight format `mxtq`, profile `JANGTQ2`; this is JANGTQ/MXTQ
  Hadamard/codebook quantization, not affine JANG and not base MLX MXFP;
- `config.json` describes the `nemotron_h` text decoder;
- `jang_config.json` declares Omni capability;
- `config_omni.json` declares Parakeet at 16 kHz and the RADIO vision config;
- indexed tensors include 710 `sound_encoder.*`, three
  `sound_projection.*`, 390 `vision_model.*`, and three `mlp1.*` entries.

## Current live Electron proof

The dev app was relaunched with:

- user data: `/Users/eric/.vmlx-v1613-responsive-dev`
- CDP: `127.0.0.1:9335`
- engine: `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`

The current launch log printed:

`[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`

The real settings drawer showed Auto both before and after clicking the real
Start button. The Omni Backend selector remained visible. SQLite after Start
records no `isMultimodal` property, proving that Auto was not converted into
Force On. See:

- `03-auto-before-current-start.png`
- `04-auto-persists-after-current-start.png`
- `session-current.json`

The visible CLI preview and the actual process argv contain neither
`--is-mllm` nor `--text-only`. Actual argv is in `argv-current.txt`.

The real Start button materialized PID 84552 before any request. Current
`health-current.json` reports:

- `model_loaded=true`
- `last_request_time=null`
- about 9.35 GB active model memory
- Omni modalities `text`, `audio`, `image`, and `video`
- components `radio=true`, `parakeet=true`, `media_projector=true`
- hybrid typed cache with attention KV + native SSM companion + async rederive
- q4 TurboQuant only at the attention-KV storage boundary
- prefix, paged RAM, and block-disk L2 enabled

The active Chat screenshot and DOM show the exact Nemotron session, a Stop
control, and an attachment input accepting image, video, and audio types:
`06-chat-attachment-current.png`.

## Focused validation

`focused-validation.txt` records:

- `model-config-registry.test.ts`: 88 passed
- `settings-flow.test.ts`: 289 passed
- total: 377 passed
- panel typecheck: passed

Synthetic registry coverage fails closed when Omni metadata exists without
matching component tensors. Command-preview coverage pins Auto, Force On, and
Force Off routing.

## Remaining boundary

This gate does not promote signed-app behavior, Stage 2 native MLX quality, or
unrelated family/media rows. It deliberately does not rerun the already-proven
Nemotron audio/media-salt/L2/protocol matrix.
