# DSV4 terminal-first native-cache live gate

Date: 2026-07-19

Status: `VERIFIED-LIVE_SCOPED` for current-source Responses/Electron terminal
delivery, eager materialization, exact RAM reuse, and exact process-restart L2
restore. Strict sampled output quality and non-terminal partial-prefix reuse are
`PARTIAL` / architecture-limited. The overall release remains blocked.

## Current source and artifact

- Source: `0c9436bce7c6c2bdfc0a31c742b324269b203a50`
- Branch: `reconcile/1.5.68`
- Push target: `origin/codex/live-electron-gates-20260715`
- Artifact: `/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`
- Bundle truth: `model_type=deepseek_v4`, `weight_format=affine`; this is JANG
  affine, not JANGTQ/MXTQ.
- Cache truth: native `deepseek_v4_v8` composite SWA + CSA compressed pool +
  HCA compressed pool + incomplete-tail state. Generic TurboQuant is disabled;
  every live snapshot reported zero TQ-native block writes/hits.

## Source trace

- `vmlx_engine/scheduler.py:6970` derives the clean DSV4 N-1 key only when a
  prompt snapshot is unavailable.
- `vmlx_engine/scheduler.py:7024` schedules the DSV4 paged deferred descriptor
  instead of running a clean prefill inside response finalization.
- `vmlx_engine/scheduler.py:7495` owns shared post-terminal materialization for
  DSV4, ZAYA, mixed full/SWA, and MiniMax-M3.
- `vmlx_engine/scheduler.py:7685` calls that materializer from finished-request
  cleanup, after terminal output dispatch and before the next request is
  admitted.

## Raw Responses proof

The pre-restart driver sent one 766-token cold prompt, an exact repeat, and a
changed suffix. Every request returned HTTP 200, 11 progressive content deltas,
one `response.output_text.done`, one `response.completed`, and a terminal gap
below 0.15 seconds.

| run | elapsed | last content -> completed | cache observation |
| --- | ---: | ---: | --- |
| cold A | 22.1417 s | 0.1085 s | four native blocks written to SSD across A/B |
| exact warm A | 0.8166 s | 0.1458 s | 765 prompt tokens saved from L1 |
| changed suffix B | 2.1242 s | 0.0994 s | unsafe non-terminal composite partial was rejected |

The changed-suffix log is an intentional correctness boundary, not a missing
generic feature: intermediate DSV4 blocks contain local SWA fragments and
pending markers, while the complete CSA/HCA state lives in a terminal composite
block. The scheduler logged that it ignored the incomplete hit and avoided
reconstructing a corrupt cache.

After the real Electron `Save & Restart` replaced PID 45021 with 46544,
`model_loaded=true` and `last_request_time=null` proved eager model
materialization before any prompt. The first raw repeat refaulted the exact
prefix from SSD. The three replay requests retained progressive output and
sub-0.15-second terminal gaps; final health reported five disk promotion hits,
zero generic-TQ hits/writes, and 2,303 total saved tokens.

## Electron proof

- Fresh row 665 used Responses, Instruct, no tools and returned exact visible
  `DSV4-0C-UI1-DONE`, separate reasoning, no warning, and an incrementally
  changing UI region.
- Long row 671 restored 765 tokens in the real Electron chat. After a second
  UI `Save & Restart` replaced PID 46544 with 48507, Regenerate returned exact
  visible `DSV4-TERMINAL-A-DONE` with
  `cacheDetail=paged+dsv4+disk`; health reported three disk hits and one
  disk-promotion request.
- Normal UI field events plus the real Chat Settings footer Save persisted
  `temperature=0`, `max_tokens=128`, `wire_api=responses`,
  `enable_thinking=0`, and tools off in `chat_overrides`. The earlier automation
  attempt that saved NULL overrides clicked the prompt-template Save and is not
  classified as an app defect.

## Honest boundary

- Strict marker fidelity is not stable: both raw A runs emitted
  `DSV4-TERTERMINAL-A-DONE`, and a later capped Electron regeneration repeated
  the same `TER` duplication. Other Electron samples were exact. No output
  rewriting, token bias, or sampler coercion was added.
- DSV4 can restore exact terminal composite states from RAM and SSD, but cannot
  reuse arbitrary non-terminal partial prefixes without a complete CSA/HCA
  boundary state. Compatible full-KV/hybrid families retain the separate
  disk-only partial-prefix requirement.
- This scoped proof does not close DSV4 long-output quality/performance,
  cross-protocol tool continuation, cancellation/disconnect recovery, or the
  broader model/release matrix.
