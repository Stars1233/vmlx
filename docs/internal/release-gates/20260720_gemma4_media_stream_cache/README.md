# Gemma 4 media ordering, image budget, streaming, and cache gate

Date: 2026-07-20
Host: `erics-m5-max.local`
Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Branch: `codex/postrelease-ui-drawers-20260720`

## Verdict

| Axis | Current verdict | Current live evidence |
|---|---|---|
| Real Electron Start/load | PASS-LIVE | The Korean UI Start button loaded PID 35513, then later PIDs 35937 and 36376 after visible Stop/Start cycles. The dev log named `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`; `/health` reported `model_loaded=true`. |
| Bundle/config classification | PASS-SOURCE+LIVE | Real bundle is `gemma4_unified`, affine `JANG_4M` (`weight_format=jang_affine`), 40 rotating-SWA plus 8 full-attention layers, vision+audio and no advertised video. Runtime layout reported 40 `RotatingKVCache` and 8 `TurboQuantKVCache` layers. This is not JANGTQ/MXTQ. |
| Electron composer media order | PASS-LIVE | Commit `7687f237b` follows the real bundle contract: visual media before text and audio after text. Electron request diagnostics show `[image_url, text]`. The old `[text, image]` path looped or mangled output. |
| Explicit Gemma image-token budget | PASS-SOURCE+LIVE | Commit `a0abd7ab3` adds validated 70/140/280/560/1120 request/session values and media-cache salting. The UI visibly selected 1120; request diagnostics carried `image_token_budget=1120`; prompt size rose from 328 to 1,144 tokens. |
| Reasoning/content SSE separation | PASS-LIVE transport | `gemma-a6-raw1120-sse.txt` starts reasoning deltas at 1.44s, starts progressive content at 7.40s, and finishes with `response.completed` at 8.28s. No reasoning/channel marker appears in content. |
| Resident prefix reuse | PASS-LIVE | Identical A6 reused 1,137/1,138 tokens as `paged+mixed_swa+tq-native`; first reasoning fell to 0.32s. Health recorded 20 native-TQ block writes. |
| Restart/L2 restore | PASS-LIVE cache; output quality separate | After visible Electron Stop/Start, A7 restored 1,137/1,138 tokens as `paged+mixed_swa+disk+tq-native`; health recorded 18 disk promotions and 18 native-TQ hits with zero new writes. The turn completed, but OCR was still inexact. |
| Reasoning-budget fallback | PASS-LIVE after `1b89e1118` | A8 deliberately capped thinking at 64 tokens. The real Gemma fallback ran, emitted 57 reasoning events followed by 30 progressive content events, leaked no literal `thought`, and terminated `response.completed`. Focused regression: 57 passed, two intentional skips. |
| Small-text OCR fidelity | **PARTIAL / FAIL exactness** | Electron A5 read `jiang-ai/...`; raw A6 read `jangg-ai/...`; A7/A8 also altered characters/case. The real target is `jangq-ai/gemma-4-12B-it-qat-JANG_4M`. Transport, budget, streaming, and cache behavior must not be described as an OCR pass. |
| Different-media salt and return-A | PASS-LIVE | Same-size 2800x1800 image A visibly contained two active session cards and image B contained one. With an identical prompt, A/B/A answered `2`/`1`/`2`; B claimed no cached tokens, while return-A restored 1,097/1,098 as `paged+mixed_swa+tq-native`. After a real Electron Stop/Start, return-A again answered `2` and restored 1,097 tokens as `paged+mixed_swa+disk+tq-native`; health recorded 18 disk promotions and 18 native-TQ hits with no writes. |
| Post-media text/tool history | PASS-LIVE | In the same real Electron chat, a text-only turn recalled the preceding image-turn marker exactly. After enabling built-in tools in the real Chat Settings UI, the next turn made exactly one `file_info(panel/package.json)` call, consumed its real 5.2 KB result, and exact-finaled `G4-POSTMEDIA-TOOL-DONE SIZE=5.2 KB` with no warning. |
| Audio/video breadth | PARTIAL | Gemma audio was not rerun here. Video remains a negative-capability row because this bundle advertises `has_video=false`; rejection/fallback behavior was not rerun. |

## Source trace

- `panel/src/shared/composerContentOrder.ts` keeps Gemma visual parts before
  text and audio after text without changing other families.
- `vmlx_engine/api/models.py`, `vmlx_engine/server.py`,
  `vmlx_engine/engine/batched.py`, `vmlx_engine/mllm_scheduler.py`, and
  `vmlx_engine/mllm_batch_generator.py` carry and validate the request-local
  `image_token_budget` and include it in media cache identity.
- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` exposes the
  real supported budgets, while `panel/src/main/ipc/chat.ts` forwards the saved
  Gemma value on both Chat Completions and Responses requests.
- `vmlx_engine/server.py::_ANSWER_PASS_CONTROL_PREFIXES` now recognizes the
  degraded `thought\n` Gemma channel before any irreversible content delta.
- `vmlx_engine/server.py::_ANSWER_PASS_FRESH_CONTEXT_FAMILIES` now includes
  Gemma because its real template renders `reasoning_content` only on assistant
  tool-call turns; appending a bare reasoning turn produced an empty duplicate
  model turn instead of a valid direct-answer prompt.

## Live chronology

1. A2 with text-before-image looped. Disabling cache did not change it, which
   isolated the failure away from prefix/TQ/L2 state.
2. A4 used the corrected `[image_url, text]` Electron request. It completed in
   13.5s with separate reasoning and content, but OCR was inexact.
3. A5 selected 1120 through the real Server Settings drawer and restarted from
   the real UI. It completed without looping; prompt tokens rose to 1,144, but
   `jangq-ai` was read as `jiang-ai`.
4. A6 raw Responses proved progressive reasoning/content. Its identical warm
   repeat restored 1,137 tokens from resident paged mixed-SWA/TQ state.
5. The first process-restart A6 promoted 18 native-TQ blocks from disk, but its
   bounded fallback leaked the partial degraded channel word `thought` and hit
   the total output cap. This retained failure is
   `gemma-a6-disk1120-sse.txt`.
6. Commit `1b89e1118` buffers the degraded channel prefix and reruns Gemma's
   direct fallback from fresh original context. A7 then restored the same disk
   prefix and completed; A8 forced the fallback and progressively emitted a
   visible answer with no channel leak.
7. The same-size A/B/A media-salt probe returned `2`, `1`, and `2`. B did not
   reuse A; return-A restored 1,097 tokens from resident cache. After a real
   Electron Stop/Start, return-A restored the same 1,097 tokens from L2 with 18
   disk/native-TQ hits and no new writes.
8. The original Electron image chat then completed a text-only history turn and
   one exact `file_info` tool loop after tools were visibly enabled in Chat
   Settings. This proves the media turn did not poison later text or tool state.

## Validation

- `tests/test_answer_pass_families_dsv4_step37.py`: 9/9 passed.
- Expanded Gemma/server/reasoning/media selection: 57 passed, two intentional
  skips, 697 deselected.
- Earlier image-budget focused selection: 121/121 Python tests.
- Panel chat/settings selection: 429/429; TypeScript typecheck passed.

## Evidence files

- `gemma-media-a5-budget1120.png` — visible Electron A5 output; structurally
  complete but OCR-inexact.
- `gemma-a6-raw1120-sse.txt` — cold raw Responses timing/events.
- `gemma-a6-warm1120-sse.txt` — identical resident-cache hit.
- `gemma-a6-disk1120-sse.txt` — retained pre-fix restart/L2 fallback failure.
- `gemma-a7-disk-fallback-fix-sse.txt` — post-fix restart/L2 completion.
- `gemma-a8-fallback-fix-sse.txt` — forced bounded fallback with progressive
  content and no `thought` leak.
- `gemma-salt-a1.sse.txt`, `gemma-salt-b.sse.txt`, and
  `gemma-salt-a2-return.sse.txt` — same-size A/B/A isolation and resident
  return-A proof.
- `gemma-salt-a3-restart-disk.sse.txt` — real Electron restart/L2 return-A
  proof.
- `gemma-salt-ui-stopped.png` and `gemma-salt-ui-restarted.png` — visible
  Electron lifecycle proof.
- `gemma-postmedia-text-pass.png` and `gemma-postmedia-tool-current.png` —
  visible same-chat text recall and exact one-tool continuation.

## Remaining work

- Run a controlled same-artifact reference A/B before classifying exact OCR as
  quant quality. Do not postprocess or silently rewrite the model output.
- Exercise real audio through Electron and raw APIs because this bundle
  advertises audio. Keep video separate because `jang_config.json` says
  `has_video=false`.
