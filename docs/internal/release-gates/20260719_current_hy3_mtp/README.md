# HY3 JANG_2K native-MTP D1 current-source Electron/API gate

Date: 2026-07-19

Source cutoff: `0e09ce789` on `reconcile/1.5.68`, pushed to
`origin/codex/live-electron-gates-20260715`.

Verdict: `VERIFIED-LIVE_SCOPED` for current-source Electron multi-turn,
Responses/Chat streaming and tool continuation, native MTP depth 1, q4
stored-prefix reuse, and process-restart L2 restoration. The HY3 family remains
`PARTIAL` for long/stochastic soak and a fresh current-head MTP-Off performance
A/B. This is not a release verdict.

## Artifact truth

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`.
- `config.json`: `model_type=hy_v3`, 80 layers,
  `num_nextn_predict_layers=1`.
- `jang_config.json`: affine mixed JANG, `profile=JANG_2K`, routed gate/up 2-bit,
  routed down 3-bit, attention/shared/mtp 8-bit, routed average 2.333 bits.
  This is JANG affine, not JANGTQ/MXTQ and not base MLX MXFP.
- The tensor index contains 42 MTP tensors for exactly one indexed MTP layer.
- The sidecar says text only: no vision, audio, or video claim is made.

## Source trace

- `vmlx_engine/model_configs.py:1438-1498` registers `hy_v3` with the Hunyuan
  tool parser and qwen3 reasoning parser.
- `vmlx_engine/native_mtp.py:504-567,705-797` resolves the real bundle/tuning
  metadata and clamps HY3 to effective depth 1.
- `vmlx_engine/patches/mlx_lm_mtp/batch_generator.py:593-710` selects the
  effective depth and owns independent main/draft verification state.
- `vmlx_engine/utils/turboquant_config.py:123-153` selects q4 only for HY3's
  compatible full-KV stored-prefix boundary. It does not relabel model weights
  or store rejected speculative drafts as prompt history.
- `vmlx_engine/tool_parsers/hunyuan_tool_parser.py:54-170` parses the native HY3
  XML tool envelope.
- `vmlx_engine/server.py:1889-1975` maps public reasoning controls into HY3's
  template contract; the normal Responses/Chat streamers keep reasoning,
  visible content, tool arguments, terminal, and usage events separate.
- `panel/src/main/model-config-registry.ts:256-259,752-810` derives HY3 parser,
  reasoning, MTP, and text-only capabilities from config/bundle metadata.

No source change was required by this current-head rerun.

## Real Electron load and single-model swap

The fully relaunched Electron dev app was attached at CDP 9335 with
`VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev`. The real HY3
Sessions `Start` control was clicked while Qwen3.6-35B was running.

- The main-process log said single-model mode was stopping the Qwen session
  before starting HY3; Qwen PID 26427 disappeared and only HY3 PID 27632
  remained.
- `[Engine Manager] Found in PATH` resolved the project
  `/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- The session loaded with `last_request_time=null` and no UI error toast.
- The real argv used `--tool-call-parser hunyuan`, `--reasoning-parser qwen3`,
  Auto tools, paged cache, block-disk L2, `--native-mtp-depth 1`, deterministic
  MTP policy, and JIT.
- UI card/header label: `JANG_2K (2.33b routed)`.

## Electron four-turn behavior

Fresh chat `02aabc6f-4b47-4379-8ff0-8543ec053dd0`:

1. Row 443, no tool: exact `HY3-CURRENT-T1-DONE VALUE=703`, non-empty visible
   content, separate 844-character reasoning, no warning.
2. Row 446: exactly one real `file_info({"path":"panel/package.json"})`, exact
   `HY3-CURRENT-T2-DONE SIZE=5.2 KB`, separate 1,263-character reasoning,
   192 `paged+disk+tq-native` cached tokens, no warning.
3. Row 449, no additional tool: exact same-chat recall
   `HY3-CURRENT-T3-DONE VALUE=703 SIZE=5.2 KB`, distinct 483-character
   reasoning, 3,776 `paged+disk+tq-native` cached tokens, no warning.
4. Electron Stop/Start replaced PID 27632 with 29852. Before the request, L1
   and MTP request counters were empty while 68,667 L2 tokens remained. Row 452
   then exact-recalled both facts without a tool and restored 4,655/4,872
   tokens as `paged+disk+tq-native`.

The browser MutationObserver proves actual paint rather than terminal batching:

- Turn 2: 1,502 mutations; the visible final grew character-by-character from
  `HY3-CURRENT-T2-` to `...SIZE=5.2 KB` over about 264 ms after the tool.
- Turn 3: 655 mutations; the final marker grew progressively over about 292 ms.
- Restart turn 4: 521 mutations; the final marker grew progressively over about
  226 ms.

All four reasoning strings are different; there is no stale reasoning replay,
reasoning leakage, empty final, loop, warning, or unreported truncation in these
rows.

## Raw `curl -N` Responses and Chat proof

Each request has both raw SSE and curl `--trace-time` evidence.

Responses:

- No-tool: 215 reasoning deltas, 14 content deltas, exact
  `HY3-API-RESP-NO-DONE VALUE=42`, no call, one `response.completed` with usage.
- Required tool: 97 reasoning deltas, two argument deltas, exactly one valid
  `file_info(panel/package.json)`, one completed terminal.
- Real-result follow-up: 107 reasoning and 17 visible deltas, no second call,
  exact `HY3-API-RESP-FOLLOW-DONE SIZE=5.2 KB`, one completed terminal.

Chat Completions:

- No-tool: 142 reasoning and 13 content deltas, exact
  `HY3-API-CHAT-NO-DONE VALUE=42`, `finish_reason=stop`.
- Required tool: 112 reasoning deltas, exactly one valid `file_info`,
  `finish_reason=tool_calls`.
- Real-result follow-up: 89 reasoning and 16 content deltas, no second call,
  exact `HY3-API-CHAT-FOLLOW-DONE SIZE=5.2 KB`, `finish_reason=stop`.
- All Chat streams had zero ordinary chunks with non-null usage, exactly one
  choices-empty terminal usage chunk, and one `[DONE]`.

## Native MTP and cache truth

Before the process replacement, ten current requests recorded 1,194 drafted and
497 accepted tokens (41.62% acceptance). After restart, row 452 independently
recorded depth-1 execution with 87 drafted and 35 accepted tokens (40.23%), 88
MTP forwards, and no fallback.

After the restart row, health reported:

- one scheduler hit saving 4,655 tokens;
- 73 actual block-disk reads, all 73 `tq_native_hits`;
- five current q4 native-TQ disk writes;
- zero resident L1 bytes, so the payload was reconstructed from L2 rather than
  retained model-process RAM.

The restored prompt includes prior visible outputs produced while native MTP was
active. This proves accepted/main-history tokens reach the ordinary q4
stored-prefix boundary and are reusable. It does **not** claim that rejected
speculative draft tokens are persisted or compressed; they must not become
conversation prefix state.

## Focused tests

Current source passed 318/318 across HY3 native-MTP, depth/telemetry,
autodetection/policy, Hunyuan parser, TurboQuant cache/clone/disk/paged, and
reasoning/tool-interaction suites. See `focused-tests.txt`.

## Retained scope

- `VERIFIED-LIVE_SCOPED`: current Auto/D1 Electron load, four-turn history/tool
  behavior, progressive paint, literal curl Responses/Chat tool continuations,
  MTP counters, q4 L2 restart restore, and single-model swap.
- `PARTIAL`: broader long/stochastic soak and a new current-source MTP-Off versus
  D1 performance A/B. The earlier controlled D1 speed gate remains evidence but
  was not repeated here.
- `N/A`: VL/video/audio for this text-only bundle.
- No packaging, signing, notarization, or release readiness is inferred.

## Evidence

- `hy3-ui-loaded.png`, `hy3-process-argv.txt`
- `hy3-electron-three-turn.png`, `hy3-electron-four-turn-rows.json`
- `hy3-electron-restart-recall.png`
- `hy3-health-after-electron.json`, `hy3-health-before-restart.json`
- `hy3-health-after-restart-before-turn.json`, `hy3-health-after-restart-turn.json`
- `responses-*.json`, `responses-*.sse`, `responses-*.trace`
- `chat-*.json`, `chat-*.sse`, `chat-*.trace`
- `api-summary.json`, `electron-paint-summary.txt`, `focused-tests.txt`
