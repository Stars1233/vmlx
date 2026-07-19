# MiniMax-M3 current-head Auto/tool stream recheck

Date: 2026-07-19

Source head: `0f08edd04` with runtime repair `359ce6b2b`

Verdict: scoped `PASS-LIVE` for current-head text Auto reasoning, ordinary
answer streaming, required tool generation, real tool-result continuation,
Chat Completions, Responses, and the real Electron renderer. Overall M3 and
release status remain `PARTIAL`.

## Bundle and runtime identity

The visible Sessions-card Start button loaded
`/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small` as PID 2277. The
same single-model transition stopped Bonsai; SQLite and `ps` showed exactly
one running local engine.

Bundle truth was re-read before generation:

- `config.json`: `model_type=minimax_m3_vl`, architecture
  `MiniMaxM3SparseForConditionalGeneration`, and a real CLIP-style vision
  configuration with four frames per vision segment.
- `jang_config.json`: affine mixed `JANG_2L`, not JANGTQ/MXTQ and not base
  MLX MXFP; architecture `gqa+msa_sparse`, cache `kv+msa_index_dual`, vision
  and MoE enabled.
- `generation_config.json`: temperature 1.0 and top-p 0.95 bundle defaults.

The real process argv selected `minimax_m3` reasoning and tool parsers,
prefix cache, paged cache (64-token blocks, 1,000 blocks), and block-disk L2.
Health reported native schema `minimax_m3_msa_v1`, dense KV layers 0-2,
sparse MSA/index layers 3-59, and components `attention_kv`, `msa_idx_keys`,
and `absolute_block_index`. Generic TurboQuant KV is correctly forced Off for
this typed tuple. No q4/q8 generic-TQ claim is made for M3.

The real Chat Settings drawer showed Auto reasoning selected, Responses wire,
built-in tools On, blank model-default Max Tokens, temperature 1.0, and top-p
0.95. `m3-current-chat-settings.png` preserves that state.

## Current Electron multi-turn proof

Fresh no-tool Auto row 388:

- 2,745 characters of separately persisted reasoning;
- coherent non-empty 240-character visible answer ending
  `separation confirmed.`;
- no tool call and no warning;
- 128 cached tokens with `paged+disk`.

The CDP observer recorded 3,116 mutations and progressive visible synthesis
through the final sentence, rather than a terminal batch.

The next turn in the same chat requested one real `file_info` call. Row 391:

- executed exactly `file_info({"path":"panel/package.json"})` once;
- received the real metadata containing `Size: 5.2 KB`;
- retained 1,905 characters of distinct pre/post-tool reasoning;
- progressively painted `panel/package.json is` -> `... 5` -> `... 5.2 KB`
  -> the exact one-sentence final;
- stored no warning and reused 8,980 prompt tokens as `paged+disk`.

The mutation trace recorded 2,374 UI changes, a visible reasoning rail, the
Info tool card, and character-level post-tool content. Screenshots:
`m3-current-ui-notool-auto.png` and
`m3-current-ui-tool-multiturn.png`.

## Current raw Responses and Chat proof

`m3-current-protocol-events.json` preserves every timed SSE event generated
by PID 2277 after the shared Responses finalizer repair.

| Case | Reasoning deltas | Content/tool deltas | Terminal |
|---|---:|---:|---|
| Responses Auto, tools offered but unnecessary | 263 | 54 content | `response.completed` |
| Responses required tool | 42 | one call / two argument chunks | `response.completed` |
| Responses real-result continuation | 27 | 15 content | `response.completed` |
| Chat Auto, tools offered but unnecessary | 115 | 50 content | `stop`, `[DONE]` |
| Chat required tool | 42 | two tool chunks | `tool_calls`, `[DONE]` |
| Chat real-result continuation | 33 | 14 content | `stop`, `[DONE]` |

Both tool requests assembled exactly one
`file_info({"path":"panel/package.json"})`. Both result continuations emitted
fresh separate reasoning and a progressive visible answer reporting 5.2 KB.
There was no repeated call, truncated argument, reasoning-to-content leak,
missing visible answer, incomplete terminal, or delayed all-at-once final.

After the raw matrix, health was idle and recorded nine scheduler hits,
11,358 tokens saved, 5,687 indexed native M3 tokens, 66 request-local disk
hits, 405 block-store disk hits, 27 writes, and zero native-TQ writes/hits.

## Focused tests and packaging blocker

The first test invocation lacked the remote Node path and stopped at the
bundled-Python verifier with `node: command not found`. Repeating with
`PATH=/Users/eric/.local/node/bin:$PATH` exposed the real gate:

```text
RELEASE BLOCKED - bundled vmlx_engine/server.py content drift
source sha256 : 8e462960f6bf97385a0d89cc4b59c58a7f0ccb17740aa4655bab4decd3c498f3
bundled sha256: 193ad56245e0a234803a7a6ac134affe0607366ee77168a6fcf3d8ee3236e92e
```

That drift is expected after `359ce6b2b`, but it is a real release blocker.
It was not hidden or fixed by editing the test. The next release packaging
chain must run `panel/scripts/bundle-python.sh` before build/sign/notarize.

With only that packaging verifier explicitly deselected, the current M3,
reasoning, streaming, answer-pass, cache, and VL/video selection completed:

```text
759 passed, 46 skipped, 1 deselected
```

This is a focused runtime/source pass, not a full-suite or package pass.

## Remaining boundaries

- `BUNDLE-PYTHON-SOURCE-DRIFT` is `OPEN / RELEASE-BLOCKED` until the release
  cutoff rebundles and verifies the current engine.
- Current text Auto/tool protocol behavior is scoped `PASS-LIVE`, but larger
  videos, alternate-video isolation, digit OCR exactness, terminal delay,
  REAP32 memory headroom, cancellation/disconnect repeats, and signed-app
  rechecks remain separate M3 rows.
- Other reasoning/parser families do not inherit this live result from shared
  source alone.

## Preserved evidence

- `m3-current-protocol-events.json`
- `m3-current-health-start.json`
- `m3-current-health-after.json`
- `m3-current-message-rows.json`
- `m3-current-ui-loaded.png`
- `m3-current-chat-settings.png`
- `m3-current-ui-notool-auto.png`
- `m3-current-ui-tool-multiturn.png`
