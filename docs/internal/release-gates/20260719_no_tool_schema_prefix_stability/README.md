# Explicit no-tool request-shape and prefix stability

Status: `VERIFIED-LIVE` for the shared Electron Responses/Chat request-builder
fix at source commit `258cf16f91d4629acd0b71d77b407e9683ef3777`.

## Artifact and UI configuration

- Real Electron dev app on CDP 9335, relaunched with
  `VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev` and the repo
  `.venv/bin` first in `PATH`.
- Startup log contained
  `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
- Model: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M2.7-Small-JANGTQ`.
- Bundle truth: `model_type=minimax_m2`, 62 full-attention layers,
  `weight_format=mxtq`, profile `JANGTQ2`. This is JANGTQ/MXTQ, not affine
  JANG and not base MLX MXFP.
- Saved UI cache policy: Prefix On, Paged On, 64-token blocks, four cache
  blocks, Block Disk L2 On, legacy Prompt Disk Off, stored-codec Auto.
- Effective argv and health showed `--use-paged-cache`,
  `--max-cache-blocks 4`, `--enable-block-disk-cache`, and q4 native-TQ
  stored-prefix encoding.

## Live failure before the fix

The chat kept its persistent built-in-tools toggle On, while both user turns
explicitly said not to call tools. The panel correctly recognized that phrase
but still sent the complete tool catalog together with `tool_choice=none`.

The first Responses request logged `has_tools=true` and rendered 297 prompt
tokens. The same-chat follow-up replayed the prior reasoning/output items,
then MiniMax's tool-template fallback injected the catalog at the prompt
front. Its prompt jumped to 1,185 tokens, the first cached blocks no longer
matched, and health recorded zero hits plus three L1 evictions. The visible
answer was coherent, but this was a cache/request-shape failure and was not
counted as a pass.

## Root cause and source repair

`panel/src/main/ipc/chat.ts` already derives `userForbidsToolCalls` from a
guarded current-turn parser. The Responses and Chat builders nevertheless
always attached `availableToolDefinitions()` whenever the persistent tools
toggle was enabled.

Both wire builders now omit the tool definitions when
`userForbidsToolCalls` is true. Omitting unavailable schemas is semantically
equivalent to `tool_choice=none`, removes unusable prompt cost, and preserves
prefix identity. Normal tool-enabled turns still send the same definitions.

`panel/tests/tool-auto-continue.test.ts` pins the two shared wire branches and
the absence of the obsolete `obj.tool_choice = "none"` path.

Focused validation:

```text
tool-auto-continue.test.ts + tool-history-replay.test.ts: 24 passed
TypeScript typecheck: passed
```

## Current-source live Electron result

After a full Electron main-process relaunch, the fresh base request logged:

```text
request_shape ... wireApi="responses" ... has_tools=false
```

It returned exact visible `M27-PAGED-FIX-BASE-DONE` with 188 prompt tokens and
separate reasoning. The same-chat follow-up also omitted tools, restored 192
tokens as `paged+disk+tq-native`, retained a separate fresh reasoning rail,
and returned exact visible
`M27-PAGED-FIX-FOLLOW-DONE VALUE=R-64920` with no warning.

Health recorded one cache-hit request, three q4-native block hits, successful
reconstruction, and no dequantization. This particular cache configuration is
frugal block-L2 mode: L1 keeps the chain index while disk is authoritative for
the q4-native payload, so this is deliberately classified as same-process L2
reuse rather than a RAM-resident payload hit.

## Evidence files

- `m27-paged-settings-before-restart.png`
- `m27-paged-loaded.png`
- `m27-paged-fix-base.png`
- `m27-paged-fix-follow.png`
- `m27-health-after-fix-base.json`
- `m27-health-after-fix-follow.json`
- `m27-paged-argv.txt`
- `m27-paged-session.json`

## Remaining gate

This closes the shared explicit-no-tool request-shape defect. It does not close
the parent paged-cache gate. RAM-resident reuse must be rerun with block L2
explicitly Off; then block L2 must be re-enabled for forced eviction/refault,
partial-block reuse, and process-restart disk-only restore. Required-tool
turns remain a separate live acceptance axis.
