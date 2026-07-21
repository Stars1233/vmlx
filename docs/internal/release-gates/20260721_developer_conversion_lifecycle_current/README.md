# Developer Tools conversion lifecycle — current source

Date: 2026-07-21

Status: `VERIFIED-LIVE_SCOPED` for shared Developer Tools process ownership,
navigation/reconnect state, cancellation, and post-cancel error recovery.
General quantizer correctness remains `PARTIAL`.

Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Base source cutoff: `280e29280da29e5f80885ee4601b8241e71fb3f5`

## Why this gate was selected

The affine JANG_4M happy path and produced-model agent loop already had a
dedicated real Electron proof. Repeating that model/profile would not close
the remaining user-facing failure modes. Source inspection instead found two
shared lifecycle defects affecting both Model Converter and Model Doctor:

1. process ownership and cancellation lived in module-global
   `activeProcess`/`cancelled` variables, while app shutdown duplicated the
   termination logic and cleared ownership before child settlement;
2. the reconnect contract restored logs only while `running=true`. A
   conversion finishing while the user was elsewhere lost its terminal
   result, and a running conversion returned with empty source/output fields
   and the default profile instead of the actual child arguments.

## Current production source trace

- `panel/src/main/ipc/streaming-operation-lifecycle.ts` is the single owner for
  active process, per-child cancellation, force-kill timer, one-time
  settlement, and durable terminal result.
- `panel/src/main/ipc/developer.ts` uses that owner for start, IPC Cancel, child
  close/error, `getBufferedLogs`, and app-quit cleanup. The second manual kill
  branch and lossy globals were removed.
- `panel/src/renderer/src/components/tools/useStreamingOperation.ts`
  subscribes before taking the reconnect snapshot, closes the snapshot versus
  completion race, and restores completed logs/result plus the actual command
  descriptor.
- `ModelConverter.tsx` reconstructs source, output, affine JANG profile/method
  or MLX options from the actual CLI descriptor. `ModelDoctor.tsx` restores its
  model and inference choice from the same shared contract.
- preload and renderer types expose the result/descriptor without bypassing
  Electron IPC.

This change does not add a generic JANGTQ/MXTQ converter. The UI's ordinary
JANG profiles remain affine JANG. The source-proven ZAYA branch remains the
only special JANGTQ conversion dispatch in this CLI path.

## Focused validation

From `panel/`:

```text
npm test -- --run tests/developer-operation-lifecycle.test.ts tests/developer-tools.test.ts
2 files passed; 97 tests passed

npm run typecheck
tsc --noEmit
```

The new lifecycle tests pin durable completion, per-child cancellation,
duplicate close/error suppression, protection from a late old-child terminal,
concurrent-start rejection, and cancel-without-child behavior.

## Real Electron/CDP proof

The dev app was fully relaunched from current source with the existing test
profile, CDP 9335, and the current checkout on `PYTHONPATH`. Startup reported:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
[Engine Manager] Version: 1.6.14
```

No server/model session was started for this lifecycle gate.

### Running operation survives navigation with truthful settings

The real Tools -> Convert Model form selected affine `JANG_4L`, MSE, source
`/Users/eric/models/OsaurusAgent-9b-BF16`, and an explicit temporary output.
After the real Convert button started the venv CLI, the UI navigated to Server
and back to Tools. The restored page still showed:

- the exact source path;
- the exact output path;
- selected `4L — Premium`, not the default 3M;
- `Cancel Conversion`;
- the restored output log;
- one live `vmlx-engine convert ... --jang-profile JANG_4L` child.

Screenshot: `descriptor-return-running.png`.

### Current-source cancel and terminal persistence

A second attempt used the same real Convert button, then a same-page scheduled
click on the rendered `Cancel Conversion` button to eliminate CDP reconnect
latency. The UI navigated away and back after termination. Current source then
showed:

```text
Conversion cancelled
```

alongside the restored source, output, and 4L selection. No converter child
remained and the temporary output contained no files.

Screenshot: `current-cancel-descriptor-visible.png`.

### Post-cancel recovery is not poisoned by stale cancellation

The next real Convert click used an explicit nonexistent source. It completed
as an ordinary failure, not a stale cancellation. The visible page showed
`Conversion failed` and the exact actionable source-not-found log. Renderer
IPC returned:

```json
{
  "running": false,
  "result": {
    "success": false,
    "cancelled": false,
    "error": "Conversion failed"
  },
  "operation": {
    "command": "convert",
    "args": [
      "/private/tmp/codex-nonexistent-source-current-20260721",
      "--jang-profile",
      "JANG_4L"
    ]
  }
}
```

Screenshot: `current-recovery-visible.png`.

## Probe cleanup

All temporary completed, cancelled, and failure output directories under
`/private/tmp/codex-convert-*20260721` were removed after evidence capture.
No official bundle was modified. No converter child remained.

## Honest boundary and newly observed issue

This closes the shared navigation/cancel/error lifecycle only. Still open:

- generic JANGTQ/MXTQ Hadamard-codebook conversion outside the special ZAYA
  branch;
- force-overwrite, low-disk, unwritable-volume, calibration/AWQ/imatrix, and
  large-MoE conversion rows;
- resume of an interrupted quantization job (distinct from UI reconnection to
  a still-running child);
- complete Chat/Responses/Anthropic/Ollama and cache/media certification for
  every newly produced profile.

One incidental affine JANG_3M probe finished before its first cancel click and
was therefore retained as a completed-result navigation check, not promoted to
a family release gate. Its minimal converter smoke generated
`Paris.\n<think>` and still labeled the load/decode check `PASS`. That proves
the current smoke is not a chat-template/output-quality certification. The
runtime must continue to require a separate real Electron multi-turn proof;
the output was not rewritten, rationalized, or called coherent. The temporary
5.36 GB output was removed.
