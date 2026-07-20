# vMLX 1.6.13

Streaming-failure integrity checkpoint.

## Highlights

- Preserves progressive partial output, authoritative terminal usage, and
  immediate same-chat recovery when Chat Completions or Responses fails after
  streaming has begun.
- Keeps UI-only interruption notices out of subsequent model history while
  retaining the safe visible prefix for the user.
- Makes Anthropic Messages and all Ollama streaming routes terminate engine
  failures in their native wire format. Ollama chat, templated generate, and
  raw generate emit an `error` object and never synthesize a false successful
  `done:true` terminal after a failure.
- Isolates the packaged engine's Python import path so an unrelated top-level
  `mlx` directory in the app launch cwd cannot shadow the bundled MLX runtime.
- Retains the v1.6.12 architecture-aware cache, model-derived settings,
  gateway-lifecycle, reasoning/tool streaming, and scoped media checkpoint
  evidence without promoting untested model-family or stress rows.

## Verification boundary

- Current full Python and panel suites, TypeScript typecheck, bundled-runtime
  source/import verification, and production Electron builds are required for
  this exact versioned source.
- Both Sequoia-compatible and Tahoe-native DMGs must pass Developer ID signing,
  fresh Apple notarization, stapling, Gatekeeper verification, and isolated
  installed-app smoke before publication.
- Literal unbuffered HTTP failure/recovery pairs cover Chat Completions,
  Responses, Anthropic Messages, Ollama chat, templated generate, and raw
  generate. Installed-app proof is scoped to release packaging and a real UI
  Start/generation/Stop smoke; it does not rerun every retained model row.
- The final installed Sequoia checkpoint completed a three-turn
  reasoning/tool/recall loop and a process-restart q4 TurboQuant L2 restore;
  Sequoia and Tahoe both completed progressive reasoning/content API streams.

## macOS builds

- Sequoia/Sonoma-compatible build (`macosx_14_0_arm64`; macOS 14.5+):
  `vMLX-1.6.13-sequoia-arm64.dmg`
- Tahoe-native build (`macosx_26_0_arm64`):
  `vMLX-1.6.13-tahoe-arm64.dmg`

## Known follow-up

This is a public usable checkpoint, not closure of every retained family or
stress row. Broader signed-app model repetition, remaining parser-family tool
rows, long/stochastic reliability and latency work, additional media/audio
axes, gateway/network-loss soak, and the family-specific PARTIAL rows in the
master matrix remain deferred.

Gemma strict marker-only compliance is also retained as partial: the model can
add coherent explanatory prose even when asked for only a marker. The release
does not rewrite output or silently coerce sampling to hide that behavior.
