# Gemma4 mixed-SWA current-source release gate

Status: **PASS for this scoped Gemma4 cache/stream/tool gate at `7b62ea4a3`**.
This is not a whole-release verdict.

## Source trace

- `3e54a4509` makes Gemma4's typed mixed-SWA path default to prefix + paged +
  block-disk L2 in the Electron registry and keeps the generic multimodal guard
  from overriding that architecture-specific policy.
- `7b62ea4a3` makes `native_cache` distinguish live TurboQuant encoding from
  storage-boundary TQ4 using the same computed TurboQuant status used by the
  top-level health fields.
- Focused panel contracts: 377 passed across registry, settings-flow, and
  cache-control tests; TypeScript typecheck passed.
- Focused Python contracts: 4 passed for Gemma4/Step/MiMo mixed-SWA native-cache
  reporting.

## Live Electron proof

- Real `Launch Session` button, session
  `7bf80d27-3937-426e-90a8-6152f2159f3b`, project engine 1.6.11.
- Visible defaults: Prefix ON, Paged ON and architecture-locked, Block Disk L2
  ON, legacy disk OFF, Auto codec shown as TQ4 full-attention KV plus native
  rotating SWA.
- Launch argv contains paged/block-L2/parser/stream flags recorded in
  `ui-proof.json`.
- Tool turn executed exactly one real `file_info(panel/package.json)` and
  returned the 5.2 KB result. Two follow-ups were coherent and reused 3,584 and
  3,821 tokens with `paged+mixed_swa`.
- After Stop/Start without clearing L2, a fresh identical tool prompt restored
  149 tokens as `paged+mixed_swa+disk`, executed exactly one real tool, and
  completed with non-empty visible content.

## Raw API proof

- Chat Completions: 100 reasoning deltas, then 93 content deltas, one `stop`.
- Responses: 99 reasoning deltas, then 91 content deltas, one
  `response.completed`.
- The visible answer streamed over roughly 1.1 seconds on both surfaces; it was
  not delivered as a single post-reasoning batch.
- Replaying the byte-identical raw request after restart restored 55 tokens as
  `paged+mixed_swa+disk`; health recorded disk reconstruction/dequantization and
  TQ-native hits.

The first history-extension turn after restart was not byte-identical and
therefore missed L2; it is retained in SQLite but is not counted as a disk-hit
proof. The identical raw and fresh-Electron replays are the persistence rows.
