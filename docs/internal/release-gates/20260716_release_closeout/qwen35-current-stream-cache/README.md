# Qwen 3.6 35B current-source streaming and cache proof

Status: `PASS-LIVE` for streamed Chat multi-turn, Electron two-turn tool use,
q4 attention-KV plus native SSM/GDN prefix reuse, and process-restart Block
Disk L2 restore. Status remains `PARTIAL` for strict sampled reliability because
one Responses run repeated its requested marker; no transport-side replay was
found and no synthetic deduplication was added.

## Artifact and source trace

- Artifact: `dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK` on port 8029. The bundle
  name and tensor index do not declare MTP, so the nested architecture hint is
  correctly inactive. This row is not used as an MTP proof.
- `/health` identifies `qwen3_5_moe` hybrid state with ten attention-KV layers,
  thirty native companion layers, q4 TurboQuant stored-prefix encoding,
  paged-prefix RAM, Block Disk L2, and SSM companion disk restore.
- `vmlx_engine/reasoning/think_parser.py:143-193` maps each new delta to either
  reasoning or content once.
- `vmlx_engine/server.py:18671-18830` accumulates parsed Responses deltas and
  selects the emitted reasoning/content rail. `server.py:18873-18900` emits one
  SSE event for each selected delta. The bounded visible-answer pass at
  `server.py:19397-19425` is gated on an empty visible answer and therefore did
  not replay the already-emitted content in the retained duplicate sample.
- The cache implementation selects eligible attention slots from the runtime
  layer graph, keeps hybrid companions native, and stores/restores native TQ
  blocks plus the independently fingerprinted SSM companion boundary. The
  saved health records are the live ownership/accounting evidence.

## Raw streaming

- `q35-shared2.json`: Chat turn 1 emitted 379 reasoning and 14 content deltas;
  turn 2 emitted 448 reasoning and 23 content deltas, recalled the exact
  codeword, and reused 46 `paged+ssm` tokens. Both assembled the exact marker.
- The same file retains a Responses reliability miss: 339 reasoning and 25
  content deltas contained the requested marker twice. `output_text.done`
  matched those already-doubled deltas, so this was not a terminal-only replay.
- `q35-rep1.json` through `q35-rep3.json` reran the complete Chat/Responses
  harness at a 512-token budget and each produced one exact Responses marker.
  This does not erase the retained miss; it establishes that the duplication
  was not deterministic in the saved repeats and that this audit found no
  transport replay. The sampled reliability row remains open.

## Electron agent loop

- `q35-electron-shared2.json`: 56 reasoning events, 12 progressive content
  events, exactly one `file_info(panel/package.json)`, and exact final
  `Q35-ELECTRON-SHARED2-DONE`.
- `q35-electron-shared2-t2.json`: a second turn in the same Electron chat emitted
  149 reasoning and 14 progressive content events, called exactly
  `file_info(README.md)`, and exact-finaled
  `Q35-ELECTRON-SHARED2-T2-DONE`.
- `q35-electron-shared2-t2-final.png` visibly shows both distinct tool paths and
  both exact final answers in the current Electron UI.

## Long-prefix RAM and L2 restore

- `q35-tqfair2.json` used one exact 4,625-token prompt. Cold first content was
  8.8514s. The identical resident request restored 4,624 `paged+ssm` tokens in
  0.5543s, a 15.969x first-content speedup, with 0.274409s reconstruction.
- A visible Electron Save & Restart changed PID 88980 to 89919 without clearing
  L2. `q35-tqfair2-l2.json` then restored the same 4,624 tokens as
  `paged+ssm+disk` in 0.4785s with 0.240948s reconstruction; the next resident
  request used `paged+ssm` in 0.5596s. Every answer was exact and progressive.
- `q35-health-after-l2-current.json` records two scheduler hits / 9,248 saved
  tokens, 292 q4 native-TQ block hits including 73 disk hits, one real SSM disk
  hit, two companion stores, and zero unsafe KV-without-companion reuse.

## Release boundary

This closes the current-source streaming/cache/agent-loop row for this specific
non-MTP Qwen 35B artifact. It does not close Qwen 27B MTP, broader Qwen
variants, eviction on the current head, or the retained sampled duplicate.
