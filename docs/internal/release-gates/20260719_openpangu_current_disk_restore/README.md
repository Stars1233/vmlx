# openPangu current-head prompt-L2 restart recheck

Date: 2026-07-19
Source head: `117c3d20699316bdb2ee8a17d40c60b69e9f22b9`
Model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`
Status: **VERIFIED-LIVE scoped** for exact memory/SSD restore and streaming; **PARTIAL** for the new forward-suffix strict-format control.

This is a current-head recheck of the architecture-specific openPangu cache
route. It must not be cited as generic paged/block-L2 or TurboQuant proof.

## Bundle, launch, and gateway lifecycle

- The tested bundle is affine `JANG_3M` (`affine_quantized_matmul`), not
  JANGTQ/MXTQ and not base MLX MXFP.
- The real Electron Sessions-card Start action stopped Step and eagerly loaded
  only openPangu PID 65893 before a request. Health reported
  `model_loaded=true` and `last_request_time=null`.
- The process argv used `--no-paged-cache --enable-disk-cache`, the native
  openPangu tool/reasoning parsers, and no KV-quantization or block-L2 flag.
- A real Electron Stop/Start replaced the process with PID 66691. Before the
  first post-restart request, health reported zero prompt-cache memory entries,
  zero generic L1 indexed/resident bytes, and 6,502 prompt-L2 tokens on SSD.

Evidence: `openpangu-start-before-request.png`, `openpangu-argv.txt`,
`electron-engine-log.txt`, and `health-before-disk-restore.json`.

## Architecture-owned cache policy

Health and `source-trace.txt` agree on schema
`openpangu_v2_composite_v2`: MLA latent KV, DSA indexer state, rotating SWA
window state, and path-dependent convolution state. Its policy is exact typed
N-1 prompt snapshots with exact/forward-prefix prompt-L2 reuse. Generic paged
blocks, block-disk L2, and generic TurboQuant KV are unsupported and remained
off. The prompt snapshots are typed full precision.

Therefore arbitrary block-aligned partial reuse is **N/A for openPangu**. That
acceptance gate belongs to compatible full-KV/hybrid/mixed-SWA families; forcing
it here would restore an incomplete path-dependent state.

## Electron cold, memory, and SSD proof

Three fresh Electron chats used the same 818-token prompt and returned exact,
non-empty `PANGU-L2-A-DONE` content:

1. Cold: stored an 817-token typed boundary. The reasoning rail and visible
   answer painted progressively.
2. Same-process exact: restored 817 tokens as `memory`; TTFT was 0.79 seconds.
3. First turn after process replacement: with zero in-process entries before
   the request, restored 817 tokens as `disk`; TTFT was 0.49 seconds and health
   recorded one SSD hit with no reconstruction/dequantization/TQ.

The three reasoning strings differ byte-for-byte, so this did not reproduce
stale reasoning replay. Each visible answer is persisted separately from its
reasoning with no tool call and no warning.

Evidence: `openpangu-ui-cold.{json,png}`, `openpangu-ui-warm.{json,png}`,
`openpangu-ui-disk-restore.{json,png}`, and
`electron-assistant-rows.json`.

## Raw Responses streaming and forward-prefix control

`openpangu-responses-forward.json` contains current-source SSE events:

- cold A: 307 reasoning deltas, six content deltas, one text-done, one
  completed terminal, 48.7 ms from last content to completed;
- exact A: 307 reasoning deltas, six content deltas, one text-done, one
  completed terminal, 156.3 ms terminal gap, and a 566-token memory hit;
- forward B: a 592-token forward-prefix memory hit, 512 reasoning deltas, 12
  content deltas, one text-done, one completed terminal, and a 34.5 ms terminal
  gap.

The forward B output included both the earlier A marker and the requested B
marker. Cache selection and transport completed, but strict output fidelity is
**PARTIAL**. The control deliberately made A a literal prefix of B and the
model followed both conflicting visible instructions after consuming the
512-token reasoning budget. This result is retained, not reclassified as a
clean exact-format pass and not blamed on the official quant artifact.

## Verdict and remaining boundaries

- Exact RAM hit: **PASS-LIVE**.
- Exact first-turn SSD restore with no prior RAM entry: **PASS-LIVE**.
- Separate progressive Electron and Responses reasoning/content: **PASS-LIVE**.
- Terminal completion: **PASS-LIVE** for these bounded requests.
- Generic block partial reuse/TQ: **N/A**, architecture-incompatible.
- Forward-prefix cache selection: **PASS-LIVE**; strict B-only model output:
  **PARTIAL**.
- 512K soak, cancellation/disconnect/failure recovery, additional bit variants,
  and signed packaged-app repeat remain open campaign rows.

Overall release status remains `PARTIAL_NO_RELEASE`.
