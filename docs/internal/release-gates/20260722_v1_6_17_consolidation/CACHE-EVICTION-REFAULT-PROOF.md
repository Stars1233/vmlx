# Mandatory RAM-Eviction to SSD-Refault Cache Proof

Status: `REQUIRED / NOT UNIVERSALLY CLOSED`.

This is the minimum live-proof procedure for any claim that block-disk/L2
prefix caching works after an entry leaves in-memory paged cache. It is not
enough to show a warm hit while the prefix is still resident in RAM.

## Required test sequence

1. Record exact provenance:
   - source commit and executable/imported-module paths;
   - model bundle path and hashes or revision;
   - bundle-grounded cache architecture and effective settings;
   - in-memory block budget, prefix-cache RAM percentage, block size, L2 path,
     L2 GB cap, and TurboQuant/native-cache policy.
2. Clear only the isolated test cache and prove both RAM and disk start empty.
3. Submit prefix `A` with a unique token sequence and a deterministic,
   checkable suffix. Record the cold prefill token count, TTFT, output, RAM
   blocks, disk blocks, and cache counters.
4. Submit distinct `B`, `C`, `D`, and additional prompts until the configured
   RAM block budget must evict `A`. Do not infer eviction from prompt count:
   prove that `A` is absent from the RAM-resident index while its valid blocks
   remain in the L2 inventory.
5. Submit `A + NEW_SUFFIX`, where `NEW_SUFFIX` has never been cached.
6. Require a longest-contiguous-prefix match from SSD. The execution record
   must show:
   - matching blocks refaulted from disk;
   - only the unmatched suffix prefilling;
   - restored-token and disk-hit counters increasing by plausible amounts;
   - no false RAM-hit attribution;
   - coherent output equivalent to a cold control within the model's allowed
     stochastic behavior.
7. Compare cold and refault runs using raw timing:
   prompt tokens, restored tokens, actually-prefilled tokens, TTFT, prefill
   rate, decode rate, and end-to-end wall time. A log phrase alone is not
   performance proof.
8. Restart the model process and the Electron session, then repeat
   `A + ANOTHER_NEW_SUFFIX`. Require the on-disk blocks to survive and refault
   without relying on process-resident metadata.
9. Repeat the entire sequence with in-memory paged cache **On**:
   first prove L1/RAM lookup, then force eviction and prove fallback to L2/SSD.
10. Repeat the entire sequence with in-memory paged cache **Off**:
    block-disk caching must remain selectable and the replay must restore
    directly from SSD before prefilling only the unmatched suffix.
11. Set a deliberately small L2 GB cap, exceed it, and prove least-recently-used
    disk blocks are deleted while a surviving prefix still refaults correctly.
12. Set a deliberately small prefix-cache RAM percentage, exceed it, and prove
    old unused RAM blocks evict without crossing the configured ceiling.
13. Verify the UI warns when the selected RAM percentage is unsafe relative to
    physical device memory. The warning must not depend only on currently free
    memory.
14. Corrupt or remove one required cache companion artifact and repeat the
    request. The only acceptable outcomes are a safe full-prefill miss or a
    proven architecture-supported async rederive—never partial wrong-state
    restoration.

## Required architecture variants

The sequence above must be repeated on representatives of each materially
different cache state:

- standard full-attention KV with TurboQuant encode/decode;
- hybrid SSM/GDN state;
- mixed full-attention plus rotating/SWA state;
- typed CCA or other native companion state;
- MiniMax M3 dense KV plus sparse/MSA indexer state;
- DSV4 composite MLA/SWA/CSA/HCA state;
- OpenPangu native prompt-disk state.

TurboQuant q4 must apply only to cache components whose source and bundle
contract support it. Native SSM, rotating, sparse-index, composite, or prompt
state must be restored or rederived through its typed path, never flattened
into generic KV/TurboQuant blocks.

## Required retained evidence

Each closed row must retain:

- a machine-readable test manifest with provenance and effective settings;
- pre-run, post-store, post-eviction, post-refault, and post-restart health/cache
  snapshots;
- raw server log or structured execution events identifying RAM versus disk
  hits and the number of restored and actually-prefilled tokens;
- raw API capture and Electron screenshot for the replayed turn;
- cold and refault timing records;
- disk inventory before store, after eviction pressure, after L2-cap rotation,
  and after restart;
- output comparison and an explicit PASS/FAIL verdict for every step above.

## Evidence that does **not** close this row

The following remain `PARTIAL`:

- a second identical prompt that may still be resident in RAM;
- an exact-match replay with no new suffix;
- a disk store without a later disk hit;
- a `Cache disk hit` log line without counters, residency proof, and timing;
- a same-process restore without process/session restart;
- Paged-On proof used to infer Paged-Off behavior, or the reverse;
- restored token counts without coherent output comparison;
- one cache architecture used to infer all typed/native architectures;
- source inspection or unit tests without real model/API and Electron proof.
