# DSV4 native composite RAM and Block Disk L2 evidence

Scope: current DSV4 `deepseek_v4_v7` native composite cache, 2026-07-17.
Generic TurboQuant KV is disabled for this architecture.

- Cold ORCHID request used a 3,245-token prompt.
- Identical resident reuse restored 3,244 tokens as `paged+dsv4` with 1.50s
  TTFT.
- Visible Electron process restart without clearing Block Disk L2 restored the
  same 3,244-token boundary as `paged+dsv4+disk`.
- Changed TULIP input was isolated and did not leak the ORCHID control.
- Raw Responses produced progressive deltas, matching done text, and one
  completed terminal.
- Pre-prompt restart health records the model already materialized, L1 empty,
  and 46 DSV4 Block Disk records present before the first request.

The JSON, SSE timing trace, health capture, and Electron screenshots in this
directory are the retained live artifacts. This is a scoped short-prefix
restore/isolation proof; it does not clear the long-context performance or
strict-output quality gates.
