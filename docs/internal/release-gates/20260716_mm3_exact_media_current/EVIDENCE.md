# MiniMax-M3 exact media, tool finalization, and typed-cache gate

Date: 2026-07-16

Model: `/Volumes/EricsLLMDrive/jangq-ai/MiniMax-M3-Coder-Small`

Current source after repair: pending commit from `2f5b7786d`

Verdict: `SCOPED_PASS_WITH_RECORDED_OCR_FORMAT_MISSES / BROAD_RELEASE_GATE_OPEN`

## Source trace and repair

- `vmlx_engine/scheduler.py` now treats a missing `_uses_openpangu_cache` flag as false in every post-init request/store branch. This restores MiniMax-M3 cache fixtures that construct a partial Scheduler and prevents an unrelated family flag from aborting typed M3 store/fetch.
- Current M3 health reports native schema `minimax_m3_msa_v1`: dense KV layers 0-2, sparse MSA/`idx_keys` layers 3-59, absolute block indexes, generic TurboQuant off, and generic stored KV quantization off.
- `/v1/capabilities` advertises text, vision/image, and video runtime support. `health.mtp.vl_runtime_available=false` is an MTP-artifact field: this bundle has vision weights but no indexed MTP tensors and is not used as the media-support gate.

## Live Electron verification

All media rows used Responses streaming with built-in coding tools visibly enabled.

- Row 1956, punctuation-heavy OCR: expected `ZXQ-7294-M3-OCR`, actual `Z XQ–7294–M3–OCR`. Recorded strict-format FAIL; no zero-tool dead end.
- Row 1959, same-chat replacement attachment: expected `ZXQ7294M3OCR`, actual `Z XQ–7294–M3`. Recorded FAIL and treated as possible history/attachment contamination rather than a pass.
- Row 1962, fresh-chat high-contrast control: exact `BANANA8426`, no tool call, 32 tokens, 4.4 s. This proves the current tools-enabled image path and exact alphanumeric OCR on an uncontaminated chat.
- Row 1965, same chat after the image: exactly one `file_info(panel/package.json)` plus exact `MM3-TOOL-CURR1-DONE`; 64 native paged tokens reused.
- Row 1968 read the two video frames correctly but inserted spaces around the requested separator: `FRAME START 2468 | FRAME END 9753`. Recorded strict-format PARTIAL.
- Row 1971, no-reattach follow-up: exact `FRAME START 2468|FRAME END 9753`; 128 native paged tokens reused.
- A visible Electron Stop/Start replaced PID 99564 with PID 1794. Row 1974 then made exactly one `file_info(README.md)`, returned exact `MM3-RESTART-CURR1-DONE`, and restored 128 `paged+disk` tokens.
- After the scheduler source repair, a second visible Stop/Start loaded PID 2921. Row 1977 made exactly one `file_info(pyproject.toml)`, returned exact `MM3-SCHED-CURR1-DONE`, and restored 449 `paged+disk` tokens on the edited code.
- Final health reports `reconstructed=true`, `dequantized=false`, 4,548 cache-hit tokens across two current-process requests, 151 disk blocks / 9,437 tokens, and zero native TQ reads/writes.

## Tests

- 34/34 MiniMax-M3 cache/loader tests passed after the repair (four were red before it).
- 3/3 MiniMax server finalization tests passed.
- 4/4 media-cache contracts passed.
- 75/75 openPangu model/parser/tool-prompt regressions passed.
- 581/581 engine-audit tests passed.
- 8/8 native TQ paged-block tests passed.
- Python compile and `git diff --check` passed.

## Boundary

- Exact alphanumeric OCR and two-frame video recognition pass, but punctuation normalization remains model-sensitive; the two strict misses are retained in this bundle.
- MTP is unavailable for this exact artifact because config metadata expects one next-token layer while the weight index contains zero `mtp.*` tensors. No MTP claim is made.
- Long-context soak, concurrency, and full external-protocol coverage remain open. No release surface changed.
