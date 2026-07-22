# Public v1.6.15 Laguna mixed-bit provenance audit

Date: 2026-07-22

Verdict: `PUBLIC SIGNED APP 1.6.15 PASS FOR THIS DEFECT`; public Python/CLI
v1.6.15 dependency floor `FAIL`; stale installed 1.6.9 negative control
`FAIL-LIVE`; JANG 2.5.33 plus the vMLX 1.6.16 dependency guard
`VERIFIED-LIVE_SCOPED`. The broader v1.6.16 campaign remains
`PARTIAL / NOT RELEASE-READY`.

## Report investigated

An independent run reported:

```text
[dequantize] matrix (1,seq,576) vs scales (1,seq,48), group_size=64, bits=8
```

The report attributed 48 scales to a full 3072-wide weight and 576 to a sliced
Laguna attention/gating path. The first dimension inference is useful, but the
conclusion is not correct for the inspected S-2.1 artifacts:

- `48 = 3072 / 64` is the scales width for a 3072-input affine projection;
- `576 = 3072 * 6 / 32` is the packed width for that same projection at 6 bits;
- the bundle declares `model.embed_tokens` as 6-bit affine;
- `self_attn.g_proj` is declared 8-bit and does not have this packed shape.

The signature therefore means that a 6-bit packed projection was instantiated
or executed as 8-bit. It is not evidence that a 3072-wide weight was sliced to
576 in `LagunaForCausalLM.forward`.

## Source trace

Two signed apps coexist on the proof host:

- `/Applications/vMLX.app` is version 1.6.9. Its bundled
  `jang_tools/laguna/runtime.py` SHA-256 is
  `1ae98b927a89bccffcf9bf32ea0abf53107a0514ff441cb871d9a7b9075389f7`.
  Lines 185-199 read only the top-level `quantization.bits` value and pass it
  uniformly to `nn.quantize`.
- `/Applications/vMLX-1.6.15-Tahoe-Checkpoint.app` is version 1.6.15. Its
  bundled runtime SHA-256 is
  `4a531e91e8af94a36ee4b4af04f8cf7aade64ce836f79a415e974a95b18dfa05`.
  Lines 261-306 derive each affine module's true bit width from its packed
  weight and scales shapes before calling `nn.quantize`. The derivation permits
  2/3/4/5/6/8-bit modules and returns 6 for a `(…,576)` packed weight paired
  with 48 groups at group size 64.

The Sequoia and Tahoe 1.6.15 checkpoint apps have identical Laguna runtime and
model hashes. Both are Developer ID signed by ShieldStack LLC (team
`55KGF2S5AY`). The fixed runtime was already part of those app bundles.

That does **not** make the public Python/CLI 1.6.15 surface safe. Current PyPI
metadata for `vmlx==1.6.15` permits `jang>=2.5.29`. Published JANG wheels at
that older floor did not guarantee the per-module mixed-affine loader
contract, so a standards-compliant fresh Python install could resolve a stale
runtime and reproduce the defect. vMLX commits `b6d38eac7` and `e4c6762ce`
therefore raise the floor to `jang>=2.5.33`, reject stale mixed-affine Laguna
runtimes before model execution, and log the exact imported module/marker.

Both development venvs now import the published no-cache JANG 2.5.33 wheel
from physical `site-packages`, not either machine's dirty editable JANG tree.
Fresh wheel inspection recorded version 2.5.33, runtime marker 1, runtime SHA
`96091e33...`, and exact `(576,48,group_size=64) -> 6` inference. Runtime
provenance must always use `module.__file__`, capability marker, and source
hash—not package metadata or directory inventory alone.

## Current live positive controls

All tests below used the real S-2.1 bundles on the M5 Max proof host.

1. The exact signed 1.6.15 bundled engine loaded
   `Laguna-S-2.1-JANG_2L` with prefix, paged RAM, block-disk L2,
   memory-aware cache, and KV-cache quantization disabled. Its streaming Chat
   request completed progressively with exact visible content
   `LAGUNA-1615-NOCACHE-DONE`, a normal stop, and `[DONE]`.
2. The same signed bundled engine and cache-disabled flags loaded
   `Laguna-S-2.1-JANG_4M`. Its stream completed progressively with exact visible
   content `LAGUNA4M-1615-NOCACHE-DONE`, a normal stop, and `[DONE]`.
3. The exact signed 1.6.15 Tahoe Electron app was launched on isolated CDP
   9336. The real Sessions Start button eagerly loaded S-2.1 JANG_2L through
   its bundled Python, and a fresh UI chat exact-finaled
   `REL1615-LAGUNA-UI-DONE` with no warnings. This UI row used its saved default
   cache policy; it is not presented as the cache-disabled proof.

Retained artifacts:

- `r16-public-1615-laguna-nocache-health.json`
- `r16-public-1615-laguna-nocache-sse.txt`
- `r16-public-1615-laguna-nocache.log`
- `r16-public-1615-laguna4m-nocache-health.json`
- `r16-public-1615-laguna4m-nocache-sse.txt`
- `r16-public-1615-laguna4m-nocache.log`
- `r16-signed-1615-electron-tail.log`
- `r16-signed-1615-laguna-argv.txt`
- `r16-signed-1615-laguna-health.json`
- `r16-signed-1615-laguna-ui-pass.png`

## Live negative control

The stale signed `/Applications/vMLX.app` version 1.6.9 was run against the
same S-2.1 JANG_4M artifact with prefix, paged RAM, memory-aware cache, and
KV-cache quantization disabled. It reproduced the claimed defect exactly:

```text
Provided matrix of shape (1,54,576) and scales/biases of shape (1,54,48)
with group_size=64 and bits=8.
```

The server was stopped after capture and port 8064 was confirmed closed.

Retained artifacts:

- `r16-stale-169-laguna-negative-health.json`
- `r16-stale-169-laguna-negative-sse.txt`
- `r16-stale-169-laguna-negative.log`
- `local-dev-venv-refresh.txt`

## Operational conclusion

The signed v1.6.15 Sequoia/Tahoe apps are not affected by this exact fault,
but the public `vmlx==1.6.15` Python dependency contract is too weak and must
not be described as passing. The repaired public JANG package is 2.5.33; the
next vMLX Python release must require that floor and retain the runtime marker
guard. Anyone seeing the error must record the exact app/package version,
executable, `vmlx_engine.__file__`, `jang_tools.laguna.runtime.__file__`, and
runtime SHA before diagnosing model math.

Current-source Electron plus Chat/Responses proof for the repaired dependency
surface is retained under
`../20260722_jang_2533_laguna_distribution/`. This scoped result does not
close the v1.6.16 release matrix or Laguna's separate long-context,
Paged-Off restart, cache-eviction, four-protocol agentic, gateway, or latency
gates.
