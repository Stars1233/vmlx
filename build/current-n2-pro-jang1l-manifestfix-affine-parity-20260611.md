# N2 Manifestfix Affine Parity Probe - 2026-06-11

Artifact:

`/Volumes/EricsLLMDrive/jangq-ai/Nex-N2-Pro-JANG_1L-manifestfix-20260611`

## Result

Status: vMLX affine primitive parity passed for sampled routed expert slices.

This no-full-model-load probe compared the exact MLX primitive used by
`nn.QuantizedLinear`:

```python
mx.quantized_matmul(
    x,
    weight,
    scales=scales,
    biases=biases,
    transpose=True,
    group_size=group_size,
    bits=bits,
    mode="affine",
)
```

against explicit:

```python
x @ mx.dequantize(weight, scales, biases, group_size, bits).T
```

## Coverage

- Layers: `0`, `1`, `30`, `59`.
- Projections: `switch_mlp.gate_proj`, `switch_mlp.up_proj`,
  `switch_mlp.down_proj`.
- Expert slice: expert `0`.
- Dtypes: synthetic `float16` and `bfloat16` activations.
- Cases: `12` projection/layer cases, `24` dtype comparisons.

Tensor shape examples:

- `gate_proj.weight`: `(512, 1024, 256)` uint32.
- `gate_proj.scales`: `(512, 1024, 32)` float16.
- `down_proj.weight`: `(512, 4096, 64)` uint32.
- `down_proj.scales`: `(512, 4096, 8)` float16.

These decode as 2-bit, group-size 128, matching the module-keyed config
overrides.

## Metrics

No cases failed the threshold:

- relative max error threshold: `0.02`
- cosine threshold: `0.999`

Worst observed case:

- layer `0`, `down_proj`, `float16`
- relative max error: `0.00256238527769491`
- cosine: `0.9999967217445374`
- `mx.quantized_matmul` time for the slice: `0.000256s`

## Classification

This reduces the likelihood of a generic `mx.quantized_matmul` argument/order,
dequant, bit/group-size, or upcast bug for sampled N2 routed affine slices.

It does not prove the full routed MoE aggregation path healthy. The remaining
runtime-side areas are:

- expert gather/routing/aggregation around `switch_mlp`;
- full-model memory-pressure synchronization;
- full decode timing across routed experts, shared experts, linear-attn/SSM,
  and lm_head.

If JANG tensor-quality probes show severe source-vs-dequant error on routed
2-bit tensors, routed 2-bit sensitivity becomes the stronger root cause.

## Proof Files

Full JSON:

`build/current-n2-pro-jang1l-manifestfix-affine-parity-20260611.json`
