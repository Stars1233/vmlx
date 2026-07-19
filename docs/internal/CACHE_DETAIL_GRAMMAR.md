# Cache Detail Grammar

`cache_detail` is an API/health observability label, not a scheduling control.
It reports which cache components actually participated in a request that used
cached prompt tokens.

Use `+` as the only composition operator:

```text
cache_detail := <component> ("+" <component>)*
component := paged | memory | prefix | disk | tq | tq-native | ssm | dsv4 | zaya_cca
```

Canonical labels:

| Label | Meaning |
| --- | --- |
| `paged` | In-process paged KV prefix hit. |
| `paged+disk` | Paged KV prefix hit that restored at least one block from block L2 disk. |
| `paged+disk+tq` | Prompt L2 TQ-native cache was promoted into paged KV storage. |
| `paged+ssm` | Hybrid KV+SSM prompt hit with SSM companion state available in process. |
| `paged+ssm+disk` | Hybrid KV+SSM prompt hit where paged blocks restored from block L2 disk and SSM companion state matched. |
| `paged+dsv4` | DSV4 native composite cache hit; generic SSM labels do not apply. |
| `paged+zaya_cca` | ZAYA CCA path-dependent cache hit. |
| `memory` | Memory-aware prefix cache hit. |
| `prefix` | Legacy trie prefix cache hit. |
| `disk` | Prompt-level L2 disk hit without paged promotion. |
| `disk+tq` | Prompt-level TQ-native L2 disk hit without paged promotion. |
| `disk+tq+tq-native` | Prompt-level L2 hit restored from the compact native TurboQuant safetensors record, then reattached to the model's q4/q8 cache objects without paged promotion. `tq-native` is evidence about the persisted record format; `tq` is the active stored-cache codec. |
| `paged+disk+tq-native` | Paged/block L2 hit whose persisted block payload used the native TurboQuant record format. |
| `paged+ssm+disk+tq-native` | Hybrid paged KV hit restored from native TurboQuant block L2 with the matching SSM/GDN companion state. |

Do not emit arrow labels such as `disk->paged`; they split stats buckets for
the same cache event. `tq-native` is deliberately a separate component because
it proves the persisted bytes used the native packed codec rather than merely
being recompressed after a standard float cache load. Do not encode SSM layer
counts into `cache_detail`; keep layer counts in separate debug fields such as
`_cache_detail_ssm_layers` so `cache_hit_tokens_by_detail` remains a stable
rollup.
