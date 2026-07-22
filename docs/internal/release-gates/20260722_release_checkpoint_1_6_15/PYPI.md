# vMLX 1.6.15 PyPI publication proof

Status: `PASS-PUBLIC`.

## Exact source provenance

- Release tag: `v1.6.15`.
- Tag peel / detached build commit:
  `344b6c88e46ce3eaf4aeb32108a48ed7144c7d2f`.
- Clean build worktree:
  `/Users/eric/mlx/vllm-mlx-pypi-v1.6.15`.
- `python -m build --sdist --wheel` and `twine check` completed from that
  detached source. Both public files are byte-identical to those exact-tag
  build outputs.

## Publication path

The official GitHub workflow was attempted first:

- Run: `https://github.com/jjang-ai/vmlx/actions/runs/29907439101`.
- Checkout, release-ref validation, package build, and metadata checks passed.
- Trusted publishing failed only at the identity exchange with
  `invalid-publisher` for `repo:jjang-ai/vmlx:environment:pypi`.

The explicitly authorized authenticated fallback then uploaded the same two
exact-tag build outputs. No tag, source file, or tracked package file was
altered in this lane.

## Public files

Fresh PyPI JSON and independent downloads report:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vmlx-1.6.15-py3-none-any.whl` | 1,720,560 | `cffa81c3b4093394bd70874a9b4623ef3651cbfea0d3442ceecc7bb06be21f0e` |
| `vmlx-1.6.15.tar.gz` | 2,744,159 | `1114d8bd5872a6d2e5b6b1d5fc6b547560b76190339ae160fc5aaa77fae07c4c` |

Both files are not yanked and advertise Python `>=3.11,<3.15`. PyPI upload
times are `2026-07-22T09:20:42.240598Z` for the wheel and
`2026-07-22T09:20:44.961576Z` for the sdist.

## Clean install proof

A fresh virtual environment at
`/Users/eric/.cache/vmlx-release/v1.6.15-clean-install` installed
`vmlx==1.6.15` from the public index with cache disabled and dependencies
suppressed so the package artifact itself was isolated. From `/`, both
`importlib.metadata.version("vmlx")` and `vmlx_engine.__version__` returned
`1.6.15`, and the imported module path was inside that new environment.

The import printed the expected warning that the optional DSV4 JANG runtime
patch was unavailable because this proof intentionally used `--no-deps`; that
does not represent the fully bundled MLXStudio dependency set and is not a
DSV4 runtime proof.

## Evidence files

- `workflow-29907439101.json`
- `workflow-29907439101-failed.log`
- `package-index-exact-tag-build.txt`
- `package-index-upload.log`
- `package-index-proof-sanitization.txt`
- `public-package-index-1.6.15.json`
- `public-package-index-files.txt`
- `public-package-index-metadata.txt`
- `public-package-index-final-verification.txt`
- `public-package-index-clean-install.txt`

The publication logs and metadata were scanned for the PyPI credential prefix;
no secret-like value is retained in these artifacts.
