#!/usr/bin/env python3
import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path("/Users/eric/mlx/vllm-mlx-release-1.6.13")
BUNDLE = Path("/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK")
DB = Path("/Users/eric/.vmlx-v1613-responsive-dev/chats.db")
ROWID = 225
TEMPERATURE = float(os.environ.get("DSV4_AB_TEMP", "0.6"))
TOP_P = float(os.environ.get("DSV4_AB_TOP_P", "0.95"))
MAX_TOKENS = int(os.environ.get("DSV4_AB_MAX_TOKENS", "512"))

sys.path.insert(0, str(REPO))
os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", "115")
os.environ.setdefault("DSV4_LONG_CTX", "1")
os.environ.setdefault("DSV4_POOL_QUANT", "1")
os.environ.setdefault("DSV4_LAYERWISE_PREFILL", "1")
os.environ.setdefault("VMLX_DISABLE_TQ_KV", "1")

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from vmlx_engine.utils.jang_loader import load_jang_model

encoding_path = BUNDLE / "encoding" / "encoding_dsv4.py"
spec = importlib.util.spec_from_file_location("encoding_dsv4_direct_ab", encoding_path)
encoding = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(encoding)

with sqlite3.connect(DB) as db:
    row = db.execute("select content from messages where rowid=?", (ROWID,)).fetchone()
if not row:
    raise SystemExit(f"missing SQLite row {ROWID}")
user_text = row[0]
prompt = encoding.encode_messages(
    [{"role": "user", "content": user_text}],
    thinking_mode="thinking",
)

load_start = time.perf_counter()
model, tokenizer = load_jang_model(BUNDLE)
load_seconds = time.perf_counter() - load_start

prompt_ids = tokenizer.encode(prompt)
generation_start = time.perf_counter()
raw = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=MAX_TOKENS,
    sampler=make_sampler(temp=TEMPERATURE, top_p=TOP_P),
    verbose=False,
)
generation_seconds = time.perf_counter() - generation_start

result = {
    "source": str(REPO),
    "bundle": str(BUNDLE),
    "sqlite_rowid": ROWID,
    "execution": "direct mlx_lm.generate using vmlx loader; no server, scheduler, paged RAM, or block-disk L2",
    "env": {key: os.environ.get(key) for key in (
        "DSV4_LONG_CTX", "DSV4_POOL_QUANT", "DSV4_LAYERWISE_PREFILL", "VMLX_DISABLE_TQ_KV"
    )},
    "prompt_bytes": len(user_text.encode()),
    "prompt_tokens": len(prompt_ids),
    "mode": "thinking",
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "load_seconds": round(load_seconds, 3),
    "generation_seconds": round(generation_seconds, 3),
    "active_memory_mb": round(mx.get_active_memory() / 1_000_000, 2),
    "raw_length": len(raw),
    "has_thinking_end": "</think>" in raw,
    "has_start_secret": "EMBER-4137" in raw,
    "has_end_secret": "VIOLET-8624" in raw,
    "has_done_marker": "DSV4-MEDIUM-UI-AUTO-DONE" in raw,
    "raw_prefix": raw[:2000],
    "raw_suffix": raw[-2000:],
}
print(json.dumps(result, indent=2, ensure_ascii=False))
