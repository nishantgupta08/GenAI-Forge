#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face crawler:
- Collect base (non-quantized) models with downloads >= threshold (default 100_000)
- Extract name, type, family (from config.model_type), parameter count
- Discover quantized forks (gguf/gptq/awq/exl2 + bitsandbytes hints)
- For each quant variant, record downloads; also compute min downloads across quant forks
- Optionally download repos to disk

Usage examples:
  python hf_crawler.py --out models.json
  python hf_crawler.py --out models.json --min-downloads 250_000 --quant-min-downloads 10_000
  python hf_crawler.py --out models.json --download base,quant --dst ./hf_models

Requirements:
  pip install "huggingface_hub>=0.16.4"
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils._errors import HfHubHTTPError

# Detect common quantization suffixes
QUANT_PATTERNS = {
    "gguf": r"(?:^|[-_/])gguf(?:$|[-_/])|\.gguf$",
    "gptq": r"(?:^|[-_/])gptq(?:$|[-_/])",
    "awq":  r"(?:^|[-_/])awq(?:$|[-_/])",
    "exl2": r"(?:^|[-_/])exl2(?:$|[-_/])",
}

# Architecture family mapping from config.model_type
FAMILY_FROM_MODEL_TYPE = {
    # decoders
    "llama": "llama", "mistral": "mistral", "mixtral": "mistral",
    "qwen2": "qwen", "qwen": "qwen", "gemma": "gemma",
    "phi3": "phi", "phi": "phi", "gpt2": "gpt2",
    "gpt_neox": "neox", "falcon": "falcon",
    # encoder-decoders
    "t5": "t5", "mt5": "t5", "flan-t5": "t5",
    "bart": "bart", "led": "led", "mbart": "bart",
    # encoders
    "mpnet": "mpnet", "minilm": "minilm", "e5": "e5",
    "bge": "bge", "gte": "gte", "roberta": "roberta", "bert": "bert",
}

# ---------------------- helpers ----------------------

def is_quant_repo_id(repo_id: str) -> bool:
    low = repo_id.lower()
    return any(re.search(pat, low) for pat in QUANT_PATTERNS.values())

def detect_quant_formats(repo_id: str, tags: List[str]) -> List[str]:
    out = set()
    low = repo_id.lower()
    for fmt, pat in QUANT_PATTERNS.items():
        if re.search(pat, low):
            out.add(fmt)
    tset = " ".join([str(t) for t in (tags or [])]).lower()
    for fmt, pat in QUANT_PATTERNS.items():
        if re.search(pat, tset):
            out.add(fmt)
    return sorted(out)

def arch_type_from_tags(tags: List[str]) -> str:
    t = set([str(x).lower() for x in (tags or [])])
    if {"text-embedding", "feature-extraction"} & t:
        return "encoder"
    if {"seq2seq", "text2text-generation"} & t:
        return "encoder-decoder"
    return "decoder"

def infer_family_from_model_type(model_type: Optional[str]) -> Optional[str]:
    if not model_type:
        return None
    mt = model_type.lower()
    return FAMILY_FROM_MODEL_TYPE.get(mt, mt)

def safe_get_downloads(info) -> int:
    # ModelInfo has downloads / downloadsAllTime depending on hub version
    val = getattr(info, "downloads", None)
    if val is None:
        val = getattr(info, "downloadsAllTime", None)
    try:
        return int(val) if val is not None else 0
    except Exception:
        return 0

def load_config_json(repo_id: str) -> Optional[dict]:
    try:
        p = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_num_parameters(repo_id: str, api: HfApi) -> Optional[int]:
    # Try config.json keys
    cfg = load_config_json(repo_id)
    if cfg:
        for k in ("n_parameters", "num_parameters", "model_size", "params"):
            if k in cfg:
                try:
                    return int(cfg[k])
                except Exception:
                    pass
    # Fallback: model card metadata (cardData)
    try:
        mi = api.model_info(repo_id)
        card = getattr(mi, "cardData", None) or {}
        for k in ("n_parameters", "num_parameters", "model_size", "params"):
            if k in card:
                try:
                    return int(card[k])
                except Exception:
                    continue
    except Exception:
        pass
    return None

def parse_human_num(s: str) -> int:
    """Accept 100_000, '100k', '2M', etc."""
    s = s.strip().lower().replace("_", "")
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)

# ---------------------- core crawler ----------------------

def find_base_models(api: HfApi, min_downloads: int, limit_per_query: int) -> List[str]:
    """
    Enumerate popular base models by several broad queries. Post-filter by downloads.
    We deliberately skip quant repos (gguf/gptq/awq/exl2) at this stage.
    """
    queries = [
        "instruct", "llama", "mistral", "mixtral", "qwen", "gemma", "phi",
        "bart", "t5", "flan t5", "mbart", "led",
        "e5", "bge", "gte", "mpnet", "minilm", "roberta", "bert",
        "embedding", "text-embedding"
    ]
    seen = set()
    bases = []
    for q in queries:
        try:
            models = api.list_models(search=q, limit=limit_per_query)
        except TypeError:
            # older hub versions may not accept search param
            models = api.list_models(limit=limit_per_query)
            models = [m for m in models if q.lower() in m.modelId.lower()]
        for m in models:
            rid = m.modelId
            if rid in seen:
                continue
            seen.add(rid)
            if is_quant_repo_id(rid):
                continue
            dl = safe_get_downloads(m)
            if dl >= min_downloads:
                bases.append(rid)
    return sorted(bases)

def discover_quant_forks(api: HfApi, base_repo_id: str, limit: int = 60) -> List[Tuple[str, int, List[str]]]:
    """
    For a base model, search for common quant forks by name. Return list of:
    (repo_id, downloads, [formats])
    """
    base_name = base_repo_id.split("/")[-1]
    results = {}
    for fmt in QUANT_PATTERNS.keys():
        q = f"{base_name} {fmt}"
        try:
            models = api.list_models(search=q, limit=limit)
        except TypeError:
            models = api.list_models(limit=200)
            models = [m for m in models if base_name.lower() in m.modelId.lower() and fmt in m.modelId.lower()]
        for m in models:
            rid = m.modelId
            # Avoid counting the base itself; keep only “forkish” repos
            if rid == base_repo_id:
                continue
            dl = safe_get_downloads(m)
            fmts = detect_quant_formats(rid, getattr(m, "tags", None) or [])
            if not fmts:
                continue
            prev = results.get(rid)
            if prev:
                # Merge formats; keep max downloads we’ve seen for that repo id
                prev_fmts = set(prev[2]) | set(fmts)
                results[rid] = (rid, max(prev[1], dl), sorted(prev_fmts))
            else:
                results[rid] = (rid, dl, fmts)
    return list(results.values())

def crawl(
    min_downloads: int = 100_000,
    quant_min_downloads: int = 10_000,
    limit_per_query: int = 200,
    sleep_s: float = 0.2,
    download: str = "none",
    dst: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crawl HF Hub for base models with downloads >= min_downloads.
    Discover quantized forks and compute min quant downloads.
    Optionally download base and/or quant repos.

    download: "none" | "base" | "quant" | "base,quant"
    """
    api = HfApi()
    base_ids = find_base_models(api, min_downloads, limit_per_query)

    out_models = []
    for i, rid in enumerate(base_ids, 1):
        try:
            info = api.model_info(rid)
        except HfHubHTTPError:
            continue
        except Exception:
            continue

        tags = getattr(info, "tags", None) or []
        entry: Dict[str, Any] = {
            "name": rid,
            "type": arch_type_from_tags(tags),
            "family": None,
            "source": rid.split("/")[0] if "/" in rid else "unknown",
            "params": None,
            "downloads": safe_get_downloads(info),
            "quantizations": [],
            "quant_min_downloads": None,
        }

        # family + params for base
        cfg = load_config_json(rid)
        if cfg:
            entry["family"] = infer_family_from_model_type(cfg.get("model_type"))
        entry["params"] = get_num_parameters(rid, api)

        # quantized forks with their downloads
        qforks = discover_quant_forks(api, rid, limit=80)
        # filter by quant_min_downloads threshold
        qforks = [q for q in qforks if q[1] >= quant_min_downloads]
        if qforks:
            entry["quant_min_downloads"] = int(min(dl for _, dl, _ in qforks))
            entry["quantizations"] = [
                {"repo": qid, "downloads": int(dl), "formats": fmts, "params": entry["params"]}
                for (qid, dl, fmts) in qforks
            ]
            # add BitsAndBytes hints for encoders/enc-decoders (no separate repo)
            if entry["type"] in {"encoder", "encoder-decoder"}:
                entry["quantizations"].extend([
                    {"format": "bitsandbytes", "bits": 8, "method": "int8", "params": entry["params"]},
                    {"format": "bitsandbytes", "bits": 4, "method": "nf4", "params": entry["params"]},
                ])

        out_models.append(entry)

        # polite pacing (avoid rate limits)
        if sleep_s:
            time.sleep(sleep_s)

        # Optional download
        dl_modes = {m.strip() for m in (download or "none").split(",")}
        try:
            if dst and ("base" in dl_modes or "quant" in dl_modes):
                if "base" in dl_modes:
                    snapshot_download(repo_id=rid, local_dir=dst, repo_type="model", ignore_patterns=["*.md"])
                if "quant" in dl_modes:
                    for qid, _, _ in qforks:
                        snapshot_download(repo_id=qid, local_dir=dst, repo_type="model", ignore_patterns=["*.md"])
        except Exception:
            # ignore download errors; keep metadata
            pass

    return {"generated_at": datetime.now(timezone.utc).isoformat(), "count": len(out_models), "models": out_models}

# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models_popular.json", help="Output JSON path")
    p.add_argument("--min-downloads", default="100_000", help="Base model minimum downloads (int, '50k', '2M', etc.)")
    p.add_argument("--quant-min-downloads", default="0", help="Quantized repo minimum downloads (int, '10k', etc.)")
    p.add_argument("--limit", type=int, default=200, help="Per-query search limit (HF API)")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between model requests (seconds)")
    p.add_argument("--download", default="none", help="What to download: none|base|quant|base,quant")
    p.add_argument("--dst", default=None, help="Directory to download repos into (used if --download != none)")
    args = p.parse_args()

    min_dl = parse_human_num(args.min_downloads)
    qmin_dl = parse_human_num(args.quant_min_downloads)

    data = crawl(
        min_downloads=min_dl,
        quant_min_downloads=qmin_dl,
        limit_per_query=args.limit,
        sleep_s=args.sleep,
        download=args.download,
        dst=args.dst,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} with {data['count']} models.")

if __name__ == "__main__":
    main()
