#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face crawler:
- Collect base (non-quantized) models with downloads >= threshold (default 100_000)
- Extract ONLY: name, type (encoder/decoder/encoder-decoder), family (from config.model_type),
  params (num parameters), downloads (base)
- Discover quantized forks (gguf/gptq/awq/exl2) and keep ONLY those with downloads >= quant_min_downloads (default 5_000)
- Output includes downloads for base and for each quantized repo

Usage:
  python hf_crawler_min.py --out models.json
  python hf_crawler_min.py --out models.json --min-downloads 250k --quant-min-downloads 10k

Requirements:
  pip install "huggingface_hub>=0.16.4"
"""

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download

# Back/forward compatible HTTP error (optional)
try:
    from huggingface_hub.errors import HfHubHTTPError  # newer
except Exception:  # pragma: no cover
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError  # older
    except Exception:
        class HfHubHTTPError(Exception):
            pass

# ---------------------- config ----------------------

# Quantization format detectors
QUANT_PATTERNS = {
    "gguf": r"(?:^|[-_/])gguf(?:$|[-_/])|\.gguf$",
    "gptq": r"(?:^|[-_/])gptq(?:$|[-_/])",
    "awq":  r"(?:^|[-_/])awq(?:$|[-_/])",
    "exl2": r"(?:^|[-_/])exl2(?:$|[-_/])",
}

# Map config.model_type -> simplified family
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

def parse_human_num(s: str) -> int:
    """Accept 100_000, '100k', '2M', etc."""
    s = s.strip().lower().replace("_", "")
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)

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
    mt = str(model_type).lower()
    return FAMILY_FROM_MODEL_TYPE.get(mt, mt)

def safe_get_downloads(info) -> int:
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

# --- Robust parameter parsing (handles 'Model size', 'num_parameters', '7B', etc.) ---

_PARAM_KEY_ALIASES = {
    # normalized forms (lowercase, non-alnum stripped)
    "nparameters",
    "numparameters",
    "parameters",
    "params",
    "modelsize",        # handles "Model size" / "model_size" / "model-size"
    "parametercount",
    "paramcount",
    "nparams",
    "numparams",
}

def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())

def _parse_params_value(v) -> Optional[int]:
    """
    Accept ints/floats or strings like '7B', '7.1B', '500M', '7B parameters'.
    Returns absolute parameter count as int.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return None
    s = str(v).strip().lower().replace(",", "").replace("_", "")
    m = re.search(r"([\d\.]+)\s*([kmb])?", s)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    mult = 1
    if unit == "k":
        mult = 1_000
    elif unit == "m":
        mult = 1_000_000
    elif unit == "b":
        mult = 1_000_000_000
    try:
        return int(num * mult)
    except Exception:
        return None

def _pick_params_from_mapping(d: dict) -> Optional[int]:
    for k, v in (d or {}).items():
        if _norm_key(k) in _PARAM_KEY_ALIASES:
            n = _parse_params_value(v)
            if n is not None:
                return n
    return None

def get_num_parameters(repo_id: str, api: HfApi) -> Optional[int]:
    # 1) Try config.json
    cfg = load_config_json(repo_id)
    n = _pick_params_from_mapping(cfg) if cfg else None
    if n is not None:
        return n
    # 2) Fallback: model card metadata (cardData)
    try:
        mi = api.model_info(repo_id)
        card = getattr(mi, "cardData", None) or {}
        n = _pick_params_from_mapping(card)
        if n is not None:
            return n
    except Exception:
        pass
    return None

# ---------------------- core crawler ----------------------

def find_base_models(api: HfApi, min_downloads: int, limit_per_query: int) -> List[str]:
    """
    Enumerate popular base models (skip repos that look quantized),
    and keep those with downloads >= min_downloads.
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

def discover_quant_forks(api: HfApi, base_repo_id: str, limit: int = 80) -> List[Tuple[str, int, List[str]]]:
    """
    For a base model, search for common quant forks by name. Return list of:
    (repo_id, downloads, [formats])
    """
    base_name = base_repo_id.split("/")[-1]
    results: Dict[str, Tuple[str, int, List[str]]] = {}
    for fmt in QUANT_PATTERNS.keys():
        q = f"{base_name} {fmt}"
        try:
            models = api.list_models(search=q, limit=limit)
        except TypeError:
            models = api.list_models(limit=200)
            models = [m for m in models if base_name.lower() in m.modelId.lower() and fmt in m.modelId.lower()]
        for m in models:
            rid = m.modelId
            if rid == base_repo_id:
                continue
            dl = safe_get_downloads(m)
            fmts = detect_quant_formats(rid, getattr(m, "tags", None) or [])
            if not fmts:
                continue
            prev = results.get(rid)
            if prev:
                prev_fmts = sorted(set(prev[2]) | set(fmts))
                results[rid] = (rid, max(prev[1], dl), prev_fmts)
            else:
                results[rid] = (rid, dl, fmts)
    return list(results.values())

def crawl_models(
    min_downloads: int = 100_000,
    quant_min_downloads: int = 10_000,
    limit_per_query: int = 200,
) -> Dict[str, Any]:
    """
    Crawl HF for base models and qualifying quant forks.

    Returns JSON with:
      name, type, family, source, params, downloads, quant_min_downloads, quantizations[]
      where each quantization has: repo, formats, downloads
    """
    api = HfApi()
    base_ids = find_base_models(api, min_downloads, limit_per_query)
    out_models = []

    for rid in base_ids:
        try:
            info = api.model_info(rid)
        except HfHubHTTPError:
            continue
        except Exception:
            continue

        tags = getattr(info, "tags", None) or []
        base_downloads = safe_get_downloads(info)

        entry: Dict[str, Any] = {
            "name": rid,
            "type": arch_type_from_tags(tags),
            "family": None,
            "source": rid.split("/")[0] if "/" in rid else "unknown",
            "params": None,
            "downloads": base_downloads,
            "quant_min_downloads": None,
            "quantizations": [],
        }

        # family + params (base only)
        cfg = load_config_json(rid)
        if cfg:
            entry["family"] = infer_family_from_model_type(cfg.get("model_type"))
        entry["params"] = get_num_parameters(rid, api)

        # discover quant forks; keep only those meeting quant_min_downloads
        qforks = discover_quant_forks(api, rid, limit=80)
        qforks = [q for q in qforks if q[1] >= quant_min_downloads]
        if qforks:
            entry["quant_min_downloads"] = int(min(dl for _, dl, _ in qforks))
            entry["quantizations"] = [
                {"repo": qid, "formats": fmts, "downloads": int(dl)}
                for (qid, dl, fmts) in qforks
            ]

        out_models.append(entry)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(out_models),
        "models": out_models,
    }

# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models.json", help="Output JSON path")
    p.add_argument("--min-downloads", default="100_000",
                   help="Base model minimum downloads (int, '50k', '2M', etc.)")
    p.add_argument("--quant-min-downloads", default="5_000",
                   help="Quantized repo minimum downloads (int, '5k', etc.)")
    p.add_argument("--limit", type=int, default=200, help="Per-query search limit (HF API)")
    args = p.parse_args()

    min_dl = parse_human_num(args.min_downloads)
    qmin_dl = parse_human_num(args.quant_min_downloads)

    data = crawl_models(
        min_downloads=min_dl,
        quant_min_downloads=qmin_dl,
        limit_per_query=args.limit,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} with {data['count']} models.")

if __name__ == "__main__":
    main()
