#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a curated models catalog (famous models + quantized variants)
and keep ONLY the architecture `family` (llama, mistral, qwen, t5, bart, phi, gemma, etc.).

Requirements:
  pip install "huggingface_hub>=0.14.0"

Usage:
  python generate_models_config_family_only.py --out models_config_family_only.json
"""

import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from huggingface_hub import HfApi, hf_hub_download

# -------- Settings --------

MIN_DOWNLOADS = 50_000
RECENCY_DAYS = 540  # ~18 months

CURATED_ALLOWLIST = {
    # Encoders (embeddings)
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-v2",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "Snowflake/snowflake-arctic-embed-l",
    # Encoder-decoder
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/bart-large-cnn",
    "allenai/led-base-16384",
    # Decoder/chat
    "meta-llama/Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
}

SEARCH_QUERIES = [
    # Encoders
    "sentence-transformers",
    "e5 large v2",
    "bge large en",
    "gte large en",
    "nomic embed text",
    "snowflake arctic embed",
    # Encoder-decoder
    "flan t5 base",
    "flan t5 large",
    "bart large cnn",
    "led 16384",
    # Decoder/chat
    "Llama-3 Instruct 8B",
    "Mistral-7B Instruct",
    "Mixtral-8x7B Instruct",
    "Qwen2-7B Instruct",
    "Phi-3 mini 4k instruct",
]

QUANT_PATTERNS = {
    "gguf": r"(?:^|[-_/])gguf(?:$|[-_/])|\.gguf$",
    "gptq": r"(?:^|[-_/])gptq(?:$|[-_/])",
    "awq":  r"(?:^|[-_/])awq(?:$|[-_/])",
    "exl2": r"(?:^|[-_/])exl2(?:$|[-_/])",
}

DEFAULT_BNB_HINTS = [
    {"format":"bitsandbytes","bits":8,"method":"int8"},
    {"format":"bitsandbytes","bits":4,"method":"nf4"}
]

FAMILY_FROM_MODEL_TYPE = {
    # decoder
    "llama": "llama", "mistral": "mistral", "mixtral": "mistral",
    "qwen2": "qwen", "qwen": "qwen",
    "gemma": "gemma", "phi3": "phi", "phi": "phi",
    "gpt2": "gpt2", "gpt_neox": "neox", "falcon": "falcon",
    # encoder-decoder
    "t5": "t5", "mt5": "t5", "flan-t5": "t5",
    "bart": "bart", "led": "led", "mbart": "bart",
    # encoders
    "mpnet": "mpnet", "minilm": "minilm", "e5": "e5", "bge": "bge", "gte": "gte",
    "roberta": "roberta", "bert": "bert"
}

# -------- Helpers --------

def arch_type_from_tags(tags: List[str]) -> str:
    t = set([str(x).lower() for x in (tags or [])])
    if {"text-embedding", "feature-extraction"} & t:
        return "encoder"
    if {"seq2seq", "text2text-generation"} & t:
        return "encoder-decoder"
    return "decoder"

def task_types_from_tags(tags: List[str]) -> List[str]:
    known = {
        "text-generation","text2text-generation","text-embedding",
        "question-answering","summarization","chat","feature-extraction",
        "fill-mask","translation"
    }
    out = [x for x in (tags or []) if x in known]
    return sorted(out)

def parse_last_modified(card) -> Optional[datetime]:
    iso = getattr(card, "lastModified", None) or getattr(card, "lastModifiedAt", None)
    if not iso:
        return None
    try:
        if isinstance(iso, str):
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        from datetime import datetime as dt
        if isinstance(iso, dt):
            return iso if iso.tzinfo else iso.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def get_downloads(card) -> int:
    val = getattr(card, "downloads", None)
    if val is None:
        val = getattr(card, "downloadsAllTime", None)
    try:
        return int(val) if val is not None else 0
    except Exception:
        return 0

def is_famous(card) -> bool:
    dl = get_downloads(card)
    lm = parse_last_modified(card)
    recency_ok = False
    if lm:
        from datetime import datetime as dt
        try:
            recency_ok = (datetime.now(timezone.utc) - lm) <= timedelta(days=RECENCY_DAYS)
        except Exception:
            recency_ok = False
    allowlisted = card.modelId in CURATED_ALLOWLIST
    return allowlisted or dl >= MIN_DOWNLOADS or recency_ok

def detect_quant_formats(name: str, tags: List[str]) -> List[dict]:
    q = []
    lower = name.lower()
    for fmt, pattern in QUANT_PATTERNS.items():
        if re.search(pattern, lower):
            q.append({"format": fmt, "repo": name})
    tset = " ".join([str(t) for t in (tags or [])]).lower()
    for fmt, pattern in QUANT_PATTERNS.items():
        if re.search(pattern, tset) and not any(d["format"] == fmt for d in q):
            q.append({"format": fmt, "repo": name})
    return q

def merge_quantizations(base_q: List[dict], new_q: List[dict]) -> List[dict]:
    out = list(base_q or [])
    seen = {(d.get("format"), d.get("repo")) for d in out}
    for d in new_q or []:
        key = (d.get("format"), d.get("repo"))
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out

def add_bnb_hints_if_applicable(entry: dict):
    if entry.get("type") in {"encoder", "encoder-decoder"}:
        q = entry.setdefault("quantizations", [])
        have = {d.get("format") for d in q}
        if "bitsandbytes" not in have:
            entry["quantizations"] = q + [{"format":"bitsandbytes","bits":8,"method":"int8"},{"format":"bitsandbytes","bits":4,"method":"nf4"}]

def load_config_dict(repo_id: str) -> Optional[dict]:
    try:
        p = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=None)
    except Exception:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def infer_family_from_model_type(model_type: Optional[str]) -> Optional[str]:
    if not model_type:
        return None
    mt = model_type.lower()
    return FAMILY_FROM_MODEL_TYPE.get(mt, mt)

# -------- Build --------

def build_catalog(limit_per_query: int = 100) -> dict:
    api = HfApi()
    base_models: Dict[str, dict] = {}

    # 1) discover candidate base models
    for q in SEARCH_QUERIES:
        try:
            cards = api.list_models(search=q, limit=limit_per_query)
        except TypeError:
            cards = api.list_models(limit=limit_per_query)
            cards = [c for c in cards if q.lower() in c.modelId.lower()]
        for card in cards:
            model_id = card.modelId
            if any(re.search(pat, model_id.lower()) for pat in QUANT_PATTERNS.values()):
                continue
            if not is_famous(card):
                continue

            tags = getattr(card, "tags", None) or []
            card_data = getattr(card, "cardData", None) or {}

            entry = base_models.get(model_id) or {
                "name": model_id,
                "type": arch_type_from_tags(tags),
                "task_type": [],  # keep lean
                "family": None,
                "source": (model_id.split("/")[0] if "/" in model_id else "unknown"),
                "license": card_data.get("license") if isinstance(card_data, dict) else None,
                "quantizations": []
            }
            base_models[model_id] = entry

    # 2) attach quantized forks and set family from config.json
    for model_id, entry in list(base_models.items()):
        basename = model_id.split("/")[-1]
        # quant forks
        for fmt in QUANT_PATTERNS.keys():
            q = f"{basename} {fmt}"
            try:
                q_cards = api.list_models(search=q, limit=30)
            except TypeError:
                q_cards = api.list_models(limit=200)
                q_cards = [c for c in q_cards if basename.lower() in c.modelId.lower() and fmt in c.modelId.lower()]
            for qc in q_cards:
                quant = detect_quant_formats(qc.modelId, getattr(qc, "tags", None) or [])
                if quant:
                    entry["quantizations"] = merge_quantizations(entry.get("quantizations", []), quant)

        # bitsandbytes hint
        add_bnb_hints_if_applicable(entry)

        # family via config.json model_type
        cfg = load_config_dict(model_id)
        if cfg:
            entry["family"] = infer_family_from_model_type(cfg.get("model_type"))

    # 3) ensure curated allowlist entries exist with family if possible
    for name in CURATED_ALLOWLIST:
        if name not in base_models:
            base_models[name] = {
                "name": name,
                "type": "decoder",
                "task_type": [],
                "family": None,
                "source": (name.split("/")[0] if "/" in name else "unknown"),
                "license": None,
                "quantizations": []
            }
            add_bnb_hints_if_applicable(base_models[name])
            cfg = load_config_dict(name)
            if cfg:
                base_models[name]["family"] = infer_family_from_model_type(cfg.get("model_type"))

    catalog = sorted(base_models.values(), key=lambda d: d["name"].lower())
    return {"models": catalog}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models_config_family_only.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=100, help="Per-query search limit")
    args = parser.parse_args()

    out = build_catalog(limit_per_query=args.limit)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} with {len(out['models'])} models.")

if __name__ == "__main__":
    main()
