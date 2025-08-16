
import json
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


def _format_quantizations(q):
    """Return a compact, human-friendly quantization summary string."""
    try:
        if not q:
            return ""
        if isinstance(q, str):
            return q
        out = []
        gguf_cnt = sum(1 for x in q if isinstance(x, dict) and 'formats' in x and 'gguf' in x['formats'])
        if gguf_cnt:
            out.append(f"GGUF x{gguf_cnt}")
        other = []
        for x in q:
            if isinstance(x, dict) and 'formats' in x:
                for f in x['formats']:
                    if f != 'gguf':
                        other.append(f)
        if other:
            other_unique = sorted(set(other))
            out.append(", ".join(other_unique))
        return " | ".join(out)
    except Exception:
        return ""


def _safe_json(v):
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return ""


def _preprocess_models_df(models_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for display with the required columns only."""
    df = models_df.copy()
    for col in ["name", "family", "source", "params", "quantizations"]:
        if col not in df.columns:
            df[col] = ""
    df["quantizations_display"] = df["quantizations"].apply(_format_quantizations)
    df["quantizations_raw"] = df["quantizations"].apply(_safe_json)
    display_cols = ["name", "source", "family", "params", "quantizations_display", "quantizations_raw"]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing]


def aggrid_model_picker(models_df, key="aggrid_model_picker"):
    """Show models with search and a *single* Family dropdown. Return the selected row as dict or None."""
    df = _preprocess_models_df(models_df)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        q = st.text_input("Search", placeholder="name, family, source…", key=f"{key}_search")
    with c2:
        families = ["(All)"] + sorted([x for x in df["family"].dropna().unique() if x != ""])  # dropdown
        sel_family = st.selectbox("Family", options=families, index=0, key=f"{key}_family")
    fdf = df.copy()
    if q:
        ql = q.lower()
        fdf = fdf[fdf[["name", "family", "source"]].apply(lambda r: r.astype(str).str.lower().str.contains(ql).any(), axis=1)]
    if sel_family and sel_family != "(All)":
        fdf = fdf[fdf["family"] == sel_family]
    gb = GridOptionsBuilder.from_dataframe(fdf)
    gb.configure_default_column(resizable=True, filter=True, sortable=True, floatingFilter=True, wrapText=True, autoHeight=True)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_side_bar()
    gb.configure_column("name", pinned="left", width=320)
    gb.configure_column("quantizations_display", header_name="quantizations", tooltipField="quantizations_display")
    gb.configure_column("quantizations_raw", header_name="quantizations_raw", hide=True)
    grid_return = AgGrid(
        fdf,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        key=key,
        height=520,
        fit_columns_on_grid_load=True,
    )
    sel = grid_return.get("selected_rows")
    if isinstance(sel, pd.DataFrame):
        return sel.iloc[0].to_dict() if not sel.empty else None
    if isinstance(sel, list) and sel:
        return sel[0]
    return None


def select_model_with_quantization(models_df, key="model_and_quantization"):
    """Two-step selection: pick a base model row, then choose a quantized variant.
    
    Returns a dict like:
      {
        'name': ...,
        'source': ...,
        'family': ...,
        'params': ...,
        'selection': {
            'variant': 'base' | 'quantized',
            'quantization_repo': <repo or None>,
            'formats': <list or None>,
        }
      }
    or None if nothing selected.
    """
    row = aggrid_model_picker(models_df, key=f"{key}_picker")
    if not row:
        return None

    st.markdown("---")
    st.markdown(f"### Selected base model\n**{row.get('name','')}**  ")  # brief confirmation

    # Build dropdown: Base + quantizations
    try:
        quant_list = json.loads(row.get("quantizations_raw", "[]"))
        # Build label -> entry map
        options = [("Base (no quantization)", None)]
        for q in quant_list:
            if isinstance(q, dict):
                repo = q.get("repo") or ""
                fmts = q.get("formats") or []
                label = repo if repo else (", ".join(fmts) if fmts else "Quantized" )
                options.append((label, q))
    except Exception:
        options = [("Base (no quantization)", None)]

    labels = [lbl for lbl, _ in options]
    sel_label = st.selectbox("Quantized version", options=labels, key=f"{key}_quant_sel")
    sel_entry = dict(options)[sel_label]

    selection = {
        "name": row.get("name"),
        "source": row.get("source"),
        "family": row.get("family"),
        "params": row.get("params"),
        "selection": {
            "variant": "quantized" if sel_entry else "base",
            "quantization_repo": (sel_entry or {}).get("repo") if isinstance(sel_entry, dict) else None,
            "formats": (sel_entry or {}).get("formats") if isinstance(sel_entry, dict) else None,
        },
    }
    return selection
