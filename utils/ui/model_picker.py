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
        # Count GGUF entries
        ggufs = sum(1 for x in q if isinstance(x, dict) and x.get("format") == "gguf")
        if ggufs:
            out.append(f"gguf×{ggufs}")

        # Bitsandbytes entries
        for x in q:
            if isinstance(x, dict) and x.get("format") == "bitsandbytes":
                bits = x.get("bits")
                method = (x.get("method") or "").upper()
                if bits:
                    out.append(f"{bits}-bit {method}" if method else f"{bits}-bit")

        # De-duplicate while preserving order
        seen, cleaned = set(), []
        for item in out:
            if item not in seen:
                cleaned.append(item)
                seen.add(item)
        return ", ".join(cleaned)
    except Exception:
        return ""


def _preprocess_models_df(models_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for display: add formatted fields, keep raw for selection."""
    df = models_df.copy()
    # Ensure expected columns exist
    for col in ["name", "type", "family", "source", "license", "quantizations", "task_type"]:
        if col not in df.columns:
            df[col] = ""

    # Human-friendly display columns
    df["quantizations_display"] = df["quantizations"].apply(_format_quantizations)
    df["task_type_display"] = df["task_type"].apply(
        lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else (v or "")
    )

    # Keep a JSON-encoded raw column (hidden) so selection preserves metadata
    def _safe_json(v):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return ""

    df["quantizations_raw"] = df["quantizations"].apply(_safe_json)

    display_cols = [
        "name", "type", "family", "source", "license",
        "task_type_display", "quantizations_display", "quantizations_raw"
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing]


def aggrid_model_picker(models_df, key="aggrid_model_picker"):
    """
    Show models with search, filters, pagination, and clean columns.
    Returns the selected row as dict (includes hidden raw fields) or None.
    """
    df = _preprocess_models_df(models_df)

    # ---- Filters above the grid
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2])
    with c1:
        q = st.text_input("Search", placeholder="name, family, source…", key=f"{key}_search")
    with c2:
        types = sorted([x for x in df["type"].dropna().unique() if x != ""])
        sel_types = st.multiselect("Type", types, default=[], key=f"{key}_type")
    with c3:
        families = sorted([x for x in df["family"].dropna().unique() if x != ""])
        sel_fams = st.multiselect("Family", families, default=[], key=f"{key}_family")
    with c4:
        sources = sorted([x for x in df["source"].dropna().unique() if x != ""])
        sel_srcs = st.multiselect("Source", sources, default=[], key=f"{key}_source")

    # Apply filters
    fdf = df
    if q:
        ql = q.lower()
        mask = (
            fdf["name"].astype(str).str.lower().str.contains(ql)
            | fdf["family"].astype(str).str.lower().str.contains(ql)
            | fdf["source"].astype(str).str.lower().str.contains(ql)
            | fdf["task_type_display"].astype(str).str.lower().str.contains(ql)
        )
        fdf = fdf[mask]
    if sel_types:
        fdf = fdf[fdf["type"].isin(sel_types)]
    if sel_fams:
        fdf = fdf[fdf["family"].isin(sel_fams)]
    if sel_srcs:
        fdf = fdf[fdf["source"].isin(sel_srcs)]

    # ---- AgGrid config
    gb = GridOptionsBuilder.from_dataframe(fdf)
    gb.configure_default_column(
        resizable=True, filter=True, sortable=True, floatingFilter=True,
        wrapText=True, autoHeight=True,
    )
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_side_bar()  # show filter/sort panel
    gb.configure_column("name", pinned="left", width=320)
    gb.configure_column("task_type_display", header_name="task_type")
    gb.configure_column("quantizations_display", header_name="quantizations", tooltipField="quantizations_display")
    gb.configure_column("quantizations_raw", header_name="quantizations_raw", hide=True)  # keep raw, hide in UI

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
