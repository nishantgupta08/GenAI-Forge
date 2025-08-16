
import json
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


BASE_LABEL = "Base (no quantization)"


def _format_quantizations(q):
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


def _labels_and_map(quant_list):
    labels = [BASE_LABEL]
    mapping = {BASE_LABEL: None}
    for q in (quant_list or []):
        if isinstance(q, dict):
            repo = q.get("repo") or ""
            fmts = q.get("formats") or []
            label = repo if repo else (", ".join(fmts) if fmts else "Quantized")
            # ensure uniqueness if duplicate labels
            orig = label
            i = 2
            while label in mapping:
                label = f"{orig} ({i})"
                i += 1
            labels.append(label)
            mapping[label] = q
    return labels, mapping


def _preprocess_models_df(models_df: pd.DataFrame) -> pd.DataFrame:
    df = models_df.copy()
    for col in ["name", "family", "source", "params", "quantizations"]:
        if col not in df.columns:
            df[col] = ""
    # Summary text
    df["quantizations_display"] = df["quantizations"].apply(_format_quantizations)
    # Raw + editor options
    df["quantizations_raw"] = df["quantizations"].apply(_safe_json)

    # Build per-row options + mapping + default choice
    labels_series = []
    mapping_series = []
    for _, row in df.iterrows():
        qlist = row.get("quantizations") or []
        labels, mapping = _labels_and_map(qlist if isinstance(qlist, list) else [])
        labels_series.append(labels)
        mapping_series.append(_safe_json(mapping))
    df["quant_options"] = labels_series              # list[str] per row (used by JS)
    df["quant_map"] = mapping_series                 # json map label -> entry (hidden)
    df["quant_choice"] = BASE_LABEL                  # editable value shown to user

    display_cols = ["name", "source", "family", "params", "quant_choice", "quantizations_display", "quant_options", "quant_map"]
    return df[display_cols]


def aggrid_model_picker_with_row_dropdown(models_df, key="aggrid_model_picker_row"):
    df = _preprocess_models_df(models_df)

    # Controls: Search + Family
    c1, c2 = st.columns([1.3, 1])
    with c1:
        q = st.text_input("Search", placeholder="name, family, source…", key=f"{key}_search")
    with c2:
        families = ["(All)"] + sorted([x for x in df["family"].dropna().unique() if x != ""])
        sel_family = st.selectbox("Family", options=families, index=0, key=f"{key}_family")

    fdf = df.copy()
    if q:
        ql = q.lower()
        fdf = fdf[fdf[["name", "family", "source"]].apply(lambda r: r.astype(str).str.lower().str.contains(ql).any(), axis=1)]
    if sel_family and sel_family != "(All)":
        fdf = fdf[fdf["family"] == sel_family]

    # JS function: per-row select options from quant_options
    editor_params_fn = JsCode(
        """function(params) {
              return { values: params.data.quant_options || ["Base (no quantization)"] };
            }"""
    )

    gb = GridOptionsBuilder.from_dataframe(fdf)
    gb.configure_default_column(resizable=True, filter=True, sortable=True, floatingFilter=True, wrapText=True, autoHeight=True)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_side_bar()
    gb.configure_column("name", pinned="left", width=320)
    gb.configure_column("quant_choice", header_name="quantization", editable=True, cellEditor="agSelectCellEditor", cellEditorParams=editor_params_fn)
    gb.configure_column("quantizations_display", header_name="quantizations", tooltipField="quantizations_display")
    gb.configure_column("quant_options", hide=True)
    gb.configure_column("quant_map", hide=True)

    grid_return = AgGrid(
        fdf,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,  # capture edits
        allow_unsafe_jscode=True,
        key=key,
        height=520,
        fit_columns_on_grid_load=True,
        # data_return_mode = 'AS_INPUT' is default in newer versions; keep to ensure edits come back
    )

    # Selected row (with current quant_choice)
    sel = grid_return.get("selected_rows")
    if isinstance(sel, pd.DataFrame):
        sel_row = sel.iloc[0].to_dict() if not sel.empty else None
    elif isinstance(sel, list) and sel:
        sel_row = sel[0]
    else:
        sel_row = None

    if not sel_row:
        return None

    # Resolve selected label -> mapping
    try:
        mapping = json.loads(sel_row.get("quant_map", "{}"))
    except Exception:
        mapping = {}
    label = sel_row.get("quant_choice") or BASE_LABEL
    entry = mapping.get(label)

    return {
        "name": sel_row.get("name"),
        "source": sel_row.get("source"),
        "family": sel_row.get("family"),
        "params": sel_row.get("params"),
        "selection": {
            "variant": "quantized" if entry else "base",
            "quantization_repo": (entry or {}).get("repo") if isinstance(entry, dict) else None,
            "formats": (entry or {}).get("formats") if isinstance(entry, dict) else None,
            "label": label,
        },
    }
