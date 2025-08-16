def cast_value_to_type(value, value_type):
    try:
        if value_type == "int":
            return int(value)
        if value_type == "float":
            return float(value)
        if value_type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        return str(value)
    except Exception:
        return value

def init_param_state(st, val_key: str, typ: str, cfg: dict, ideal):
    if val_key in st.session_state:
        return
    value_type = cfg.get("value_type")
    if typ in ["number", "slider"]:
        if isinstance(ideal, (int, float)) or (isinstance(ideal, str) and ideal.replace('.', '', 1).isdigit()):
            st.session_state[val_key] = cast_value_to_type(ideal, value_type or ("float" if isinstance(ideal, float) else "int"))
        else:
            min_val = cfg.get("min", 0)
            st.session_state[val_key] = cast_value_to_type(min_val, value_type or ("float" if isinstance(min_val, float) else "int"))
    elif typ == "checkbox":
        st.session_state[val_key] = cast_value_to_type(ideal if ideal != "" else False, value_type or "bool")
    else:
        st.session_state[val_key] = cast_value_to_type(ideal if ideal != "" else "", value_type or "str")


