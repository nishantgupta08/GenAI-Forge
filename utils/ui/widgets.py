import streamlit as st

def ensure_slider_types(min_val, max_val, step, current_val):
    if isinstance(step, float) or any(isinstance(x, float) for x in [min_val, max_val, current_val]):
        return float(min_val), float(max_val), float(step), float(current_val)
    return int(min_val), int(max_val), int(step), int(current_val)

def get_numeric_bounds(cfg: dict):
    return cfg.get("min", 0), cfg.get("max", 100), cfg.get("step", 1)

def render_checkbox_widget(val_key: str):
    return st.checkbox("Enable", value=st.session_state[val_key], key=f"widget_{val_key}")

def render_option_widget(val_key: str, options: list):
    return st.radio(
        "Choose an option:",
        options,
        index=options.index(st.session_state[val_key]) if st.session_state[val_key] in options else 0,
        key=f"radio_{val_key}"
    )

def render_slider_widget(val_key: str, cfg: dict):
    min_val, max_val, step = get_numeric_bounds(cfg)
    min_val, max_val, step, current_val = ensure_slider_types(min_val, max_val, step, st.session_state[val_key])
    return st.slider("Adjust value:", min_value=min_val, max_value=max_val, value=current_val, step=step, key=f"widget_{val_key}")

def render_number_widget(val_key: str, cfg: dict):
    min_val, max_val, step = get_numeric_bounds(cfg)
    min_val, max_val, step, current_val = ensure_slider_types(min_val, max_val, step, st.session_state[val_key])
    return st.number_input("Enter value:", min_value=min_val, max_value=max_val, value=current_val, step=step, key=f"widget_{val_key}")

def render_text_widget(val_key: str):
    return st.text_input("Enter value:", value=st.session_state[val_key], key=f"widget_{val_key}")


