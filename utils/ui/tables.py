import os
import json
import streamlit as st

from .widgets import (
    render_checkbox_widget,
    render_option_widget,
    render_slider_widget,
    render_number_widget,
    render_text_widget,
)
from .display import show_current_and_ideal, show_selected_option_details
from .state import init_param_state


def _load_doc():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")
    doc_path = os.path.join(config_dir, "parameter_documentation.json")
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _render_param_block(name, cfg, param_doc, val_key, title_prefix):
    with st.expander(f"**{cfg.get('label', name)}** - {cfg.get('info', '')}", expanded=False):
        st.markdown(f"**Description:** {cfg.get('info', '')}")

        if param_doc:
            with st.expander("📖 Parameter Documentation", expanded=False):
                from .display import display_parameter_doc_sidebar
                display_parameter_doc_sidebar(param_doc)

        ideal = cfg.get("ideal", "")
        typ = cfg.get("type", "text")
        options = cfg.get("options", [])

        init_param_state(st, val_key, typ, cfg, ideal)

        if typ == "checkbox":
            value = render_checkbox_widget(val_key)
            st.session_state[val_key] = value
            return value
        elif typ in ("dropdown", "select") and options:
            selected_option = render_option_widget(val_key, options)
            st.session_state[val_key] = selected_option
            show_selected_option_details(param_doc, selected_option)
            show_current_and_ideal(val_key, cfg)
            return selected_option
        elif typ == "slider":
            value = render_slider_widget(val_key, cfg)
            st.session_state[val_key] = value
            show_current_and_ideal(val_key, cfg)
            return value
        elif typ == "number":
            value = render_number_widget(val_key, cfg)
            st.session_state[val_key] = value
            show_current_and_ideal(val_key, cfg)
            return value
        else:
            value = render_text_widget(val_key)
            st.session_state[val_key] = value
            show_current_and_ideal(val_key, cfg)
            return value


def create_preprocessing_table(params, task, model_name=None):
    st.subheader("📝 Preprocessing Parameters")
    with st.expander("Configure Preprocessing Parameters", expanded=True):
        doc_data = _load_doc()
        param_values = {}
        for name, cfg in params.items():
            param_doc = doc_data.get("preprocessing_parameters", {}).get(name, {})
            val_key = f"preprocessing_{name}_val"
            value = _render_param_block(name, cfg, param_doc, val_key, "📝")
            param_values[name] = value

        st.markdown("---")
        st.markdown("### Current Configuration")
        st.json(param_values)
        return param_values


def create_encoding_table(params, task, model_name=None):
    st.subheader("🔧 Encoding Parameters")
    
    # Filter parameters based on selected model
    if model_name:
        try:
            from components.encoder import LangchainEncoder
            supported_params = LangchainEncoder.get_supported_parameters(model_name)
            filtered_params = {k: v for k, v in params.items() if k in supported_params}
            
            if len(filtered_params) != len(params):
                unsupported = set(params.keys()) - supported_params
                st.info(f"⚠️ The following parameters are not supported by {model_name} and are hidden: {', '.join(unsupported)}")
                params = filtered_params
        except Exception as e:
            st.warning(f"Could not determine supported parameters for {model_name}: {e}")
    
    with st.expander("Configure Encoding Parameters", expanded=True):
        doc_data = _load_doc()
        param_values = {}
        for name, cfg in params.items():
            param_doc = doc_data.get("encoding_parameters", {}).get(name, {})
            val_key = f"encoding_{name}_val"
            value = _render_param_block(name, cfg, param_doc, val_key, "🔧")
            param_values[name] = value

        st.markdown("---")
        st.markdown("### Current Configuration")
        st.json(param_values)
        return param_values


def create_decoding_table(params, task, model_name=None):
    st.subheader("🎲 Decoding Parameters")
    
    # Filter parameters based on selected model
    if model_name:
        try:
            from components.decoder import LangchainDecoder
            supported_params = LangchainDecoder.get_supported_parameters(model_name)
            filtered_params = {k: v for k, v in params.items() if k in supported_params}
            
            if len(filtered_params) != len(params):
                unsupported = set(params.keys()) - supported_params
                st.info(f"⚠️ The following parameters are not supported by {model_name} and are hidden: {', '.join(unsupported)}")
                params = filtered_params
        except Exception as e:
            st.warning(f"Could not determine supported parameters for {model_name}: {e}")
    
    with st.expander("Configure Decoding Parameters", expanded=True):
        doc_data = _load_doc()
        param_values = {}
        for name, cfg in params.items():
            param_doc = doc_data.get("decoding_parameters", {}).get(name, {})
            val_key = f"decoding_{name}_val"
            value = _render_param_block(name, cfg, param_doc, val_key, "🎲")
            param_values[name] = value

        st.markdown("---")
        st.markdown("### Current Configuration")
        st.json(param_values)
        return param_values


