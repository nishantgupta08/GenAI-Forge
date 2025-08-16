import streamlit as st

def show_current_and_ideal(val_key: str, cfg: dict):
    st.markdown(f"**Current Value:** `{st.session_state[val_key]}`")
    ideal_value = cfg.get("ideal", "Not specified")
    ideal_reason = cfg.get("ideal_value_reason", "No reason provided")
    st.markdown(f"**Ideal Value:** `{ideal_value}`")
    st.markdown(f"**Reason:** {ideal_reason}")

def show_selected_option_details(param_doc: dict, selected_option: str):
    if param_doc and param_doc.get('options') and selected_option in param_doc['options']:
        option_doc = param_doc['options'][selected_option]
        with st.expander(f"Details for {selected_option}", expanded=False):
            st.markdown(option_doc.get('description', ''))
            if option_doc.get('advantages'):
                st.markdown("**Advantages:**")
                for advantage in option_doc['advantages']:
                    st.markdown(f"✅ {advantage}")
            if option_doc.get('disadvantages'):
                st.markdown("**Disadvantages:**")
                for disadvantage in option_doc['disadvantages']:
                    st.markdown(f"❌ {disadvantage}")

def display_parameter_doc_sidebar(param_doc):
    if param_doc.get('description'):
        st.markdown(f"**Description:** {param_doc['description']}")
    if param_doc.get('mathematical_definition'):
        st.markdown("**Mathematical Definition:**")
        st.code(param_doc['mathematical_definition'], language='text')
    if param_doc.get('use_cases'):
        st.markdown("**Use Cases:**")
        for use_case in param_doc['use_cases'][:3]:
            st.markdown(f"• {use_case}")
        if len(param_doc['use_cases']) > 3:
            st.markdown(f"*... and {len(param_doc['use_cases']) - 3} more*")
    if param_doc.get('options'):
        st.markdown("**Available Options:**")
        option_names = list(param_doc['options'].keys())
        option_labels = [param_doc['options'][key].get('name', key) for key in option_names]
        selected_option = st.selectbox(
            "Select an option:", option_labels, key=f"option_selector_{param_doc.get('name', 'param')}"
        )
        selected_option_key = option_names[option_labels.index(selected_option)]
        option_info = param_doc['options'][selected_option_key]
        with st.expander(f"**{option_info.get('name', selected_option_key)}**", expanded=True):
            st.markdown(option_info.get('description', ''))
            if option_info.get('mathematical_definition'):
                st.markdown("**Mathematical Definition:**")
                st.code(option_info['mathematical_definition'], language='text')
            if option_info.get('use_cases'):
                st.markdown("**Use Cases:**")
                for use_case in option_info['use_cases']:
                    st.markdown(f"• {use_case}")
            if option_info.get('advantages'):
                st.markdown("**Advantages:**")
                for advantage in option_info['advantages']:
                    st.markdown(f"✅ {advantage}")
            if option_info.get('disadvantages'):
                st.markdown("**Disadvantages:**")
                for disadvantage in option_info['disadvantages']:
                    st.markdown(f"❌ {disadvantage}")
            if option_info.get('recommended_for'):
                st.markdown("**Recommended For:**")
                for rec in option_info['recommended_for']:
                    st.markdown(f"🎯 {rec}")
            if option_info.get('not_recommended_for'):
                st.markdown("**Not Recommended For:**")
                for not_rec in option_info['not_recommended_for']:
                    st.markdown(f"⚠️ {not_rec}")
    st.markdown("---")
    st.markdown("**Parameter Configuration:**")


def query_input_box(label="Enter your query:", key="user_query"):
    """Renders a text area for user queries and returns the input string."""
    return st.text_area(label, key=key)


def pdf_upload_widget(label="Upload a PDF file", key="pdf_upload"):
    """Renders a file uploader for PDF files and returns the uploaded file object."""
    return st.file_uploader(label, type=["pdf"], key=key)

