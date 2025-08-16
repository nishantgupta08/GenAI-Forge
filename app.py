import streamlit as st
import json
import pandas as pd
import os

from core.config_manager import ConfigManager
from utils.ui import aggrid_model_picker, create_preprocessing_table, create_encoding_table, create_decoding_table

# --- Initialize configuration manager ---
config_manager = ConfigManager()

# --- Load models from new config structure ---
encoder_models = config_manager.get_encoder_models()
decoder_models = config_manager.get_decoder_models()
encoder_decoder_models = config_manager.get_encoder_decoder_models()

# Create DataFrames for model selection
encoder_df = pd.DataFrame(encoder_models) if encoder_models else pd.DataFrame()
decoder_df = pd.DataFrame(decoder_models) if decoder_models else pd.DataFrame()
encoder_decoder_df = pd.DataFrame(encoder_decoder_models) if encoder_decoder_models else pd.DataFrame()

st.set_page_config(page_title="🧠 GenAI Playground", layout="wide")
st.title("🧠 GenAI Playground")

# --- Sidebar with task selection ---
with st.sidebar:
    st.markdown("## 🎯 Task Selection")
    tasks = config_manager.get_available_tasks()
    task = st.selectbox("Choose Task", tasks)

# --- Main content area ---
st.markdown(f"### {config_manager.get_task_icon(task)} {task}")
st.write(config_manager.get_task_description(task))

selected_encoder = selected_decoder = selected_encoder_decoder = None

# Get task configuration dynamically
task_blocks = config_manager.get_task_blocks(task)

# --- Model selection based on task blocks ---
if "encoding" in task_blocks and "decoding" in task_blocks:
    # Tasks that need separate encoder and decoder (like RAG-based QA)
    if not encoder_df.empty:
        st.subheader("Select an Encoder Model")
        selected_encoder = aggrid_model_picker(encoder_df, key="aggrid_encoder_model_picker")
        if selected_encoder:
            st.success(f"Selected encoder: {selected_encoder['name']}")
            st.write(selected_encoder)
    else:
        st.warning("No encoder models available in configuration.")

    if not decoder_df.empty:
        st.subheader("Select a Decoder Model")
        selected_decoder = aggrid_model_picker(decoder_df, key="aggrid_decoder_model_picker")
        if selected_decoder:
            st.success(f"Selected decoder: {selected_decoder['name']}")
            st.write(selected_decoder)
    else:
        st.warning("No decoder models available in configuration.")
else:
    # Tasks that use encoder-decoder models (like Normal QA and Summarisation)
    if not encoder_decoder_df.empty:
        st.subheader("Select an Encoder-Decoder Model")
        selected_encoder_decoder = aggrid_model_picker(encoder_decoder_df, key="aggrid_encoder_decoder_model_picker")
        if selected_encoder_decoder:
            st.success(f"Selected encoder-decoder: {selected_encoder_decoder['name']}")
            st.write(selected_encoder_decoder)
    else:
        st.warning("No encoder-decoder models available in configuration.")

# --- Parameter configuration section ---
st.markdown("---")
st.markdown("## ⚙️ Parameter Configuration")

# --- Parameter configuration tabs ---
if task_blocks:
    tab_names = []
    for block in task_blocks:
        if block.lower() == "encoding":
            tab_names.append("🔧 Encoding")
        elif block.lower() == "decoding":
            tab_names.append("🎲 Decoding")
        elif block.lower() == "preprocessing":
            tab_names.append("📝 Preprocessing")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        
        for i, block in enumerate(task_blocks):
            param_type = f"{block}_parameters"
            params = config_manager.get_task_parameters(task, param_type)
            
            if params:
                with tabs[i]:
                    if block.lower() == "encoding":
                        # Pass appropriate model name based on task
                        if task == "RAG-based QA" and selected_encoder:
                            model_name = selected_encoder['name']
                        elif selected_encoder_decoder:
                            model_name = selected_encoder_decoder['name']
                        else:
                            model_name = None
                        encoding_params = create_encoding_table(params, task, model_name)
                    elif block.lower() == "decoding":
                        # Pass appropriate model name based on task
                        if task == "RAG-based QA" and selected_decoder:
                            model_name = selected_decoder['name']
                        elif selected_encoder_decoder:
                            model_name = selected_encoder_decoder['name']
                        else:
                            model_name = None
                        decoding_params = create_decoding_table(params, task, model_name)
                    elif block.lower() == "preprocessing":
                        preprocessing_params = create_preprocessing_table(params, task)

# --- Task-specific input and execution ---
from utils.ui.display import query_input_box
st.markdown("---")

# Dynamic task execution based on task configuration
def execute_task(task_name, **kwargs):
    """Execute the selected task with the given parameters."""
    from core.task_orchestrator import TaskOrchestrator
    # Create a models config dict for backward compatibility
    models_config = {
        "ENCODER_ONLY_MODELS": encoder_models,
        "DECODER_ONLY_MODELS": decoder_models,
        "ENCODER_DECODER_MODELS": encoder_decoder_models
    }
    orchestrator = TaskOrchestrator(models_config)
    
    if task_name == "RAG-based QA":
        return orchestrator.run_rag_qa(**kwargs)
    elif task_name == "Abstractive Summarization":
        return orchestrator.run_summarisation(**kwargs)
    elif task_name == "Question Answering":
        return orchestrator.run_qa(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}")

# Task-specific input widgets and execution
if task == "RAG-based QA":
    from utils.ui.display import pdf_upload_widget
    st.subheader("User Query and Document Upload")
    user_query = query_input_box()
    uploaded_pdf = pdf_upload_widget()
    
    run_clicked = st.button("Run RAG QA")
    if run_clicked:
        if not (selected_encoder and selected_decoder):
            st.error("Please select both an encoder and a decoder model.")
        elif not user_query:
            st.error("Please enter a query.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF document.")
        else:
            with st.spinner("Running RAG-based QA..."):
                kwargs = {
                    "file": uploaded_pdf,
                    "encoder_name": selected_encoder['name'],
                    "decoder_name": selected_decoder['name'],
                    "prompt": "",
                    "query": user_query,
                    "encoding_params": encoding_params if 'encoding_params' in locals() else {},
                    "decoding_params": decoding_params if 'decoding_params' in locals() else {},
                    "preprocessing_config": preprocessing_params if 'preprocessing_params' in locals() else {}
                }
                answer = execute_task(task, **kwargs)
                st.success("Answer:")
                st.write(answer)

elif task == "Abstractive Summarization":
    st.subheader("Text to Summarize")
    input_text = query_input_box(label="Enter text to summarize:", key="summarize_text")
    
    run_clicked = st.button("Run Summarisation")
    if run_clicked:
        if not selected_encoder_decoder:
            st.error("Please select an encoder-decoder model.")
        elif not input_text:
            st.error("Please enter text to summarize.")
        else:
            with st.spinner("Running Summarisation..."):
                kwargs = {
                    "model_name": selected_encoder_decoder['name'],
                    "input_text": input_text,
                    "encoding_params": encoding_params if 'encoding_params' in locals() else {},
                    "decoding_params": decoding_params if 'decoding_params' in locals() else {},
                    "preprocessing_config": preprocessing_params if 'preprocessing_params' in locals() else {}
                }
                summary = execute_task(task, **kwargs)
                st.success("Summary:")
                st.write(summary)

elif task == "Question Answering":
    st.subheader("User Query")
    user_query = query_input_box(label="Enter your question:", key="qa_query")
    
    run_clicked = st.button("Run Question Answering")
    if run_clicked:
        if not selected_encoder_decoder:
            st.error("Please select an encoder-decoder model.")
        elif not user_query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Running Question Answering..."):
                kwargs = {
                    "model_name": selected_encoder_decoder['name'],
                    "query": user_query,
                    "encoding_params": encoding_params if 'encoding_params' in locals() else {},
                    "decoding_params": decoding_params if 'decoding_params' in locals() else {},
                    "preprocessing_config": preprocessing_params if 'preprocessing_params' in locals() else {}
                }
                answer = execute_task(task, **kwargs)
                st.success("Answer:")
                st.write(answer)

else:
    st.info(f"Task '{task}' is not yet implemented. Please select a supported task.")
