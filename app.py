import streamlit as st
import json
import pandas as pd
import os

from core.config_manager import ConfigManager
from utils.ui import aggrid_model_picker, create_preprocessing_table, create_encoding_table, create_decoding_table
from utils.ui.display import document_type_selector, pdf_upload_widget, chunking_preset_selector, advanced_chunking_options

# --- Initialize configuration manager ---
config_manager = ConfigManager()

# --- Load models from new config structure ---
encoder_models = config_manager.get_encoder_models()

# Create DataFrames for model selection
encoder_df = pd.DataFrame(encoder_models) if encoder_models else pd.DataFrame()

st.set_page_config(page_title="📚 Document Indexing System", layout="wide")
st.title("📚 Document Indexing System")
st.markdown("Index your documents with domain-specific embeddings for efficient search and retrieval")

# Set the task to Document Indexing
task = "Document Indexing"

# --- Main content area ---
st.markdown(f"### {config_manager.get_task_icon(task)} {task}")
st.write(config_manager.get_task_description(task))

# Get task configuration dynamically
task_blocks = config_manager.get_task_blocks(task)

# --- Document Type Selection ---
st.subheader("📋 Document Configuration")
col1, col2 = st.columns([1, 1])

with col1:
    selected_doc_type, domain_encoding_params = document_type_selector()

with col2:
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents to index",
        type=["pdf", "txt", "docx", "pptx", "ppt"],
        accept_multiple_files=True,
        help="Upload multiple documents for batch indexing"
    )

# --- Chunking Configuration ---
if uploaded_files:
    st.subheader("✂️ Document Chunking")
    st.markdown("Choose how to split your documents into searchable chunks")
    
    # Helpful information for consultants
    with st.expander("💡 What is document chunking?", expanded=False):
        st.markdown("""
        **Document chunking** breaks your documents into smaller, searchable pieces called "chunks". 
        This helps the AI system find relevant information quickly and accurately.
        
        **Why it matters:**
        - 📚 **Better Search**: Smaller chunks = more precise search results
        - ⚡ **Faster Processing**: Optimized chunk sizes improve performance  
        - 🎯 **Context Preservation**: Smart chunking keeps related information together
        
        **Our presets are designed for different document types:**
        - **Auto**: Works great for most business documents
        - **Pages/Slides**: Perfect for presentations and scanned PDFs
        - **Sentences**: Ideal for legal documents and contracts
        - **Fixed**: Fastest option for quick results
        """)
    
    # Chunking preset selection
    selected_chunking_preset, chunking_params = chunking_preset_selector(uploaded_files)
    
    # Advanced options (collapsed by default)
    advanced_params = advanced_chunking_options()
    
    # Use advanced params if provided, otherwise use preset params
    final_chunking_params = advanced_params if advanced_params else chunking_params
    
    # Display current chunking configuration
    with st.expander("📊 Current Chunking Configuration", expanded=False):
        st.json(final_chunking_params)
else:
    final_chunking_params = {}

# --- Encoder Model Selection ---
st.subheader("🤖 Select Encoder Model")
if not encoder_df.empty:
    selected_encoder = aggrid_model_picker(encoder_df, key="aggrid_encoder_model_picker")
    if selected_encoder:
        st.success(f"Selected encoder: {selected_encoder['name']}")
        # Display model details in a compact format
        with st.expander("Model Details", expanded=False):
            st.json({
                "name": selected_encoder['name'],
                "source": selected_encoder['source'],
                "family": selected_encoder['family'],
                "params": selected_encoder['params']
            })
else:
    st.warning("No encoder models available in configuration.")
    selected_encoder = None

# --- Parameter configuration section ---
st.markdown("---")
st.markdown("## ⚙️ Parameter Configuration")

# --- Parameter configuration tabs ---
if task_blocks:
    tab_names = []
    for block in task_blocks:
        if block.lower() == "encoding":
            tab_names.append("🔧 Encoding")
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
                        # Apply domain-specific encoding parameters
                        if domain_encoding_params and selected_encoder:
                            # Merge domain-specific params with base params
                            for param_name, param_value in domain_encoding_params.items():
                                if param_name in params:
                                    params[param_name]["ideal"] = param_value
                            
                            st.info(f"🎯 Applied {selected_doc_type} domain-specific encoding parameters")
                        
                        model_name = selected_encoder['name'] if selected_encoder else None
                        encoding_params = create_encoding_table(params, task, model_name)
                    elif block.lower() == "preprocessing":
                        # Use chunking preset parameters instead of default preprocessing
                        if final_chunking_params:
                            # Override default preprocessing params with chunking preset params
                            for param_name, param_value in final_chunking_params.items():
                                if param_name in params:
                                    params[param_name]["ideal"] = param_value
                            
                            st.info(f"✂️ Applied {selected_chunking_preset} chunking configuration")
                        
                        preprocessing_params = create_preprocessing_table(params, task)

# --- Document Indexing Execution ---
st.markdown("---")
st.markdown("## 🚀 Document Indexing")

def execute_document_indexing(files, doc_type, encoder_name, encoding_params, preprocessing_params):
    """Execute document indexing with the given parameters."""
    from core.task_orchestrator import TaskOrchestrator
    # Create a models config dict for backward compatibility
    models_config = {
        "ENCODER_ONLY_MODELS": encoder_models,
        "DECODER_ONLY_MODELS": [],
        "ENCODER_DECODER_MODELS": []
    }
    orchestrator = TaskOrchestrator(models_config)
    
    return orchestrator.run_document_indexing(
        files=files,
        doc_type=doc_type,
        encoder_name=encoder_name,
        encoding_params=encoding_params,
        preprocessing_config=preprocessing_params
    )

# Indexing execution
if uploaded_files and selected_encoder:
    st.subheader("📊 Indexing Summary")
    
    # Display configuration information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Type", selected_doc_type)
        st.metric("Number of Files", len(uploaded_files))
    
    with col2:
        st.metric("Chunking Method", selected_chunking_preset)
        st.metric("Encoder Model", selected_encoder['name'])
    
    with col3:
        # Show chunking parameters
        if final_chunking_params:
            st.metric("Chunk Size", f"{final_chunking_params.get('chunk_size', 'N/A')}")
            st.metric("Overlap", f"{final_chunking_params.get('chunk_overlap', 'N/A')}")
    
    # Show file details
    with st.expander("📁 Uploaded Files", expanded=True):
        for i, file in enumerate(uploaded_files):
            file_type = file.name.split('.')[-1].upper() if '.' in file.name else 'Unknown'
            st.write(f"{i+1}. **{file.name}** ({file.size:,} bytes) - {file_type}")
    
    # Show chunking configuration
    with st.expander("✂️ Chunking Configuration", expanded=False):
        st.json(final_chunking_params)
    
    # Indexing button
    index_clicked = st.button("🚀 Start Document Indexing", type="primary")
    
    if index_clicked:
        if not uploaded_files:
            st.error("Please upload at least one document.")
        elif not selected_encoder:
            st.error("Please select an encoder model.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    # Get parameters
                    encoding_params_dict = encoding_params if 'encoding_params' in locals() else {}
                    preprocessing_params_dict = preprocessing_params if 'preprocessing_params' in locals() else {}
                    
                    # Execute indexing
                    result = execute_document_indexing(
                        files=uploaded_files,
                        doc_type=selected_doc_type,
                        encoder_name=selected_encoder['name'],
                        encoding_params=encoding_params_dict,
                        preprocessing_params=preprocessing_params_dict
                    )
                    
                    st.success("✅ Document indexing completed successfully!")
                    
                    # Display results
                    with st.expander("📈 Indexing Results", expanded=True):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"❌ Indexing failed: {str(e)}")
                    st.exception(e)

elif uploaded_files and not selected_encoder:
    st.warning("⚠️ Please select an encoder model to proceed with indexing.")
elif not uploaded_files and selected_encoder:
    st.warning("⚠️ Please upload documents to index.")
else:
    st.info("👆 Upload documents and select an encoder model to start indexing.")
