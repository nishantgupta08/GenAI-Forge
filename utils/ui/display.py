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


def document_type_selector(key="document_type"):
    """Renders a document type selector with predefined categories."""
    document_types = {
        "Healthcare": {
            "description": "Medical documents, clinical reports, research papers",
            "icon": "🏥",
            "encoding_params": {
                "pooling": "cls",
                "normalize": "l2",
                "max_length": 1024
            }
        },
        "Fintech": {
            "description": "Financial reports, trading documents, regulatory filings",
            "icon": "💰",
            "encoding_params": {
                "pooling": "mean",
                "normalize": "l2",
                "max_length": 1024
            }
        },
        "Legal": {
            "description": "Contracts, legal briefs, case law documents",
            "icon": "⚖️",
            "encoding_params": {
                "pooling": "cls",
                "normalize": "l2",
                "max_length": 2048
            }
        },
        "Technology": {
            "description": "Technical documentation, API docs, code documentation",
            "icon": "💻",
            "encoding_params": {
                "pooling": "mean",
                "normalize": "l2",
                "max_length": 1024
            }
        },
        "Education": {
            "description": "Academic papers, textbooks, educational materials",
            "icon": "📚",
            "encoding_params": {
                "pooling": "mean",
                "normalize": "l2",
                "max_length": 1024
            }
        },
        "General": {
            "description": "General purpose documents, news articles, web content",
            "icon": "📄",
            "encoding_params": {
                "pooling": "mean",
                "normalize": "l2",
                "max_length": 1024
            }
        }
    }
    
    # Create options with icons and descriptions
    options = []
    for doc_type, config in document_types.items():
        options.append(f"{config['icon']} {doc_type}")
    
    selected_option = st.selectbox(
        "Select Document Type:",
        options=options,
        key=key,
        help="Choose the domain type of your documents for optimized encoding"
    )
    
    # Extract the document type from the selected option
    selected_type = selected_option.split(" ", 1)[1] if " " in selected_option else selected_option
    
    # Display description and encoding parameters
    if selected_type in document_types:
        config = document_types[selected_type]
        st.info(f"**{config['description']}**")
        
        with st.expander("Domain-specific Encoding Parameters", expanded=False):
            st.json(config['encoding_params'])
    
    return selected_type, document_types.get(selected_type, {}).get('encoding_params', {})


def chunking_preset_selector(uploaded_files=None, key="chunking_preset"):
    """Renders a user-friendly chunking preset selector with smart defaults."""
    
    # Define chunking presets
    chunking_presets = {
        "Auto (Recommended)": {
            "description": "Understands headings/sections. Best for reports and proposals.",
            "icon": "🤖",
            "params": {
                "splitter_type": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "preserve_structure": True,
                "enhance_retrieval": True
            }
        },
        "Pages/Slides": {
            "description": "One chunk per page/slide. Best for decks & scanned PDFs.",
            "icon": "📄",
            "params": {
                "splitter_type": "page",
                "chunk_size": 2000,
                "chunk_overlap": 0,
                "preserve_structure": True,
                "enhance_retrieval": False
            }
        },
        "Sentences": {
            "description": "Keeps sentences intact. Best for legal/long-form docs.",
            "icon": "📝",
            "params": {
                "splitter_type": "sentence",
                "chunk_size": 800,
                "chunk_overlap": 100,
                "preserve_structure": True,
                "enhance_retrieval": True
            }
        },
        "Fixed": {
            "description": "Fastest. Use when you just need quick results.",
            "icon": "⚡",
            "params": {
                "splitter_type": "character",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "preserve_structure": False,
                "enhance_retrieval": False
            }
        }
    }
    
    # Smart recommendation based on uploaded files
    recommended_preset = "Auto (Recommended)"
    if uploaded_files:
        file_extensions = [f.name.lower().split('.')[-1] for f in uploaded_files if hasattr(f, 'name')]
        
        # Check for PowerPoint files
        if any(ext in ['pptx', 'ppt'] for ext in file_extensions):
            recommended_preset = "Pages/Slides"
        # Check for PDFs with many pages (estimate based on file size)
        elif any(ext == 'pdf' for ext in file_extensions):
            large_pdfs = [f for f in uploaded_files if hasattr(f, 'size') and f.size > 5 * 1024 * 1024]  # >5MB
            if large_pdfs:
                recommended_preset = "Pages/Slides"
    
    # Create options with icons and descriptions
    options = []
    for preset_name, config in chunking_presets.items():
        options.append(f"{config['icon']} {preset_name}")
    
    # Find index of recommended preset
    recommended_index = list(chunking_presets.keys()).index(recommended_preset)
    
    # Display recommendation
    if uploaded_files and recommended_preset != "Auto (Recommended)":
        st.info(f"💡 **Smart Recommendation**: Based on your files, we recommend **{recommended_preset}**")
    
    # Radio button selection
    selected_option = st.radio(
        "Choose Chunking Method:",
        options=options,
        index=recommended_index,
        key=key,
        help="Select how you want to split your documents into searchable chunks"
    )
    
    # Extract the preset name from the selected option
    selected_preset = selected_option.split(" ", 1)[1] if " " in selected_option else selected_option
    
    # Display description
    if selected_preset in chunking_presets:
        config = chunking_presets[selected_preset]
        st.caption(f"💡 {config['description']}")
    
    return selected_preset, chunking_presets.get(selected_preset, {}).get('params', {})


def advanced_chunking_options(key_prefix="advanced_chunking"):
    """Renders advanced chunking options in a collapsible section."""
    
    with st.expander("🔧 Advanced Chunking Options", expanded=False):
        st.markdown("**Fine-tune your chunking settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                key=f"{key_prefix}_size",
                help="Larger chunks = more context, smaller chunks = faster indexing"
            )
            
            overlap = st.slider(
                "Overlap",
                min_value=0,
                max_value=500,
                value=150,
                step=25,
                key=f"{key_prefix}_overlap",
                help="Overlap between chunks to prevent information loss"
            )
        
        with col2:
            preserve_structure = st.checkbox(
                "Preserve Document Structure",
                value=True,
                key=f"{key_prefix}_structure",
                help="Keep headings, paragraphs, and formatting intact"
            )
            
            enhance_retrieval = st.checkbox(
                "Enhance for Retrieval",
                value=True,
                key=f"{key_prefix}_retrieval",
                help="Add metadata and keywords for better search results"
            )
        
        # Splitter type selection
        splitter_type = st.selectbox(
            "Splitter Type",
            options=["recursive", "sentence", "character", "page"],
            index=0,
            key=f"{key_prefix}_splitter",
            help="How to split the text: recursive (smart), sentence, character, or page-based"
        )
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "preserve_structure": preserve_structure,
            "enhance_retrieval": enhance_retrieval,
            "splitter_type": splitter_type
        }
    
    return None

