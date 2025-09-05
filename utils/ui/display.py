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


def visual_chunking_selector(uploaded_files, key="visual_chunking"):
    """Renders a visual chunking interface where users can select example chunks to determine pattern."""
    
    if not uploaded_files:
        return None, {}
    
    st.subheader("👁️ Visual Chunking Pattern Selection")
    st.markdown("**Select example chunks to automatically determine the best chunking pattern for your documents**")
    
    # Get the first uploaded file for preview
    preview_file = uploaded_files[0]
    
    # Extract text from the file
    try:
        from utils.document_utils import get_text_from_file
        raw_text = get_text_from_file(preview_file)
        
        # Limit text for preview (first 5000 characters)
        preview_text = raw_text[:5000] + "..." if len(raw_text) > 5000 else raw_text
        
        st.info(f"📄 **Preview from:** {preview_file.name} (showing first 5000 characters)")
        
    except Exception as e:
        st.error(f"Could not extract text from {preview_file.name}: {str(e)}")
        return None, {}
    
    # Generate different chunking patterns for preview
    chunking_patterns = {
        "Auto (Recommended)": {
            "splitter_type": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "description": "Smart chunking that understands document structure"
        },
        "Pages/Slides": {
            "splitter_type": "page", 
            "chunk_size": 2000,
            "chunk_overlap": 0,
            "description": "One chunk per page or slide"
        },
        "Sentences": {
            "splitter_type": "sentence",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "description": "Keeps sentences intact"
        },
        "Fixed Size": {
            "splitter_type": "character",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "description": "Fixed character-based chunks"
        }
    }
    
    # Create chunks for each pattern
    pattern_chunks = {}
    for pattern_name, config in chunking_patterns.items():
        try:
            from components.preprocessor import LangchainPreprocessor
            preprocessor = LangchainPreprocessor(**config)
            chunks = preprocessor.run(preview_text)
            pattern_chunks[pattern_name] = chunks[:5]  # Show first 5 chunks
        except Exception as e:
            st.warning(f"Could not generate chunks for {pattern_name}: {str(e)}")
            pattern_chunks[pattern_name] = []
    
    # Display chunking patterns with selectable chunks
    selected_chunks = []
    selected_pattern = None
    
    for pattern_name, chunks in pattern_chunks.items():
        if not chunks:
            continue
            
        with st.expander(f"📋 {pattern_name} - {chunking_patterns[pattern_name]['description']}", expanded=pattern_name == "Auto (Recommended)"):
            
            # Pattern description
            st.caption(f"💡 {chunking_patterns[pattern_name]['description']}")
            
            # Show chunks with selection
            for i, chunk in enumerate(chunks):
                chunk_id = f"{pattern_name}_{i}"
                
                col1, col2 = st.columns([1, 20])
                
                with col1:
                    is_selected = st.checkbox(
                        "✓", 
                        key=f"chunk_select_{chunk_id}",
                        help=f"Select this chunk as an example"
                    )
                    
                    if is_selected:
                        selected_chunks.append({
                            "pattern": pattern_name,
                            "chunk_index": i,
                            "content": chunk,
                            "config": chunking_patterns[pattern_name]
                        })
                
                with col2:
                    # Display chunk with highlighting
                    chunk_preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                    
                    if is_selected:
                        st.markdown(f"**Chunk {i+1}** (Selected)")
                        st.success(chunk_preview)
                    else:
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(chunk_preview)
                
                st.markdown("---")
    
    # Pattern selection based on selected chunks
    if selected_chunks:
        # Count selections by pattern
        pattern_counts = {}
        for chunk in selected_chunks:
            pattern = chunk["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Find most selected pattern
        if pattern_counts:
            selected_pattern = max(pattern_counts, key=pattern_counts.get)
            selected_config = chunking_patterns[selected_pattern]
            
            st.success(f"🎯 **Detected Pattern:** {selected_pattern}")
            st.info(f"Based on your selections, we recommend: **{selected_pattern}**")
            
            # Show selected chunks summary
            with st.expander("📊 Selected Chunks Summary", expanded=True):
                st.write(f"**Pattern:** {selected_pattern}")
                st.write(f"**Selected Chunks:** {len(selected_chunks)}")
                st.write(f"**Configuration:**")
                st.json(selected_config)
            
            return selected_pattern, selected_config
    
    # Default fallback
    if not selected_chunks:
        st.info("👆 **Select some example chunks above to automatically determine the best chunking pattern**")
        return "Auto (Recommended)", chunking_patterns["Auto (Recommended)"]
    
    return selected_pattern, chunking_patterns.get(selected_pattern, chunking_patterns["Auto (Recommended)"])


def chunking_pattern_visualizer(chunks, pattern_name, max_chunks=5):
    """Visualize chunks in a nice format for pattern comparison."""
    
    st.subheader(f"📊 {pattern_name} Pattern Preview")
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        with st.container():
            st.markdown(f"**Chunk {i+1}**")
            
            # Chunk content with syntax highlighting
            st.code(chunk, language="text")
            
            # Chunk metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{len(chunk)} chars")
            with col2:
                st.metric("Words", f"{len(chunk.split())} words")
            with col3:
                st.metric("Lines", f"{len(chunk.splitlines())} lines")
            
            st.markdown("---")
    
    if len(chunks) > max_chunks:
        st.info(f"Showing first {max_chunks} chunks out of {len(chunks)} total chunks")


def chunking_pattern_comparison(uploaded_files, key="pattern_comparison"):
    """Show side-by-side comparison of different chunking patterns."""
    
    if not uploaded_files:
        return None
    
    st.subheader("🔍 Chunking Pattern Comparison")
    st.markdown("**Compare how different chunking methods split your document**")
    
    # Get preview text
    preview_file = uploaded_files[0]
    try:
        from utils.document_utils import get_text_from_file
        raw_text = get_text_from_file(preview_file)
        preview_text = raw_text[:3000] + "..." if len(raw_text) > 3000 else raw_text
    except Exception as e:
        st.error(f"Could not extract text: {str(e)}")
        return None
    
    # Define patterns to compare
    patterns = {
        "Auto": {"splitter_type": "recursive", "chunk_size": 1000, "chunk_overlap": 150},
        "Pages": {"splitter_type": "page", "chunk_size": 2000, "chunk_overlap": 0},
        "Sentences": {"splitter_type": "sentence", "chunk_size": 800, "chunk_overlap": 100},
        "Fixed": {"splitter_type": "character", "chunk_size": 500, "chunk_overlap": 50}
    }
    
    # Generate chunks for each pattern
    pattern_results = {}
    for name, config in patterns.items():
        try:
            from components.preprocessor import LangchainPreprocessor
            preprocessor = LangchainPreprocessor(**config)
            chunks = preprocessor.run(preview_text)
            pattern_results[name] = {
                "chunks": chunks[:3],  # Show first 3 chunks
                "total_chunks": len(chunks),
                "config": config
            }
        except Exception as e:
            st.warning(f"Could not generate {name} chunks: {str(e)}")
            pattern_results[name] = {"chunks": [], "total_chunks": 0, "config": config}
    
    # Display comparison in columns
    cols = st.columns(len(pattern_results))
    
    for i, (pattern_name, result) in enumerate(pattern_results.items()):
        with cols[i]:
            st.markdown(f"**{pattern_name} Pattern**")
            st.caption(f"Total chunks: {result['total_chunks']}")
            
            # Show first few chunks
            for j, chunk in enumerate(result["chunks"]):
                with st.expander(f"Chunk {j+1}", expanded=False):
                    chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                    st.text(chunk_preview)
                    st.caption(f"{len(chunk)} chars, {len(chunk.split())} words")
    
    return pattern_results

