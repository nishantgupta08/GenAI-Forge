"""
task_runner.py

Orchestrates task execution for RAG QA, Normal QA, and Summarisation using loaded models and decoding strategies.
"""

from langchain.chains import RetrievalQA
from core.vectorstore import VectorStoreBuilder
from utils.document_utils import get_text_from_file, create_documents_from_chunks
from components.encoder import LangchainEncoder
from components.decoder import LangchainDecoder
from components.encoder_decoder import LangchainEncoderDecoder
from components.preprocessor import LangchainPreprocessor
# Strategies removed - using empty defaults
TASK_PREPROCESSING_PARAMS = {}


class TaskOrchestrator:
    """
    Orchestrates various NLP tasks (RAG-based QA, Normal QA, Summarisation) using provided model configurations.
    """

    def __init__(self, models_config: dict):
        """
        Initialize TaskOrchestrator with model configurations.

        Args:
            models_config (dict): Dictionary of model configurations.
        """
        self.models_config = models_config



    def run_rag_qa(
        self,
        file,
        encoder_name: str,
        decoder_name: str,
        prompt: str,
        query: str,
        encoding_params: dict = None,
        decoding_params: dict = None,
        preprocessing_config: dict = None
    ) -> str:
        """
        Run Retrieval-Augmented Generation (RAG) QA pipeline.

        Args:
            file: Uploaded file object.
            encoder_name (str): Encoder model name.
            decoder_name (str): Decoder model name.
            prompt (str): Prompt for the model.
            query (str): User question.
            encoding_params (dict): Encoding parameters (pooling, normalization, etc.).
            decoding_params (dict): Decoding parameters (temperature, top_k, etc.).
            preprocessing_config (dict): Text preprocessing configuration.

        Returns:
            str: Model-generated answer.
        """
        encoding_params = encoding_params or {}
        decoding_params = decoding_params or {}
        preprocessing_config = preprocessing_config or {}

        # Extract text from file
        raw_text = get_text_from_file(file)
        
        # Apply RAG-specific preprocessing
        rag_params = TASK_PREPROCESSING_PARAMS.get("RAG-based QA", [])
        rag_config = {}
        if rag_params:
            rag_config = {param.name: param.ideal for param in rag_params}
        config = {**rag_config, **preprocessing_config}
        
        preprocessor = LangchainPreprocessor(**config)
        chunks = preprocessor.run(raw_text)
        
        # Create documents with metadata
        documents = create_documents_from_chunks(chunks, {
            "source": file.name if hasattr(file, 'name') else "uploaded_file",
            "preprocessing": preprocessing_config,
            "task": "RAG-based QA"
        })

        # Create encoder & vector store
        encoder_instance = LangchainEncoder(encoder_name, **encoding_params)
        # Note: For vector store integration, we need the underlying LangChain embedding model
        # The run() method would be used for direct encoding operations
        vectorstore_builder = VectorStoreBuilder(encoder_instance.get_encoder())
        
        # Extract text content from documents for vector store building
        document_texts = [doc.page_content for doc in documents]
        vectorstore = vectorstore_builder.build_vectorstore(document_texts)
        
        # Create retriever from the built vector store
        retriever = vectorstore.as_retriever(search_kwargs={"k": int(decoding_params.get("top_k", 5))})

        # Create decoder (LLM)
        decoder_instance = LangchainDecoder(decoder_name, **decoding_params)
        # Note: For LangChain chains, we need the underlying LangChain LLM
        # The run() method would be used for direct text generation
        decoder = decoder_instance.get_llm()

        # Step 1: Explicitly retrieve relevant documents using the query
        print(f"Query: '{query}'")
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Step 2: Prepare context from retrieved documents
        if retrieved_docs:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
            print(f"Context length: {len(context)} characters")
            for i, doc in enumerate(retrieved_docs):
                print(f"Doc {i+1} (length: {len(doc.page_content)}): {doc.page_content[:100]}...")
        else:
            context = "No relevant documents found."
            print("No documents retrieved for the query.")
        
        # Step 3: Generate answer using the retrieved context
        if not context or context.strip() == "" or context == "No relevant documents found.":
            print("Context is empty or no relevant documents found. Aborting generation.")
            return "No relevant context could be retrieved from the document. Please check your PDF or try a different query."

        if not query or query.strip() == "":
            print("Query is empty. Aborting generation.")
            return "Query is empty. Please enter a valid question."

        # Provide a default prompt if none is given
        if not prompt or prompt.strip() == "":
            prompt = "Please answer the following question based on the provided context."

        full_prompt = f"{prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        print(f"Full prompt length: {len(full_prompt)} characters")
        print(f"Full prompt preview: {full_prompt[:500]}")  # Print first 500 chars for debugging

        try:
            answer = decoder_instance.run(full_prompt)
        except Exception as e:
            print(f"Decoder failed: {e}")
            return f"Text generation failed: {e}"

        return answer

    def run_qa(
        self,
        model_name: str,
        query: str,
        encoding_params: dict = None,
        decoding_params: dict = None,
        preprocessing_config: dict = None
    ) -> str:
        """
        Run a model for direct (non-RAG) QA.

        Args:
            model_name (str): Name of the encoder-decoder model.
            query (str): Question text.
            encoding_params (dict): Encoder configuration.
            decoding_params (dict): Decoder configuration.
            preprocessing_config (dict): Text preprocessing configuration.

        Returns:
            str: Generated answer.
        """
        encoding_params = encoding_params or {}
        decoding_params = decoding_params or {}
        preprocessing_config = preprocessing_config or {}

        # Apply QA-specific preprocessing
        qa_params = TASK_PREPROCESSING_PARAMS.get("Normal QA", [])
        qa_config = {}
        if qa_params:
            qa_config = {param.name: param.ideal for param in qa_params}
        config = {**qa_config, **preprocessing_config}
        
        preprocessor = LangchainPreprocessor(**config)
        processed_chunks = preprocessor.run(query)
        processed_query = processed_chunks[0] if processed_chunks else query

        encoder_decoder = LangchainEncoderDecoder(
            model_name=model_name,
            encoding_params=encoding_params,
            decoding_params=decoding_params
        )
        answer = encoder_decoder.run(processed_query)
        return answer

    def run_summarisation(
        self,
        model_name: str,
        input_text: str,
        encoding_params: dict = None,
        decoding_params: dict = None,
        preprocessing_config: dict = None
    ) -> str:
        """
        Run a model for Summarisation.

        Args:
            model_name (str): Name of the encoder-decoder model.
            input_text (str): Text to summarize.
            encoding_params (dict): Encoder configuration.
            decoding_params (dict): Decoder configuration.
            preprocessing_config (dict): Text preprocessing configuration.

        Returns:
            str: Generated summary.
        """
        encoding_params = encoding_params or {}
        decoding_params = decoding_params or {}
        preprocessing_config = preprocessing_config or {}

        # Apply summarization-specific preprocessing
        summary_params = TASK_PREPROCESSING_PARAMS.get("Summarisation", [])
        summary_config = {}
        if summary_params:
            summary_config = {param.name: param.ideal for param in summary_params}
        config = {**summary_config, **preprocessing_config}
        
        preprocessor = LangchainPreprocessor(**config)
        chunks = preprocessor.run(input_text)
        
        # For summarization, use the first chunk or combine if needed
        if len(chunks) == 1:
            processed_text = chunks[0]
        else:
            # If multiple chunks, use the first one or combine them
            processed_text = chunks[0] if len(chunks[0]) > 500 else " ".join(chunks[:2])

        encoder_decoder = LangchainEncoderDecoder(
            model_name=model_name,
            encoding_params=encoding_params,
            decoding_params=decoding_params
        )
        answer = encoder_decoder.run(processed_text)
        return answer

    def run_document_indexing(
        self,
        files,
        doc_type: str,
        encoder_name: str,
        encoding_params: dict = None,
        preprocessing_config: dict = None
    ) -> dict:
        """
        Run document indexing pipeline for multiple files.

        Args:
            files: List of uploaded file objects.
            doc_type (str): Document domain type (e.g., 'Healthcare', 'Fintech').
            encoder_name (str): Encoder model name.
            encoding_params (dict): Encoding parameters (pooling, normalization, etc.).
            preprocessing_config (dict): Text preprocessing configuration.

        Returns:
            dict: Indexing results with statistics and metadata.
        """
        encoding_params = encoding_params or {}
        preprocessing_config = preprocessing_config or {}
        
        results = {
            "total_files": len(files),
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "document_type": doc_type,
            "encoder_model": encoder_name,
            "file_results": [],
            "indexing_metadata": {
                "encoding_params": encoding_params,
                "preprocessing_config": preprocessing_config
            }
        }
        
        # Create encoder with document type
        encoder_instance = LangchainEncoder(
            model_name=encoder_name,
            document_type=doc_type,
            **encoding_params
        )
        
        # Create vector store builder
        vectorstore_builder = VectorStoreBuilder(encoder_instance.get_encoder())
        
        all_documents = []
        all_embeddings = []
        
        for file in files:
            file_result = {
                "filename": file.name if hasattr(file, 'name') else "unknown",
                "size": file.size if hasattr(file, 'size') else 0,
                "status": "processing",
                "chunks": 0,
                "embeddings": 0,
                "error": None
            }
            
            try:
                # Extract text from file
                raw_text = get_text_from_file(file)
                
                # Apply preprocessing
                preprocessor = LangchainPreprocessor(**preprocessing_config)
                chunks = preprocessor.run(raw_text)
                
                # Create documents with metadata
                documents = create_documents_from_chunks(chunks, {
                    "source": file.name if hasattr(file, 'name') else "uploaded_file",
                    "document_type": doc_type,
                    "preprocessing": preprocessing_config,
                    "task": "Document Indexing"
                })
                
                # Generate embeddings
                document_texts = [doc.page_content for doc in documents]
                embeddings = encoder_instance.run(document_texts)
                
                # Store results
                all_documents.extend(documents)
                all_embeddings.extend(embeddings)
                
                file_result.update({
                    "status": "success",
                    "chunks": len(chunks),
                    "embeddings": len(embeddings)
                })
                
                results["successful_files"] += 1
                results["total_chunks"] += len(chunks)
                results["total_embeddings"] += len(embeddings)
                
            except Exception as e:
                file_result.update({
                    "status": "failed",
                    "error": str(e)
                })
                results["failed_files"] += 1
            
            results["file_results"].append(file_result)
        
        # Build vector store with all documents
        if all_documents:
            try:
                document_texts = [doc.page_content for doc in all_documents]
                vectorstore = vectorstore_builder.build_vectorstore(document_texts)
                
                results["vectorstore_status"] = "success"
                results["vectorstore_info"] = {
                    "total_documents": len(all_documents),
                    "total_embeddings": len(all_embeddings)
                }
                
            except Exception as e:
                results["vectorstore_status"] = "failed"
                results["vectorstore_error"] = str(e)
        else:
            results["vectorstore_status"] = "no_documents"
        
        return results
