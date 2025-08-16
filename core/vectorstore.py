"""
vectorstore.py

Provides a builder for creating vector stores and retrievers using LangChain and FAISS.
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import Any, List


class VectorStoreBuilder:
    """
    Builder class for creating vector stores and retrievers using LangChain and FAISS.
    """
    def __init__(self, embeddings: Any):
        """
        Initialize the VectorStoreBuilder with an embedding model.
        Args:
            embeddings: Embedding model compatible with LangChain vectorstores.
        """
        self.embeddings = embeddings

    def build_vectorstore(self, texts: List[str]) -> Any:
        """
        Build a FAISS vector store from a list of texts.
        Args:
            texts (List[str]): List of documents/texts to index.
        Returns:
            FAISS vector store object.
        Raises:
            RuntimeError: If vector store creation fails.
        """
        try:
            return FAISS.from_texts(texts, embedding=self.embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to build vectorstore: {e}")

    def get_retriever(self, documents: List[Any], k: int) -> Any:
        """
        Get a retriever from the vector store for top-k retrieval.
        Args:
            documents (List[Any]): List of documents (can be strings or Document objects).
            k (int): Number of top results to retrieve.
        Returns:
            Retriever object.
        Raises:
            RuntimeError: If retriever creation fails.
        """
        try:
            # Handle both Document objects and strings
            if hasattr(documents[0], 'page_content'):
                # Documents are Document objects
                vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
            else:
                # Documents are strings
                vectorstore = self.build_vectorstore(documents)
            
            return vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            raise RuntimeError(f"Failed to get retriever: {e}")

    def generate_answer(self, decoder_instance: Any, prompt: str, query: str, context: str) -> str:
        """
        Generate answer using the retrieved context.
        Args:
            decoder_instance (Any): Decoder instance to generate the answer.
            prompt (str): Prompt to use for answer generation.
            query (str): Query used for document retrieval.
            context (str): Retrieved context from relevant documents.
        Returns:
            Generated answer as a string.
        """
        full_prompt = f"{prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return decoder_instance.run(full_prompt)
