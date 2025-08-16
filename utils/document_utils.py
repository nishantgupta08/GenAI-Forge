"""
document_utils.py

Document processing utilities for the LLM application.
Contains functions for text extraction and document creation.
"""

import os
from typing import Any, Dict, List
from PyPDF2 import PdfReader
from langchain.schema import Document


def create_documents_from_chunks(chunks: List[str], metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Create LangChain Document objects from text chunks.
    
    Args:
        chunks (List[str]): Text chunks
        metadata (dict): Metadata to add to documents
    
    Returns:
        List[Document]: List of Document objects
    """
    if metadata is None:
        metadata = {}
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata.copy()
        doc_metadata.update({
            "chunk_id": i,
            "chunk_size": len(chunk),
            "total_chunks": len(chunks)
        })
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents


def get_text_from_file(file: Any, max_chars: int = None) -> str:
    """
    Extract text from a file object (PDF or text file), optionally truncating to max_chars.
    Args:
        file (Any): File-like object.
        max_chars (int, optional): Maximum number of characters to return.
    Returns:
        str: Extracted (and possibly truncated) text.
    """
    try:
        if hasattr(file, "type") and file.type == "application/pdf":
            text = "\n".join(
                [p.extract_text() for p in PdfReader(file).pages if p.extract_text()]
            )
        else:
            text = file.read().decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return text[:max_chars] if max_chars else text
