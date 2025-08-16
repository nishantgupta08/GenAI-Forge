# preprocessor.py

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter, SentenceTransformersTokenTextSplitter
import re
from typing import List, Dict, Any, Optional, Union


class LangchainPreprocessor:
    """
    A unified text preprocessor compatible with LangChain's text splitters,
    supporting various preprocessing strategies via preprocessing_params.
    """

    def __init__(
        self,
        **preprocessing_params: Optional[Dict[str, Any]]
    ):
        """
        Args:
            preprocessing_params (dict): Custom params like splitter_type, chunk_size, 
                                      chunk_overlap, clean_text, etc.
        """
        # Defaults
        defaults = {
            "splitter_type": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "remove_empty": True,
            "min_chunk_length": 50,
            "merge_small_chunks": True,
            "preserve_structure": True,
            "enhance_retrieval": True,
            "clean_text": True,
            "remove_special_chars": True,
            "extract_key_sentences": True,
        }

        # Merge defaults with passed preprocessing params
        self.preprocessing_params = {**defaults, **(preprocessing_params or {})}

        # Extract parameters
        self.splitter_type = self.preprocessing_params["splitter_type"]
        self.chunk_size = self.preprocessing_params["chunk_size"]
        self.chunk_overlap = self.preprocessing_params["chunk_overlap"]
        self.remove_empty = self.preprocessing_params["remove_empty"]
        self.min_chunk_length = self.preprocessing_params["min_chunk_length"]
        self.merge_small_chunks = self.preprocessing_params["merge_small_chunks"]
        self.preserve_structure = self.preprocessing_params["preserve_structure"]
        self.enhance_retrieval = self.preprocessing_params["enhance_retrieval"]
        self.clean_text = self.preprocessing_params["clean_text"]
        self.remove_special_chars = self.preprocessing_params["remove_special_chars"]
        self.extract_key_sentences = self.preprocessing_params["extract_key_sentences"]

        # Initialize the appropriate text splitter
        self._initialize_splitter()

    def _initialize_splitter(self):
        """Initialize the text splitter based on splitter_type."""
        if self.splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif self.splitter_type == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        elif self.splitter_type == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.splitter_type == "sentence":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""]
            )
        else:
            raise ValueError(f"Unsupported splitter type: '{self.splitter_type}'")

    def _clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace and punctuation."""
        if not self.clean_text:
            return text
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        return text.strip()

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters and symbols."""
        if not self.remove_special_chars:
            return text
        
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        return text

    def _extract_key_sentences(self, text: str) -> str:
        """Extract key sentences for summarization tasks."""
        if not self.extract_key_sentences:
            return text
        
        # Simple key sentence extraction based on length and position
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length for key sentences
                key_sentences.append(sentence)
        
        return '. '.join(key_sentences) + '.'

    def _enhance_for_retrieval(self, chunks: List[str]) -> List[str]:
        """Enhance chunks for retrieval by adding context cues."""
        if not self.enhance_retrieval:
            return chunks
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Add chunk metadata for retrieval
            enhanced_chunk = f"Chunk {i+1}: {chunk}"
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks into longer ones."""
        if not self.merge_small_chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= self.chunk_size * 1.5:
                current_chunk += " " + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks

    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """Filter chunks based on length and content."""
        filtered_chunks = []
        
        for chunk in chunks:
            # Remove empty chunks
            if self.remove_empty and not chunk.strip():
                continue
            
            # Filter by minimum length
            if len(chunk) < self.min_chunk_length:
                continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks

    def run(self, text: Union[str, List[str]]) -> List[str]:
        """
        Preprocess text with the configured parameters.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            
        Returns:
            List[str]: List of preprocessed text chunks
        """
        try:
            if isinstance(text, str):
                # Clean and preprocess single text
                if self.clean_text:
                    text = self._clean_text(text)
                if self.remove_special_chars:
                    text = self._remove_special_chars(text)
                if self.extract_key_sentences:
                    text = self._extract_key_sentences(text)
                
                # Split text into chunks
                chunks = self.splitter.split_text(text)
                
            elif isinstance(text, list):
                # Process list of texts
                processed_texts = []
                for t in text:
                    if self.clean_text:
                        t = self._clean_text(t)
                    if self.remove_special_chars:
                        t = self._remove_special_chars(t)
                    if self.extract_key_sentences:
                        t = self._extract_key_sentences(t)
                    processed_texts.append(t)
                
                # Split all texts
                chunks = []
                for t in processed_texts:
                    chunks.extend(self.splitter.split_text(t))
            else:
                raise TypeError("Input must be a string or a list of strings.")
            
            # Apply post-processing
            chunks = self._filter_chunks(chunks)
            
            if self.merge_small_chunks:
                chunks = self._merge_small_chunks(chunks)
            
            if self.enhance_retrieval:
                chunks = self._enhance_for_retrieval(chunks)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")

    def get_splitter(self) -> Any:
        """
        Return the underlying LangChain text splitter object.
        """
        return self.splitter

    def get_preprocessing_params(self) -> Dict[str, Any]:
        """
        Return the current preprocessing parameters.
        """
        return self.preprocessing_params.copy() 