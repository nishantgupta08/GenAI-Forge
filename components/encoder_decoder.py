from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from typing import Any, Dict, Optional


class LangchainEncoderDecoder:
    """
    Encoder-Decoder model class using HuggingFace's text2text-generation pipeline.
    Suitable for summarization, translation, question-answering, etc.
    """

    def __init__(
        self,
        model_name: str,
        encoding_params: Optional[Dict[str, Any]] = None,
        decoding_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the encoder-decoder pipeline.

        Args:
            model_name (str): HuggingFace model name or path.
            encoding_params (Optional[Dict]): Options like pooling, normalization, layer (used for logging or future use).
            decoding_params (Optional[Dict]): Generation parameters like temperature, max_length, top_k, top_p, etc.
        """
        self.model_name = model_name
        self.encoding_params = encoding_params or {}
        self.decoding_params = decoding_params or {}

        # Optionally store encoding params for analysis or logging
        self.pooling = self.encoding_params.get("pooling", "mean")
        self.normalization = self.encoding_params.get("normalize", "none")
        self.layer = self.encoding_params.get("layer", -1)

        try:
            self.pipeline = pipeline(
                task="text2text-generation",
                model=self.model_name,
                **self.decoding_params
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize encoder-decoder pipeline: {e}")

    def run(self, text: str, max_input_length: int = 1024) -> str:
        """
        Run the encoder-decoder pipeline on input text.

        Args:
            text (str): Input text.
            max_input_length (int): Max characters to consider from input.

        Returns:
            str: Generated text (summary or answer).
        """
        input_text = text[:max_input_length]
        try:
            output = self.pipeline(input_text)
            return output[0].get("summary_text") or output[0].get("generated_text")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e}")

    def get_llm(self) -> Any:
        """
        Return the LangChain-compatible HuggingFace pipeline.

        Returns:
            HuggingFacePipeline: LLM wrapper.
        """
        return HuggingFacePipeline(pipeline=self.pipeline)
