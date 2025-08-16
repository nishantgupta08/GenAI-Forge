# encoder.py

from langchain.embeddings import HuggingFaceEmbeddings
import torch
import numpy as np
from typing import Optional, List, Union, Any, Dict


class LangchainEncoder:
    """
    A unified encoder compatible with LangChain's vector store,
    supporting normalization and pooling strategies via encoding_params.
    """

    def __init__(
        self,
        model_name: str,
        **encoding_params: Optional[Dict[str, Any]]
    ):
        """
        Args:
            model_name (str): HuggingFace model ID or path.
            encoding_params (dict): Custom params like pooling, normalization, device, etc.
        """
        self.model_name = model_name

        # Defaults
        defaults = {
            "pooling": "mean",
            "normalize": "none",
            "layer": -1,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        # Merge defaults with passed encoding params
        self.encoding_params = {**defaults, **(encoding_params or {})}

        self.pooling = self.encoding_params["pooling"]
        self.normalization = self.encoding_params["normalize"]
        self.layer = self.encoding_params["layer"]
        
        # Handle 'auto' device parameter
        device_param = self.encoding_params["device"]
        self.device = self._resolve_device(device_param)

        allowed_normalizations = {"none", "l1", "l2", "max", "zscore", "minmax"}
        if self.normalization not in allowed_normalizations:
            raise ValueError(f"Unsupported normalization: '{self.normalization}'")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "trust_remote_code": False,
                "device": self.device
            },
            encode_kwargs={
                "normalize_embeddings": True  # Keeps HF's internal norm; your norm applies after
            }
        )
        
        # Move model to device after loading to avoid meta tensor issues
        if hasattr(self.embedding_model, 'client') and hasattr(self.embedding_model.client, 'model'):
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    # Use to_empty() for meta tensors, to() for regular tensors
                    if hasattr(self.embedding_model.client.model, 'to_empty'):
                        self.embedding_model.client.model = self.embedding_model.client.model.to_empty("cuda")
                    else:
                        self.embedding_model.client.model = self.embedding_model.client.model.to("cuda")
                else:
                    if hasattr(self.embedding_model.client.model, 'to_empty'):
                        self.embedding_model.client.model = self.embedding_model.client.model.to_empty("cpu")
                    else:
                        self.embedding_model.client.model = self.embedding_model.client.model.to("cpu")
            except Exception as e:
                # If device placement fails, continue with default device
                print(f"Warning: Could not move model to {self.device}: {e}")

    def _resolve_device(self, device_param: str) -> str:
        """
        Resolve device parameter to a valid PyTorch device.
        
        Args:
            device_param (str): Device parameter from config ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            str: Valid PyTorch device string
        """
        if device_param == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif device_param == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif device_param == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available, falling back to CPU")
            return "cpu"
        else:
            return device_param

    @staticmethod
    def get_supported_parameters(model_name: str) -> set:
        """
        Get the set of supported parameters for a specific encoder model.
        
        Args:
            model_name (str): HuggingFace model name or path
            
        Returns:
            set: Set of supported parameter names
        """
        # Standard parameters supported by most embedding models
        base_supported = {
            'pooling', 'normalize', 'device', 'batch_size', 'max_length',
            'truncation', 'stride', 'special_tokens', 'return_tensors'
        }
        
        # Model-specific parameter support
        model_lower = model_name.lower()
        
        # BERT-style models support more parameters
        if any(bert_type in model_lower for bert_type in ['bert', 'roberta', 'distilbert']):
            return base_supported | {'tokenizer', 'output_hidden_states', 'aggregation'}
        
        # Sentence transformers typically support fewer parameters
        elif any(st_type in model_lower for st_type in ['sentence-transformers', 'all-mpnet', 'all-minilm']):
            return {'device', 'batch_size', 'normalize'}
        
        # Default to base supported parameters
        return base_supported

    def _apply_normalization(self, embeddings: List[List[float]]) -> List[List[float]]:
        if self.normalization == "none":
            return embeddings

        embeddings_np = np.array(embeddings)

        if self.normalization == "l2":
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np /= (norms + 1e-8)
        elif self.normalization == "l1":
            norms = np.sum(np.abs(embeddings_np), axis=1, keepdims=True)
            embeddings_np /= (norms + 1e-8)
        elif self.normalization == "max":
            max_vals = np.max(np.abs(embeddings_np), axis=1, keepdims=True)
            embeddings_np /= (max_vals + 1e-8)
        elif self.normalization == "zscore":
            mean = np.mean(embeddings_np, axis=1, keepdims=True)
            std = np.std(embeddings_np, axis=1, keepdims=True)
            embeddings_np = (embeddings_np - mean) / (std + 1e-8)
        elif self.normalization == "minmax":
            min_vals = np.min(embeddings_np, axis=1, keepdims=True)
            max_vals = np.max(embeddings_np, axis=1, keepdims=True)
            embeddings_np = (embeddings_np - min_vals) / (max_vals - min_vals + 1e-8)

        return embeddings_np.tolist()

    def run(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Encode a string or list of strings with optional post-normalization.
        """
        try:
            if isinstance(text, str):
                embedding = self.embedding_model.embed_query(text)
                return self._apply_normalization([embedding])[0]
            elif isinstance(text, list):
                embeddings = self.embedding_model.embed_documents(text)
                return self._apply_normalization(embeddings)
            else:
                raise TypeError("Input must be a string or a list of strings.")
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}")

    def get_encoder(self) -> Any:
        """
        Return the underlying LangChain-compatible encoder object.
        """
        return self.embedding_model
