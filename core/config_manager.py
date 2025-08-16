"""
config_manager.py

Manages configuration loading and access for the new config file structure.
"""

import os
import json
from typing import Dict, List, Any, Optional

class ConfigManager:
    """Manages all configuration files and provides access methods."""
    
    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        self.tasks_config = self._load_json("tasks_config.json")
        self.models_config = self._load_json("models_config.json")
        self.parameters_config = self._load_json("parameters_config.json")
    
    def _load_json(self, filename: str) -> Dict:
        """Load a JSON configuration file."""
        try:
            with open(os.path.join(self.config_dir, filename), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using empty config.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing {filename}: {e}")
            return {}
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks."""
        return list(self.tasks_config.keys())
    
    def get_task_config(self, task_name: str) -> Dict:
        """Get configuration for a specific task."""
        return self.tasks_config.get(task_name, {})
    
    def get_task_blocks(self, task_name: str) -> List[str]:
        """Get parameter blocks for a task."""
        task_config = self.get_task_config(task_name)
        return task_config.get("blocks", [])
    
    def get_task_description(self, task_name: str) -> str:
        """Get description for a task."""
        task_config = self.get_task_config(task_name)
        return task_config.get("description", "No description available.")
    
    def get_task_icon(self, task_name: str) -> str:
        """Get icon for a task."""
        # Default icons based on task type
        icons = {
            "RAG-based QA": "🔍",
            "Abstractive Summarization": "📝",
            "Extractive Summarization": "✂️",
            "Question Answering": "❓",
            "Text Classification": "🏷️",
            "Sentiment Analysis": "😊",
            "Named Entity Recognition": "👤",
            "Text Generation": "✍️"
        }
        return icons.get(task_name, "🤖")
    
    def get_task_parameters(self, task_name: str, param_type: str) -> Dict:
        """Get parameters for a specific task and parameter type."""
        # Get base parameters
        base_params = self.parameters_config.get(param_type, {})
        
        # Get task-specific overrides
        task_config = self.get_task_config(task_name)
        overrides = task_config.get("overrides", {}).get(param_type.split("_")[0], {})
        
        # Merge base parameters with overrides
        merged_params = {}
        for param_name, param_config in base_params.items():
            merged_config = param_config.copy()
            
            # Apply task-specific overrides
            if param_name in overrides:
                override = overrides[param_name]
                if "ideal" in override:
                    merged_config["ideal"] = override["ideal"]
                if "ideal_value_reason" in override:
                    merged_config["ideal_value_reason"] = override["ideal_value_reason"]
            
            merged_params[param_name] = merged_config
        
        return merged_params
    
    def get_models_by_type(self, model_type: str) -> List[Dict]:
        """Get models filtered by type (encoder, decoder, encoder_decoder)."""
        models = self.models_config.get("models", [])
        return [model for model in models if model.get("type") == model_type]
    
    def get_encoder_models(self) -> List[Dict]:
        """Get all encoder models."""
        return self.get_models_by_type("encoder")
    
    def get_decoder_models(self) -> List[Dict]:
        """Get all decoder models."""
        return self.get_models_by_type("decoder")
    
    def get_encoder_decoder_models(self) -> List[Dict]:
        """Get all encoder-decoder models."""
        return self.get_models_by_type("encoder_decoder")
    
    def get_all_models(self) -> List[Dict]:
        """Get all models."""
        return self.models_config.get("models", [])
    
    def reload_configurations(self):
        """Reload all configuration files."""
        self.tasks_config = self._load_json("tasks_config.json")
        self.models_config = self._load_json("models_config.json")
        self.parameters_config = self._load_json("parameters_config.json")
        
    def get_models_by_type(self, model_type: str) -> List[Dict]:
        """Get models filtered by type (encoder, decoder, encoder-decoder).
        Accepts either "encoder-decoder" or "encoder_decoder" and normalizes for matching.
        """
        models = self.models_config.get("models", [])
        target = (model_type or "").replace("-", "_")
        out = []
        for model in models:
            t = model.get("type")
            t_norm = (t or "").replace("-", "_")
            if t_norm == target:
                out.append(model)
        return out
