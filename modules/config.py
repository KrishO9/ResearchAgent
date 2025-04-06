import os
import json
from typing import Optional, Dict, Any

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.input_dir = "../arvix_tmp"
        self.output_dir = "arvix_processed"
        self.cache_dir = "summary_cache"
        self.api_key = "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"
        self.model_name = "meta-llama/llama-3.3-70b-instruct:free"
        self.temperature = 0.3
        self.max_workers = 4
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.force_regenerate = False
        self.rate_limit_pause = 0.5
        
        if os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Failed to load config : {config_path}: {str(e)}")
    
    def save_to_file(self, config_path: str) -> None:
        config_data = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model_name": self.embedding_model_name,
            "force_regenerate": self.force_regenerate,
            "rate_limit_pause": self.rate_limit_pause
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"Failed to save  {config_path}: {str(e)}")

#Default configuration file
default_config = {
    "input_dir": "arvix_tmp",
    "output_dir": "arvix_processed",
    "cache_dir": "summary_cache",
    "api_key": "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",
    "model_name": "deepseek/deepseek-chat-v3-0324:free",
    "temperature": 0.3,
    "max_workers": 4,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model_name": "all-MiniLM-L6-v2",
    "force_regenerate": False,
    "rate_limit_pause": 0.5
}

if __name__ == "__main__":
    if not os.path.exists("config.json"):
        with open("config.json", 'w') as f:
            json.dump(default_config, f, indent=4)
        print("Created default config.json file")