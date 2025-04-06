import os
import json
from typing import Dict, List, Any, Optional

from modules.config import Config
from modules.logger import setup_logger

logger = setup_logger("file_manager")

class FileManager:
    def __init__(self, config: Config):
        self.config = config
    
    def get_input_files(self) -> List[str]:

        if not os.path.exists(self.config.input_dir):
            logger.warning(f"{self.config.input_dir} does not exist")
            return []
        
        return [f for f in os.listdir(self.config.input_dir) if f.endswith('.json')]
    
    def load_paper(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a paper from a JSON file.
        """
        input_path = os.path.join(self.config.input_dir, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                paper = json.load(f)
            return paper
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return None
    
    def should_skip_file(self, filename: str) -> bool:
        """
        Checks if a file is already processed or not
        """
        # Skip if force regenerate is enabled
        if self.config.force_regenerate:
            return False
        
        # Check if the output file exists and has a summary
        output_path = os.path.join(self.config.output_dir, filename)
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if "summary" in existing:
                        return True
            except Exception:
                pass
        
        return False
    
    def save_processed_paper(self, filename: str, paper: Dict[str, Any], summary: Dict[str, Any]) -> bool:

        output_path = os.path.join(self.config.output_dir, filename)
        
        try:
            # Create output object (without the original description to save space)
            output_obj = {
                "category": paper.get("category", ""),
                "scraper_id": paper.get("scraper_id", ""),
                "website_url": paper.get("website_url", ""),
                "timestamp": paper.get("timestamp", ""),
                "author": paper.get("author", ""),
                "image_url": paper.get("image_url", None),
                "source_type": paper.get("source_type", ""),
                "hyperlinks": paper.get("hyperlinks", []),
                "data": {
                    "headline": paper["data"].get("headline", "") if "data" in paper else "",
                },
                "summary": summary
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_obj, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")
            return False