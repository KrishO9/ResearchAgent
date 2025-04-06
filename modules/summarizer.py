import time
import json
import hashlib
from typing import Dict, List, Any, Optional
from openai import OpenAI

from modules.config import Config
from modules.logger import setup_logger

# Setup logger
logger = setup_logger("summarizer")

class Summarizer:
    """Generator for paper summaries using LLMs."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize Open Router client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.api_key,
        )
        
        logger.info(f"Initialized summarizer with model: {config.model_name}")
    
    def get_cache_path(self, paper_id: str) -> Optional[str]:
        if not self.config.cache_dir:
            return None
            
        return f"{self.config.cache_dir}/{paper_id}.json"
    
    def load_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """
        Load generated summary from cache if available.
        """
        if not cache_path or self.config.force_regenerate:
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug(f"No cache found at {cache_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load from cache {cache_path}: {e}")
            return None
    
    def save_to_cache(self, cache_path: str, content: Dict[str, Any]) -> bool:
        """
        Save generated summary to cache.
        """
        if not cache_path or not self.config.cache_dir:
            return False
            
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_path}: {e}")
            return False
        
    def generate_summary(self, paper_content: Dict[str, Any], representative_chunks: List[str]) -> Dict[str, Any]:
        # use the title for the hash
        paper_id = hashlib.md5(paper_content['title'].encode()).hexdigest()
        
        # Check cache first if present
        cache_path = self.get_cache_path(paper_id)
        cached_summary = self.load_from_cache(cache_path)
        
        if cached_summary:
            logger.info(f"Using cached summary for {paper_content['title']}")
            return cached_summary
        
        chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(representative_chunks)])
        
        prompt = f"""
        You are a skilled science communicator creating summaries of research papers that are 
        adaptable for readers with different levels of expertise. Create a summary of the following 
        research paper that is engaging, accurate, and layered in complexity.
        
        Paper Information:
        - Title: {paper_content['title']}
        - Abstract: {paper_content['abstract']}
        - Category: {paper_content['category']}
        
        Key Representative Excerpts from the Full Paper:
        {chunks_text}
        
        Create an adaptable summary with the following structure:
        
        1. "headline": A compelling, clear title that captures the essence of the research
        
        2. "tldr": A one-sentence summary that anyone can understand
        
        3. "context": Brief background explaining why this research matters in the real world
        
        4. "methodology": A clear explanation of the methods and approach used by the researchers
        
        5. "key_points": 3-5 bullet points highlighting the main findings and implications
        
        6. "accessible_explanation": A 2-3 paragraph explanation that a general audience can understand, 
           using analogies or examples when helpful
        
        7. "significance": The broader impact of this work and why it represents an advance
        
        8. "questions_raised": 2-3 thought-provoking questions this research raises
        
        Format your response as a JSON object with these keys.
        
        Your summary should be:
        - Factually accurate (don't add details not present in the paper)
        - Engaging for different audience types (general readers, students, researchers)
        - Written with clarity and a human touch
        - Free of unnecessary jargon, but precise about key concepts
        
        Return ONLY the JSON object, with no additional text.
        """
        
        try:
            logger.info(f"Generating summary for {paper_content['title']} using {self.config.model_name}")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.choices[0].message.content
            
            json_str = response_text.strip()
            
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            summary = json.loads(json_str)
            
            if cache_path:
                self.save_to_cache(cache_path, summary)
                
            time.sleep(self.config.rate_limit_pause)
                
            return summary
            
        except Exception as e:
            #logger.error(f"Error generating summary: {str(e)}")
            # Return fallback summary
            return {
                "headline": paper_content['title'],
                "tldr": "Error generating summary.",
                "context": "Error generating summary.",
                "methodology": "Error generating summary.",
                "key_points": ["Error generating summary."],
                "accessible_explanation": "Error generating summary.",
                "significance": "Error generating summary.",
                "questions_raised": ["Error generating summary."]
            }