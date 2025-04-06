import re
from typing import Dict, List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.logger import setup_logger

logger = setup_logger("text_processor")

class TextProcessor:
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def clean_text(self, text: str) -> str:
        """
        removes noise and formatting issues.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # remove escape sequences
        text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        
        # alphanumeric, spaces, punctuation
        cleaned = re.sub(r'[^\w\s.,!?()-:;"\'%]', ' ', text)
        
        # extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def extract_paper_content(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant content from paper for processing.
        """
        extracted = {}
        
        # Extract title/headline
        if 'data' in paper and 'headline' in paper['data'] and paper['data']['headline']:
            extracted['title'] = paper['data']['headline']
        else:
            extracted['title'] = "Untitled Research Paper"
        
        if 'data' in paper and 'description' in paper['data'] and paper['data']['description']:
            extracted['abstract'] = self.clean_text(paper['data']['description'])
        else:
            extracted['abstract'] = ""

        full_text_parts = []
        
        if extracted['abstract']:
            full_text_parts.append(extracted['abstract'])
        
        if 'data' in paper:
            if 'content' in paper['data'] and isinstance(paper['data']['content'], str):
                full_text_parts.append(self.clean_text(paper['data']['content']))
                
            if 'sections' in paper['data'] and isinstance(paper['data']['sections'], list):
                for section in paper['data']['sections']:
                    if isinstance(section, dict) and 'text' in section:
                        full_text_parts.append(self.clean_text(section['text']))
                    elif isinstance(section, str):
                        full_text_parts.append(self.clean_text(section))
        
        extracted['full_text'] = " ".join(full_text_parts)
        
        if extracted['full_text'] == extracted['abstract']:
            logger.warning(f"Only abstract found for paper: {extracted['title']}")
        
        if 'author' in paper and paper['author'] not in [None, "Unknown"]:
            extracted['author'] = paper['author']
        else:
            extracted['author'] = "Unknown"

        if 'category' in paper and paper['category']:
            extracted['category'] = paper['category']
        else:
            extracted['category'] = "Scientific Research"
        
        if 'hyperlinks' in paper and paper['hyperlinks'] and len(paper['hyperlinks']) > 0:
            extracted['url'] = paper['hyperlinks'][0]
        else:
            extracted['url'] = ""
            
        if 'timestamp' in paper and paper['timestamp']:
            extracted['timestamp'] = paper['timestamp']
        else:
            extracted['timestamp'] = ""
        
        return extracted
    
    def chunk_paper(self, text: str) -> List[str]:
        """
        Split paper text into chunks for embedding and similarity search
        """
        if not text or len(text.strip()) < 100:  # Arbitrary minimum length
            return [text] if text else []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Log chunking statistics
        logger.info(f"Split text into {len(chunks)} chunks")
        
        return chunks