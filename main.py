import os
import json
import logging
import time
from typing import Dict, List, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from modules.config import Config
from modules.logger import setup_logger
from modules.text_processor import TextProcessor
from modules.embedding_engine import EmbeddingEngine
from modules.summarizer import Summarizer
from modules.file_manager import FileManager

logger = setup_logger("main")

class ResearchSummarizerApp:
    def __init__(self, config_path: str = "config.json"):

        self.config = Config(config_path)

        os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.file_manager = FileManager(self.config)
        self.text_processor = TextProcessor(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_engine = EmbeddingEngine(self.config)
        self.summarizer = Summarizer(self.config)
        
    def process_file(self, filename: str) -> bool:
        try:
            if self.file_manager.should_skip_file(filename):
                logger.info(f"Skipping {filename} - already processed")
                return True
            
            paper = self.file_manager.load_paper(filename)
            if not paper:
                logger.warning(f"Failed to load {filename}")
                return False
            
            paper_content = self.text_processor.extract_paper_content(paper)
            
            if not paper_content.get("full_text") or len(paper_content.get("full_text", "")) < 100:
                logger.warning(f"Skipping {filename} - insufficient text content")
                return False
            
            full_text = paper_content["full_text"]
            chunks = self.text_processor.chunk_paper(full_text)
            embeddings = self.embedding_engine.embed_chunks(chunks)
            
            # top k representative chunks
            num_representative_chunks = min(5, len(chunks)) 
            representative_chunks = self.embedding_engine.get_representative_chunks(
                chunks, embeddings, num_chunks=num_representative_chunks
            )
            
            summary = self.summarizer.generate_summary(paper_content, representative_chunks)
            
            self.file_manager.save_processed_paper(filename, paper, summary)
            
            logger.info(f"Successfully processed {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return False
    
    def run(self) -> Dict[str, Any]:

        json_files = self.file_manager.get_input_files()
        total_files = len(json_files)
        
        if total_files == 0:
            logger.warning(f"No JSON files found in {self.config.input_dir}")
            return {"total": 0, "successful": 0, "failed": 0, "completion_percentage": 0}
            
        logger.info(f"Found {total_files} JSON files to process")
        
        successful = 0
        failed = 0
        
        # (IMPROVEMENT) Process files with parallel execution
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.process_file, filename): filename for filename in json_files}
            
            with tqdm(total=len(futures), desc="Processing papers") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error in future: {str(e)}")
                        failed += 1
                    finally:
                        pbar.update(1)
        

        stats = {
            "total": total_files,
            "successful": successful,
            "failed": failed,
            "completion_percentage": round((successful / total_files) * 100, 2) if total_files > 0 else 0
        }
        
        logger.info(f"Processing complete. Stats: {stats}")
        return stats

    def test_models(self, test_file: str) -> Dict[str, Any]:
        """
        To test different models
        """
        models = [
            {
                "name": "meta-llama/llama-3.3-70b-instruct:free",
                "description": "Llama 3.3 70B"
            }
        ]
        
        results = {}
        original_model = self.config.model_name
        
        for model in models:
            model_name = model["name"]
            description = model["description"]
            
            print(f"\nTesting model: {description} ({model_name})")

            self.config.model_name = model_name
            self.summarizer = Summarizer(self.config) 
            
            start_time = time.time()
            result = self.process_file(test_file)
            end_time = time.time()
            
            results[model_name] = {
                "success": result,
                "time_taken": round(end_time - start_time, 2),
                "description": description
            }
            
            print(f"Result: {'Success' if result else 'Failed'}")
            print(f"Time taken: {results[model_name]['time_taken']} seconds")
        
        self.config.model_name = original_model
        self.summarizer = Summarizer(self.config)

        print("\n==== Model Comparison ====")
        print(f"{'Model':<20} | {'Success':<10} | {'Time (sec)':<10}")
        print("-" * 45)
        
        for model_name, data in results.items():
            print(f"{data['description']:<20} | {str(data['success']):<10} | {data['time_taken']:<10}")
            
        return results


if __name__ == "__main__":
    app = ResearchSummarizerApp()
    
    stats = app.run()

    print(f"Processed: {stats['total']} files")
    print(f"Successful: {stats['successful']} ({stats['completion_percentage']}%)")
    print(f"Failed: {stats['failed']}")
    