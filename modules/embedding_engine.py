import numpy as np
from typing import List, Any
import faiss
from sentence_transformers import SentenceTransformer

from modules.config import Config
from modules.logger import setup_logger

logger = setup_logger("embedding_engine")

class EmbeddingEngine:
    """Generate embeddings of text and find representative chunks."""
    
    def __init__(self, config: Config):
        self.config = config
        
        logger.info(f"Loading embedding model: {config.embedding_model_name}")
        self.embedding_model = SentenceTransformer(config.embedding_model_name)
        
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.
        """
        if not chunks:
            return np.array([])
        
        embeddings = [self.embedding_model.encode(chunk) for chunk in chunks]
        return np.array(embeddings)
    
    def get_representative_chunks(self, chunks: List[str], embeddings: np.ndarray, num_chunks: int = 3) -> List[str]:
        """
        Select representative chunks by clustering embeddings.
        """
        if len(chunks) <= num_chunks or len(embeddings) == 0:
            return chunks
        
        n_clusters = min(num_chunks, len(chunks))
        
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)

        embeddings_float32 = embeddings.astype(np.float32)
        
        index.add(embeddings_float32)
        
        # clustering
        kmeans = faiss.Kmeans(embedding_dim, n_clusters, niter=20)
        kmeans.train(embeddings_float32)
        
        # get centroids
        centroids = kmeans.centroids
        
        # Find chunks closest to centroids
        _, centroid_chunk_indices = index.search(centroids, 1)
        
        representative_indices = list(set([idx[0] for idx in centroid_chunk_indices]))
        
        if len(representative_indices) < num_chunks:
            remaining_indices = [i for i in range(len(chunks)) if i not in representative_indices]
            representative_indices.extend(remaining_indices[:num_chunks - len(representative_indices)])
        
        return [chunks[idx] for idx in representative_indices]