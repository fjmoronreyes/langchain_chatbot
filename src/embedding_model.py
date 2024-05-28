from typing import List
from FlagEmbedding import BGEM3FlagModel
from utils import setup_logger

logger = setup_logger(__name__)

class CustomEmbeddings:
    def __init__(self, model_name: str = 'BAAI/bge-m3', use_fp16: bool = True, normalize_embeddings: bool = True, device: str = 'cuda'):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, normalize_embeddings=normalize_embeddings, device=device)
        logger.info("Model loaded without problems.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("No texts were provided to generate embeddings.")
            return []
        embeddings = self.model.encode(texts, batch_size=12, max_length=1056)['dense_vecs']
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if not text:
            logger.warning("No text was provided to generate the query")
            return None
        embeddings = self.model.encode([text], batch_size=1, max_length=1056)['dense_vecs']
        return embeddings[0].tolist()
