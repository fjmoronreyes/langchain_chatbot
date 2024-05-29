from typing import List
from langchain_community.vectorstores import Chroma
from embedding_model import CustomEmbeddings
from text_extractor import TextExtractor
from utils import setup_logger

logger = setup_logger(__name__)

class ChromaVectorDB:
    def __init__(self, embedding_model: CustomEmbeddings, folder_path: str, chroma_path: str = "chroma"):
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.create_db(folder_path=folder_path)
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> Chroma:
        """Load the vectorstore from the persistent directory."""
        return Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)

    def create_db(self, folder_path: str):
        extractor = TextExtractor()
        chunks = extractor.get_chunks(folder_path)
        texts = [chunk.page_content for chunk in chunks]
        self.vectorstore = Chroma.from_texts(texts=texts, embedding=self.embedding_model, persist_directory=self.chroma_path)
        logger.info(f"VectorDB Chroma created and persisted in {self.chroma_path}.")

    def query_db(self, query: str, k: int = 3) -> List[str]:
        if not query:
            logger.warning("Empty query")
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [result[0].page_content for result in results]
    
    def get_retriever(self):
        return self.vectorstore.as_retriever()

if __name__ == "__main__":
    folder_path = "../data"
    embedding_model = CustomEmbeddings()
    chroma_db = ChromaVectorDB(embedding_model, folder_path)

    query = "Fecha de lanzamiento del juego Stellar Blade para la consola PlayStation 5"
    import pdb; pdb.set_trace()
    results = chroma_db.query_db(query)
    for result in results:
        print(f"Chunk: {result}")
