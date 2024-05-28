from typing import List
from langchain_community.vectorstores import Chroma
from embedding_model import CustomEmbeddings
from text_extractor import TextExtractor
from utils import setup_logger

logger = setup_logger(__name__)

class ChromaVectorDB:
    def __init__(self, chroma_path: str, embedding_model: CustomEmbeddings):
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.vectorstore = None

    def create_db(self, folder_path: str):
        extractor = TextExtractor()
        chunks = extractor.get_chunks(folder_path)
        texts = [chunk.page_content for chunk in chunks]
        self.vectorstore = Chroma.from_texts(texts=texts, embedding=self.embedding_model, persist_directory=self.chroma_path)
        logger.info(f"VectorDB Chroma created and persisted in {self.chroma_path}.")

    def query_db(self, query: str, k: int =1) -> List[str]:
        if not query:
            logger.warning("Empty query")
            return None
        self.vectorstore = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        import pdb; pdb.set_trace()
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [result[0].page_content for result in results]

if __name__ == "__main__":
    chroma_path = "chroma"
    embedding_model = CustomEmbeddings()
    chroma_db = ChromaVectorDB(chroma_path, embedding_model)
    folder_path = "./data"
    chroma_db.create_db(folder_path)

    query = "Fecha de lanzamiento del juego para la consola PlayStation 5"
    results = chroma_db.query_db(query)
    for result in results:
        print(f"Chunk: {result}")
