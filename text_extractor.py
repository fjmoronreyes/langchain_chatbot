import pdfplumber
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from utils import setup_logger

logger = setup_logger(__name__)

class TextExtractor:
    def load_pdfs(self, folder_path: str) -> List[Document]:
        documents = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(folder_path, file_name)
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            documents.append(Document(page_content=text))
        logger.info(f"{len(documents)} documents have been loaded from {folder_path}.")
        return documents

    def get_chunks(self, folder_path: str) -> List[Document]:
        documents = self.load_pdfs(folder_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document chunks: {len(chunks)}.")
        unique_chunks = self.remove_duplicates(chunks)
        logger.info(f"Unique document chunks: {len(unique_chunks)}.")
        return unique_chunks

    def remove_duplicates(self, chunks: List[Document]) -> List[Document]:
        #this is not a good solution when scaling
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk.page_content not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk.page_content)
        logger.info(f"Filtered {len(unique_chunks)} unique chunks from {len(chunks)} total chunks.")
        return unique_chunks
