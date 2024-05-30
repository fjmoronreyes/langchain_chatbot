# Chat Bot Project

## Overview

IMPORTANT!
Use gradio_interface.py as a script to run the chatbot interactively.

This project contains scripts and resources for a chatbot system using a combination of language models and vector databases. The main components are:

- **`chat_bot.py`**: Defines the `ChatBot` class for interacting with the chatbot, including handling chat history and generating responses.
- **`chroma_vector_db.py`**: Manages the `ChromaVectorDB` class, which handles embedding and querying documents in the vector database.
- **`embedding_model.py`**: Defines the `CustomEmbeddings` class for generating embeddings using a specific model.
- **`gradio_interface.py`**: Provides a Gradio interface for interacting with the chatbot.
- **`inference.py`**: Script for generating responses using a pre-trained model.
- **`prompts.py`**: Contains prompt templates for the chatbot.
- **`summarizer.py`**: Defines the `Summarizer` class for summarizing text.
- **`text_extractor.py`**: Extracts and processes text from PDF documents.
- **`utils.py`**: Contains utility functions and setup for logging.

## Scripts

### `chat_bot.py`
- **Purpose**: Interact with the chatbot, manage chat history, and generate responses.
- **Key Components**:
  - `ChatBot`: A class encapsulating the chatbot logic.
  - `get_session_history(session_id: str)`: Retrieves chat history for a given session.
  - `generate_with_history(input: str, session_id: str)`: Generates a response considering the chat history.

### `chroma_vector_db.py`
- **Purpose**: Manage the vector database for storing and querying document embeddings.
- **Key Components**:
  - `ChromaVectorDB`: A class handling the creation, loading, and querying of the vector database.
  - `create_db(folder_path: str)`: Creates the vector database from documents in the specified folder.
  - `query_db(query: str, k: int)`: Queries the database for similar documents.

### `embedding_model.py`
- **Purpose**: Generate embeddings for documents and queries.
- **Key Components**:
  - `CustomEmbeddings`: A class for generating embeddings using a specific model.
  - `embed_documents(texts: List[str])`: Generates embeddings for a list of documents.
  - `embed_query(text: str)`: Generates an embedding for a query.

### `gradio_interface.py`
- **Purpose**: Provide a web interface for interacting with the chatbot using Gradio.
- **Key Components**:
  - `chat_with_bot(user_input)`: Handles user input, generates responses, and displays chat history.
  
### `inference.py`
- **Purpose**: Generate responses using a pre-trained language model.
- **Key Components**:
  - `Inference`: A class for managing the inference process.
  - `generate(input: str)`: Generates a response based on input and context from the vector database.

### `prompts.py`
- **Purpose**: Define prompt templates for the chatbot.
- **Key Components**:
  - `SYSTEM_QA_PROMPT`: Template for the system QA prompt.
  - `CONTEXTUALIZE_QA_PROMPT`: Template for contextualizing user questions.

### `summarizer.py`
- **Purpose**: Summarize text using a pre-trained model.
- **Key Components**:
  - `Summarizer`: A class for summarizing text.
  - `summarize(text: str)`: Summarizes the given text.

### `text_extractor.py`
- **Purpose**: Extract and process text from PDF documents.
- **Key Components**:
  - `TextExtractor`: A class for extracting and processing text.
  - `get_chunks(folder_path: str)`: Extracts text chunks from documents in the specified folder.

## Requirements

The project uses the following key libraries:
- `langchain`
- `gradio`
- `transformers`
- `pdfplumber`

Make sure these libraries are installed in your Python environment. You can install them using `pip`

# Using gradio

To interact with the chatbot using the Gradio interface, run:

```
python gradio_interface.py
```

This will launch a web interface where you can input queries and receive responses from the chatbot.

## Notes

- Ensure the data directory exists and contains the documents you want to use for the chatbot.
- Adjust the folder_path variable in the scripts if your documents are located elsewhere.