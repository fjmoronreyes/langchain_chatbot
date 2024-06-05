from typing import Tuple
from chroma_vector_db import ChromaVectorDB
from utils import setup_logger
from embedding_model import CustomEmbeddings
from prompts import SYSTEM_QA_PROMPT, CONTEXTUALIZE_QA_PROMPT
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import (
    create_retrieval_chain, 
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFaceEndpoint
from summarizer import Summarizer

logger = setup_logger(__name__)
TOKEN = "" #Write you API KEY here if necessary
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


class ChatBot:
    def __init__(
        self,
        folder_path: str,
        chroma_path: str = "chroma",
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        self.chroma_db = ChromaVectorDB(
            CustomEmbeddings(), folder_path, chroma_path
        )
        self.qa_prompt = SYSTEM_QA_PROMPT
        self.contextualize_prompt = CONTEXTUALIZE_QA_PROMPT

        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            max_new_tokens=512,
            temperature=0.1,
            huggingfacehub_api_token=TOKEN,
            timeout=500,
        )
        self.summarizer = Summarizer()

        self.store = {}
        logger.info("Model and tokenizer loaded successfully.")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def parse_history(self, history: BaseChatMessageHistory) -> Tuple[str, int]:
        all_messages = " ".join(
            [
                message.content
                for message in history.messages
                if isinstance(message, (HumanMessage, AIMessage))
            ]
        )
        tokens = all_messages.split()
        token_count = len(tokens)

        if token_count > 250:
            all_messages = " ".join(tokens[:250])
            token_count = 250

        return all_messages, token_count

    def generate_with_history(self, input: str, session_id: str) -> str:
        history = self.get_session_history(session_id)

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.chroma_db.get_retriever(), contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            {"input": input, "chat_history": history.messages},
            config={"configurable": {"session_id": session_id}},
        )["answer"]

        raw_history, token_count = self.parse_history(history=history)
        if token_count >= 250:
            summarized_info = self.summarizer.summarize(raw_history)
            history.messages = [AIMessage(content=summarized_info)]

        return response


if __name__ == "__main__":
    folder_path = "../data"
    inference = ChatBot(folder_path)

    session_id = "user_session"

    while True:
        user_input = input("Enter your input: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = inference.generate_with_history(user_input, session_id)
        print(f"INPUT: {user_input}")
        print(f"RESPONSE: {response}")
        print("\n")
        print("CHAT HISTORY:\n")
        chat_history = inference.get_session_history(session_id)
        print(chat_history.messages)
        print("\n")
