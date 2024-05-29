from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from chroma_vector_db import ChromaVectorDB
from utils import setup_logger
from embedding_model import CustomEmbeddings
from prompts import SYSTEM_QA_PROMPT

logger = setup_logger(__name__)
TOKEN = "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"

class Inference:
    def __init__(self, folder_path: str, chroma_path: str = "chroma", model_name: str = "google/gemma-1.1-2b-it"):
        self.chroma_db = ChromaVectorDB(CustomEmbeddings(), folder_path, chroma_path)
        self.qa_prompt = SYSTEM_QA_PROMPT
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            token=TOKEN,
        )
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        logger.info("Model and tokenizer loaded successfully.")

    def generate(self, query: str) -> str:
        results = self.chroma_db.query_db(query, k=3)
        if not results:
            logger.warning("No relevant information found in the vector database.")
            return "Sorry, I couldn't find any relevant information."

        context = " ".join(results)

        # Create the chat prompt with query and context
        chat_prompt = self.qa_prompt.format(query=query, context=context)
        
        chat = [
            {"role": "user", "content": chat_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        outputs = self.generator(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0]["generated_text"][len(prompt):]
        return response

if __name__ == "__main__":
    folder_path = "../data"
    inference = Inference(folder_path)

    query = "Fecha de lanzamiento del juego Stellar Blade para la consola PlayStation 5"
    response = inference.generate(query)
    print(f"Response: {response}")
