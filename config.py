from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGConfig:
    embedding_model_name:str = 'sentence-transformers/all-MiniLM-L6-v2'
    cross_encoder_model_name:str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    embedding_max_tokens:int = 250
    chunk_size_tokens:int = 150
    chunk_overlap_tokens:int = 20
    top_k_retrieve:int = 15
    top_k_final:int = 3

@dataclass
class LLMConfig:
    api_key:str = os.getenv('LLM_API_KEY')
    llm_url:str = os.getenv('LLM_URL')
    llm_model_name:str = os.getenv('LLM_MODEL_NAME')
    max_tokens:int = 700

if __name__ == "__main__":
    rag_config = RAGConfig()
    llm_config = LLMConfig()
    print(rag_config)
    print(llm_config.llm_url)
    print(llm_config.llm_model_name)
    print(rag_config.cross_encoder_model_name)

