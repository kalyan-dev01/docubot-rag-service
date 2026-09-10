from transformers import AutoTokenizer
from config import RAGConfig
import logging

logger = logging.getLogger(__name__)



config = RAGConfig()
tokenizer = None

def get_tokenizer():
    global tokenizer

    if tokenizer is None:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model_name
        )
        logger.info("Tokenizer loaded")

    return tokenizer



def chunk_text(text:str,chunk_size:int,overlap:int,page_number:int) -> list[dict]:
    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text,add_special_tokens=False)
    step_size = chunk_size - overlap
    chunks = []
    for start in range(0,len(token_ids),step_size):
        window = token_ids[start:start + chunk_size]
        chunks.append({
            "chunk_text":tokenizer.decode(window,skip_special_tokens=False),
            "chunk_index":len(chunks),
            "page_number":page_number 
        })

    logger.info(f'Created {len(chunks)} chunks for page {page_number}')
    return chunks
