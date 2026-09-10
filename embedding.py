import logging
from sentence_transformers import SentenceTransformer
from config import RAGConfig

logger = logging.getLogger(__name__)
config = RAGConfig()

class EmbeddingError(Exception):
    pass


def get_embedding_model():
    try:
        logger.info("Loading embedding model...")
        model = SentenceTransformer(config.embedding_model_name)
        logger.info("Embedding model loaded")
        return model
    except Exception as e:
        logger.error("Failed to load embedding model")
        raise EmbeddingError("Failed to load embedding model") from e

embedding_model = None

def embed_chunks(chunks:list[dict])-> list[dict]:
    global embedding_model
    if embedding_model is None:
        embedding_model = get_embedding_model()
    texts = []
    for chunk in chunks:
        texts.append(chunk['chunk_text'])
    try:
        embeddings = embedding_model.encode(texts)
    except Exception as e:
        logger.error('Failed to embed chunks')
        raise EmbeddingError("Failed to embed chunks") from e

    for i in range(len(chunks)):
        chunks[i]['embedding'] = embeddings[i]

    return chunks
    

    