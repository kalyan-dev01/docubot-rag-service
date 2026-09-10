import logging
from sentence_transformers import CrossEncoder
from config import RAGConfig

config = RAGConfig()
logger = logging.getLogger(__name__)

reranker_model = None

class RerankingError(Exception):
    pass

def get_reranker_model():
    try:
        logger.info("Loading reranker model...")
        model = CrossEncoder(config.cross_encoder_model_name)
        logger.info("Reranker model loaded")
        return model

    except Exception as e:
        logger.error("Failed to load reranker model")
        raise RerankingError(
            "Failed to load reranker model"
        ) from e

def rerank_chunks(query:str,chunks:list[dict],top_k_final:int)-> list[dict]:
    global reranker_model
    if reranker_model is None:
        reranker_model = get_reranker_model()
    pairs = []
    for chunk in chunks:
        pairs.append([query,chunk['chunk_text']])
    try:
        reranked_results = reranker_model.predict(pairs)
    except Exception as e:
        logger.error('Failed reranking')
        raise RerankingError("Failed reranking") from e

    logger.info(f'Reranking Process Done {len(reranked_results)}')

    for i in range(len(chunks)):
        chunks[i]['score'] = reranked_results[i]


    chunks.sort(key=lambda x : x['score'],reverse=True)

    return chunks[:top_k_final]

