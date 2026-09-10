import logging
import requests
from sentence_transformers import SentenceTransformer
from config import LLMConfig

llmconfig = LLMConfig()
logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass

def build_context(chunks: list[dict]) -> str:

    context = ""

    for i in range(len(chunks)):

        chunk = chunks[i]

        page_number = chunk["metadata"]["page_number"]
        chunk_text = chunk["chunk_text"]

        context += f"""--- SOURCE {i + 1} ---Page: {page_number}Content:{chunk_text}"""

    return context

def build_prompt(query:str,context:str) -> tuple[str,str]:
    system_prompt = """
    You are a document question-answering assistant.
    Your task is to answer the user's question using ONLY the information provided in the context.

    Rules:

    1. Use only the provided context.
    2. Do not use outside knowledge.
    3. Do not guess or make up information.
    4. If the context does not contain enough information to answer the question, say:
    "I don't know based on the provided context."
    5. Every factual statement must include the page number in this format:
    [Page N]
    6. Keep the answer concise and directly answer the user's question.
    7. If multiple pages contain relevant information, cite each relevant page.
    """
    user_prompt = f"""Context:{context}Question:{query}Answer using only the context above."""
    return system_prompt,user_prompt

def call_llm(system_prompt:str,user_prompt:str)-> str:
    payload = {
        "model":llmconfig.llm_model_name,
        "messages":[
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":"/no_think " + user_prompt
            }
        ],
        "max_tokens":llmconfig.max_tokens
    }
    headers = {
        "Content-Type":'application/json',
        "Authorization":f'Bearer {llmconfig.api_key}'
    }
    try:
        response = requests.post(llmconfig.llm_url,json=payload,headers=headers)
        response.raise_for_status()
    except Exception as e:
        logger.error("Failed to call LLM")
        raise LLMError("Failed to call LLM") from e

    return response.json()["choices"][0]["message"]["content"]
