import logging
import chromadb
import os

from embedding import get_embedding_model
from config import RAGConfig


logger = logging.getLogger(__name__)


# -----------------------------
# ChromaDB setup
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="academic_regulations"
)

config = RAGConfig()

embedding_model = None


# -----------------------------
# Custom Exceptions
# -----------------------------

class VectoreStoreError(Exception):
    pass


class RetrievalError(Exception):
    pass


class DocumentDeleteError(Exception):
    pass


# -----------------------------
# Store Chunks
# -----------------------------

def store_chunks(chunks: list[dict]):

    chunks_text = []
    ids = []
    embeddings = []
    metadatas = []

    for chunk in chunks:

        chunks_text.append(
            chunk["chunk_text"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        ids.append(
            f"{chunk['document_id']}-"
            f"{chunk['page_number']}-"
            f"{chunk['chunk_index']}"
        )

        metadatas.append({
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "document_id": chunk["document_id"],
            "filepath": chunk["filepath"],
            "filename": chunk["filename"]
        })

        logger.info(
            f"Storing document_id: {chunk['document_id']}"
        )

    try:

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks_text,
            metadatas=metadatas
        )

        logger.info(
            f"Stored {len(chunks)} chunks in vector store"
        )

    except Exception as e:

        logger.error(
            "Failed to store chunks in vector store"
        )

        raise VectoreStoreError(
            "Failed to store chunks in vector store"
        ) from e

    logger.info(
        f"Collection count: {collection.count()}"
    )


# -----------------------------
# Retrieve Chunks
# -----------------------------

def retrieve_chunks(
    query: str,
    document_id: str,
    top_k: int
) -> list[dict]:

    global embedding_model

    # Load embedding model only when needed
    if embedding_model is None:

        embedding_model = get_embedding_model()

    try:

        query_embedding = embedding_model.encode(
            [query]
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={
                "document_id": document_id
            }
        )

    except Exception as e:

        logger.error(
            "Failed retrieving chunks"
        )

        raise RetrievalError(
            "Failed retrieving chunks"
        ) from e

    logger.info(
        f"Query: {query}"
    )

    logger.info(
        f"Retrieval distances: {results['distances']}"
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for i in range(len(documents)):

        retrieved_chunks.append({
            "chunk_text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i]
        })

    return retrieved_chunks


# -----------------------------
# Delete Document
# -----------------------------

def delete_document(document_id: str):

    try:

        # Find all chunks belonging to this document
        results = collection.get(
            where={
                "document_id": document_id
            },
            include=["metadatas"]
        )

        # Document does not exist
        if not results["ids"]:

            raise DocumentDeleteError(
                f"Document not found: {document_id}"
            )

        # Get filepath from metadata
        filepath = results["metadatas"][0]["filepath"]

        # Delete all chunks from ChromaDB
        collection.delete(
            where={
                "document_id": document_id
            }
        )

        logger.info(
            f"Deleted ChromaDB chunks: {document_id}"
        )

        # Delete physical PDF file
        if os.path.exists(filepath):

            os.remove(filepath)

            logger.info(
                f"PDF file deleted: {filepath}"
            )

        else:

            logger.warning(
                f"PDF file not found on disk: {filepath}"
            )

        logger.info(
            f"Document deleted: {document_id}"
        )

    except DocumentDeleteError:

        raise

    except Exception as e:

        logger.error(
            f"Failed to delete document: {document_id}"
        )

        raise DocumentDeleteError(
            "Failed to delete document"
        ) from e


# -----------------------------
# Retrieve Documents
# -----------------------------

def retrieve_collections():

    try:

        results = collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in results["metadatas"]:

            document_id = metadata["document_id"]

            if document_id not in documents:

                documents[document_id] = {
                    "filename": metadata.get(
                        "filename",
                        "Unknown"
                    ),
                    "document_id": document_id
                }

        return list(documents.values())

    except Exception as e:

        logger.error(
            f"Failed to retrieve documents: {e}"
        )

        raise VectoreStoreError(
            "Failed to retrieve documents"
        ) from e