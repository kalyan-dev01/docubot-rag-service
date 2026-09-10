# DocuBot RAG Engine

DocuBot is a document-based AI chatbot project that can read PDF documents and answer questions based on their content.

I built this RAG engine as the core AI part of my larger **DocuBot** project. The idea is to eventually turn it into a service where businesses can upload their documents and create an AI chatbot for their website.

## How it works

The basic flow is:

```text
PDF
 ↓
Extract text
 ↓
Split text into chunks
 ↓
Create embeddings
 ↓
Store in ChromaDB
 ↓
Retrieve relevant chunks
 ↓
Rerank the results
 ↓
Send the best context to the LLM
 ↓
Generate the answer
```

The important part is that the LLM doesn't directly read the whole PDF. The system first finds the most relevant parts of the document and gives those parts to the LLM as context.

## What I used

* **Python** – Main programming language
* **FastAPI** – API for the RAG service
* **PyMuPDF** – Extract text from PDFs
* **Sentence Transformers** – Generate embeddings
* **ChromaDB** – Store and search document embeddings
* **CrossEncoder** – Rerank retrieved chunks
* **Hugging Face Transformers** – Tokenization and chunking
* **Qwen** – LLM for generating answers
* **python-dotenv** – Environment variable management

## Project Structure

```text
docubot-rag/
│
├── config.py              # Configuration
├── pdf_processing.py      # PDF extraction and cleaning
├── chunking.py            # Text chunking
├── embedding.py           # Generate embeddings
├── vectore_store.py       # ChromaDB operations
├── reranking.py           # Rerank retrieved chunks
├── generation.py          # Prompt building and LLM calls
├── pipeline.py            # Complete RAG pipeline
├── main.py                # FastAPI application
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Setup

Clone the repository:

```bash
git clone <your-repository-url>
cd docubot-rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project folder:

```env
LLM_API_KEY=your_api_key
LLM_URL=your_llm_url
LLM_MODEL_NAME=your_model_name
```

Keep the `.env` file private and don't upload it to GitHub.

## Run the project

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

## API

### Upload a PDF

```http
POST /upload
```

Upload a PDF and the system will process it, create embeddings, and store the chunks in ChromaDB.

The response contains a `document_id`.

### Ask a question

```http
POST /ask
```

Example:

```json
{
    "question": "What is the leave policy?",
    "document_id": "your-document-id"
}
```

The system retrieves relevant chunks from that specific document, reranks them, and sends the best chunks to the LLM.

### View documents

```http
GET /documents
```

Returns the documents currently stored in the system.

### Delete a document

```http
DELETE /document?document_id=your-document-id
```

This removes the document from the vector database and deletes its uploaded PDF.

## Retrieval

I'm using a two-step retrieval approach.

First, ChromaDB performs semantic search and retrieves the top 15 relevant chunks.

Then, a CrossEncoder looks at the query and each retrieved chunk together and scores how relevant they are.

Finally, the top 3 chunks are passed to the LLM.

```text
Question
   ↓
Embedding
   ↓
ChromaDB
   ↓
Top 15 chunks
   ↓
CrossEncoder
   ↓
Top 3 chunks
   ↓
LLM
   ↓
Answer
```

## Grounded Answers

The LLM is instructed to use only the information retrieved from the document.

If the available context doesn't contain enough information, it should respond:

```text
I don't know based on the provided context.
```

The generated answers can also include page numbers so that the information can be traced back to the PDF.

## Current Status

The main RAG pipeline is working.

Currently completed:

* PDF text extraction
* PDF type detection
* Text cleaning
* Token-based chunking
* Embedding generation
* ChromaDB storage
* Semantic retrieval
* CrossEncoder reranking
* Context building
* LLM generation
* FastAPI API
* PDF upload
* Document listing
* Document deletion
* Environment configuration
* Basic logging and error handling

## What's Next

This RAG engine is only one part of the bigger DocuBot project.

The next stage is to build the actual SaaS application around it.

The planned architecture is:

```text
React Frontend
      ↓
Node.js / Express Backend
      ↓
PostgreSQL
      ↓
Python RAG Service
      ↓
ChromaDB + Embeddings + Reranker + LLM
```

The final goal is to allow businesses to:

1. Create an account
2. Create a chatbot
3. Upload their documents
4. Customize their chatbot
5. Generate an embed code
6. Add the chatbot to their website
7. Let their customers ask questions about their business

## Project Status

🚧 **Currently under development**

This project started as a way for me to learn and build a complete RAG pipeline, and it will eventually become the AI backend for the larger DocuBot platform.
