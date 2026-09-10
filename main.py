from fastapi import FastAPI,File,UploadFile,HTTPException
from pydantic import BaseModel
from vectore_store import delete_document,DocumentDeleteError,retrieve_collections
from pipeline import ask,ingest
import uuid

app = FastAPI()

@app.get('/')
def home():
    return {'message':'RAG API is running'}

class QuestionRequest(BaseModel):
    question:str
    document_id:str


class QuestionResponse(BaseModel):
    question: str
    response: str

@app.post('/ask',response_model=QuestionResponse)
def ask_question(request:QuestionRequest):
    answer = ask(request.question,request.document_id)
    return {"question":request.question,
            "response":answer}

class UploadResponse(BaseModel):
    filename:str
    fileSizeMB:float
    message:str
    document_id:str

@app.post('/upload',response_model=UploadResponse)
async def upload_file(file:UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    contents = await file.read()
    filepath = f'uploads/{file.filename}'
    document_id = str(uuid.uuid4())
    try:
        with open(filepath,'wb') as f:
            f.write(contents)

        ingest(filepath,document_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"error while processing pdf {str(e)}"
        )
    return{
        "filename":file.filename,
        "fileSizeMB": round(len(contents) / 1_000_000, 2),
        "message":'PDF uploaded and processed successfully',
        "document_id":document_id
    }


@app.delete('/document')
def delete_document_api(document_id: str):
    try:
        delete_document(document_id)
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except DocumentDeleteError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

@app.get('/documents')
def get_documents():
    try:
        documents = retrieve_collections()

        return {
            "status": True,
            "documents": documents
        }

    except Exception as e:
        return {
            "status": False,
            "message": str(e)
        }
