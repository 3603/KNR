from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import fitz  # PyMuPDF

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

class QueryRequest(BaseModel):
    query: str

@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    global vectorstore
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Extract text from PDF
        text = extract_text_from_pdf(tmp_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from document")

        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create/update vector store
        if vectorstore is None:
            vectorstore = FAISS.from_texts(chunks, embeddings)
        else:
            vectorstore.add_texts(chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "sample": chunks[0][:200] + "..." if chunks else ""
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
    return text

@app.post("/query")
async def query_documents(request: QueryRequest):
    global vectorstore
    
    if not vectorstore:
        raise HTTPException(status_code=400, detail="No documents processed yet")
    
    try:
        docs = vectorstore.similarity_search(request.query, k=3)
        return {
            "results": [doc.page_content for doc in docs],
            "sources": [str(doc.metadata) for doc in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)