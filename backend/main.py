from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
import os
import hashlib
from typing import List
import tempfile

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chroma setup
client = chromadb.PersistentClient(path="./chroma_db")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
collection = client.get_or_create_collection(name="legal_docs", embedding_function=ef)

class QueryRequest(BaseModel):
    text: str
    k: int = 3

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process PDF
        text = ""
        with fitz.open(temp_path) as doc:
            text = " ".join([page.get_text() for page in doc])
        
        os.unlink(temp_path)

        # Create document ID from content hash
        doc_id = hashlib.md5(content).hexdigest()

        # Add to Chroma
        collection.add(
            documents=[text],
            metadatas=[{"source": file.filename}],
            ids=[doc_id]
        )

        return {"message": "File processed successfully", "id": doc_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        results = collection.query(
            query_texts=[request.text],
            n_results=request.k
        )
        return {
            "documents": results["documents"][0],
            "sources": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    