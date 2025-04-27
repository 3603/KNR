from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import os

# Initialize FastAPI
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Path Configuration
current_dir = Path(__file__).parent.absolute()
CHROMA_PATH = current_dir / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name="legal_docs",
        embedding_function=embedding_func
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")

# Request Models
class QueryRequest(BaseModel):
    text: str
    k: int = 3  # Default to 3 results

# API Endpoints
@app.get("/")
async def health_check():
    return {"status": "healthy", "chroma_path": str(CHROMA_PATH)}

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the vector database for similar documents
    Example request body:
    {
        "text": "what is the statute of limitations?",
        "k": 3
    }
    """
    try:
        results = collection.query(
            query_texts=[request.text],
            n_results=request.k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": float(results["distances"][0][i])
            })
            
        return {"results": formatted_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection-info")
async def get_collection_info():
    """Get information about the Chroma collection"""
    try:
        return {
            "collection_name": collection.name,
            "document_count": collection.count(),
            "embedding_model": EMBEDDING_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)