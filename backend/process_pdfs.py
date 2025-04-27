import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Use the directory of the current script as the base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # Assumes a 'data' subfolder 
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def process_pdfs():
    print("Initializing Chroma DB...")
    # Ensure the data and chroma_db directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="legal_docs",
        embedding_function=embedding_func
    )
    print(f"Processing PDFs in {DATA_DIR}...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
   
    for pdf_file in tqdm(pdf_files, desc="Processing"):
        try:
            pdf_path = os.path.join(DATA_DIR, pdf_file)
            with fitz.open(pdf_path) as doc:
                text = " ".join([page.get_text() for page in doc])
               
                collection.add(
                    documents=[text],
                    metadatas=[{"source": pdf_file}],
                    ids=[pdf_file]
                )
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    print(f"\n✅ Processed {len(pdf_files)} PDFs. Vector DB ready at {CHROMA_PATH}")

if __name__ == "__main__":
    process_pdfs()