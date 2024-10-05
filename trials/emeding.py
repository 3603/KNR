from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={
        "trust_remote_code": True,
        "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
    }
)

# Example usage
text = "This is a sample text."
embedding_vector = embeddings.embed(text)
print(embedding_vector)
