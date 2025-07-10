from langchain_huggingface import HuggingFaceEmbeddings
"""Get embeddings from HuggingFace."""

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

