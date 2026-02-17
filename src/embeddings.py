import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import yaml
import os

def load_config():
    """Load config from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_chroma_client(db_path):
    """Initialize ChromaDB client"""
    return chromadb.PersistentClient(path=db_path)

def get_embedding_model(model_name):
    """Load SentenceTransformer model"""
    return SentenceTransformer(model_name)

def get_collection(client, collection_name, model_name):
    """Get or create ChromaDB collection with correct embedding function"""
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    return client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_function
    )

def generate_embeddings(model, texts):
    """Transform texts into semantic vectors using the provided model"""
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    return embeddings.tolist()

def index_documents(collection, ids, texts, embeddings, metadatas=None):
    """Add docs & embeddings to the collection"""
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ Indexed {len(ids)} documents in ChromaDB.")

def search_documents(collection, query_text, n_results=5):
    """Query the collection for similar documents"""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    # Quick Test
    config = load_config()
    db_path = config['chromadb']['persist_directory']
    model_name = config['nlp']['embedding_model']
    collection_name = config['chromadb']['collection_name']

    client = get_chroma_client(db_path)
    model = get_embedding_model(model_name)
    collection = get_collection(client, collection_name, model_name)
    
    print(f"Collection count: {collection.count()}")
