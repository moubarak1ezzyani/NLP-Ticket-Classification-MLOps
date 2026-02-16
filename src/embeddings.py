import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import yaml
import os

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class EmbeddingManager:
    def __init__(self, config=None, db_path=None):
        """
        Initialize the embedding model and ChromaDB client using config.
        """
        self.config = config if config else load_config()
        
        # Paths and Names
        persist_dir = db_path if db_path else self.config['chromadb']['persist_directory']
        collection_name = self.config['chromadb']['collection_name']
        model_name = self.config['nlp']['embedding_model']
        
        # Initialize Model
        self.model = SentenceTransformer(model_name)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
        )

    def generate_embeddings(self, texts):
        """
        Transform texts into semantic vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
        return embeddings.tolist()

    def index_documents(self, ids, texts, metadatas=None):
        """
        Index documents in ChromaDB.
        """
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f"✅ Indexed {len(ids)} documents in ChromaDB.")

    def search(self, query_text, n_results=5):
        """
        Search for similar documents.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    # Test
    manager = EmbeddingManager()
    print(f"Collection count: {manager.collection.count()}")
