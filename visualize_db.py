import os
import sys

# Ensure current directory is in PYTHONPATH for imports
os.environ['PYTHONPATH'] = "."

from src.embeddings import load_config, get_chroma_client, get_embedding_model, get_collection, search_documents

def visualize_results():
    print("--- ChromaDB Visualization [Functional] ---")
    
    config = load_config()
    db_path = config['chromadb']['persist_directory']
    model_name = config['nlp']['embedding_model']
    collection_name = config['chromadb']['collection_name']

    client = get_chroma_client(db_path)
    collection = get_collection(client, collection_name, model_name)
    
    # 1. Show collection stats
    count = collection.count()
    print(f"Total documents indexed: {count}")
    
    # 2. Peek at some documents
    if count > 0:
        print("\nExtracting 5 entries from the database:")
        results = collection.peek(limit=5)
        
        for i in range(len(results['ids'])):
            print(f"\nID: {results['ids'][i]}")
            print(f"Text snippet: {results['documents'][i][:100]}...")
            print(f"Metadata: {results['metadatas'][i]}")
    
        # 3. Perform a semantic search test
        print("\n--- Semantic Search Test ---")
        query = "password reset issue"
        print(f"Query: '{query}'")
        
        search_results = search_documents(collection, query, n_results=3)
        
        for i in range(len(search_results['ids'][0])):
            print(f"\nResult {i+1} (Distance: {search_results['distances'][0][i]:.4f})")
            print(f"Document: {search_results['documents'][0][i]}")
    else:
        print("Database is empty. Please run src/main.py first.")

if __name__ == "__main__":
    visualize_results()
