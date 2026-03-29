
import pandas as pd
import numpy as np
import os
import sys

# Add root directory to python path for correct module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text
from src.embeddings import load_config, get_chroma_client, get_embedding_model, get_collection, generate_embeddings, index_documents
from src.model import train_ticket_classifier, save_model
 
def main():
    print("--- Starting NLP Pipeline: Steps 1, 2 & 3 [Functional] ---")
    
    # 1. Load Config
    config = load_config()
    data_path = config['data']['raw_path'] if 'data' in config else "data/raw/nlp_da.csv"
    db_path = config['chromadb']['persist_directory']
    model_name = config['nlp']['embedding_model']
    collection_name = config['chromadb']['collection_name']
    model_save_path = config['training'].get('model_save_path', "data/processed/ticket_model.joblib")

    # Ensure paths are correct relative to project root
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', data_path)
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        # Try a direct relative path as fallback
        data_path = "data/raw/nlp_da.csv"
        if not os.path.exists(data_path):
             # Try absolute path relative to current working directory
            data_path = os.path.abspath("data/raw/nlp_da.csv")
            if not os.path.exists(data_path):
                print(f"CRITICAL: Data file not found even at {data_path}")
                return

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 2. Preprocessing (Step 1)
    print("Step 1: Cleaning text...")
    df['type'] = df['type'].fillna('Unknown')
    df['combined_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Save Step 1 output
    processed_path = "data/processed/cleaned_tickets.csv"
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Preprocessed data saved to {processed_path}")

    # 3. Embeddings & Indexing (Step 2)
    print("Step 2: Embeddings & Vector DB Indexing...")
    print("Initializing ChromaDB and Embedding Model...")
    client = get_chroma_client(db_path)
    embed_model = get_embedding_model(model_name)
    
    # Reset collection for clean slate
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = get_collection(client, collection_name, model_name)

    ids = [str(i) for i in range(len(df))]
    texts = df['cleaned_text'].tolist()
    metadatas = df[['type', 'queue']].fillna('Unknown').to_dict('records')
    
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = generate_embeddings(embed_model, texts)
    
    print("Indexing in ChromaDB...")
    index_documents(collection, ids, texts, embeddings, metadatas)
    print(f"✅ Total documents indexed in collection '{collection_name}': {collection.count()}")

    # 4. Training & Evaluation (Step 3)
    print("Step 3: Training & Evaluation...")
    print("Preparing features for training...")
    X = np.array(embeddings)
    y = df['type'].astype(object).values
    
    print(f"Training model with {len(X)} samples...")
    classifier_model, X_test, y_test = train_ticket_classifier(X, y)
    
    save_model(classifier_model, model_save_path)
    
    print("--- Full Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
