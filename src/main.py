import pandas as pd
from src.preprocessing import clean_text
from src.embeddings import EmbeddingManager
from src.model import TicketClassifier
import numpy as np
import os

def main():
    print("--- Starting Full NLP Ticket Classification Pipeline (Steps 1, 2, 3) ---")
    
    # 1. Load Data
    data_path = "data/raw/nlp_da.csv"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path).head(200) # 200 for better split
    
    # 2. Preprocessing (Step 1)
    print("Cleaning text...")
    df['type'] = df['type'].fillna('Unknown')
    df['combined_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # 3. Embeddings & Indexing (Step 2)
    print("Generating embeddings and indexing in ChromaDB...")
    manager = EmbeddingManager(db_path="./data/chromadb")
    
    try:
        manager.client.delete_collection("ticket_embeddings")
        manager.collection = manager.client.get_or_create_collection(
            name="ticket_embeddings", 
            embedding_function=manager.embedding_function
        )
    except:
        pass

    ids = [str(i) for i in range(len(df))]
    texts = df['cleaned_text'].tolist()
    metadatas = df[['type', 'queue']].fillna('Unknown').to_dict('records')
    
    embeddings = manager.generate_embeddings(texts)
    manager.collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    # 4. Training & Evaluation (Step 3)
    X = np.array(embeddings)
    y = df['type'].astype(object).values
    
    classifier = TicketClassifier()
    classifier.train(X, y)
    classifier.save_model("data/processed/ticket_model.joblib")
    
    print("--- Full Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
