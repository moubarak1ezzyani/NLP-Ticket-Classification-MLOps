import pandas as pd
import os
from src.preprocessing import clean_text

def main():
    print("--- Step 1: EDA and NLP Preprocessing ---")
    
    # Load Data
    data_path = "data/raw/nlp_da.csv"

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocessing (Step 1)
    print("Cleaning text...")
    df['type'] = df['type'].fillna('Unknown')
    df['combined_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Save preprocessed data
    output_path = "data/processed/cleaned_tickets.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Preprocessed data saved to {output_path}")
    print(f"Total tickets processed: {len(df)}")
    print("--- Step 1 Completed Successfully ---")

if __name__ == "__main__":
    main()
