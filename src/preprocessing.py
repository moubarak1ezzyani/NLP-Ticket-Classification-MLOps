import spacy
import re

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    print("Downloading SpaCy model 'en_core_web_sm'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Normalization (lowercase)
    text = text.lower()
    
    # 2. Punctuation and special character removal
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 3. Tokenization & Stopwords removal - SpaCy
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.text.strip()]
    
    return " ".join(tokens)
