# src/ingest.py
import os
import pandas as pd
import faiss
import pickle
from embeddings import Embeddings

INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/meta.pkl"


def chunk_text(text, max_words=120):
    """
    Splits long text into chunks of <= max_words.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks


def build_index(csv_path="data/faqs.csv", embed_backend="hf", openai_key=None):
    """
    Build a FAISS index from a CSV containing medical FAQs.
    CSV must have columns: 'question', 'answer'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found! Please provide a CSV with FAQs.")

    # Load CSV with flexible parser
    # df = pd.read_csv(csv_path, sep=None, engine='python', quoting=3)

    # Load CSV properly
    df = pd.read_csv(
        csv_path
    )

    # Strip column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    # if 'question' not in df.columns or 'answer' not in df.columns:
    #     raise ValueError("CSV must have 'question' and 'answer' columns")
    print("dimention of df : " , df.shape)
    print("columns names : " , df.columns)
    # Drop empty rows
    df = df[['question', 'answer']].dropna().reset_index(drop=True)

    print("Loaded rows:", len(df))
    print(df.head(5))

    sample_df = df.head(100)
    records = []

    # Combine question & answer, split into chunks
    for _, row in sample_df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if not q and not a:
            continue
        combined = f"Q: {q}\nA: {a}"
        chunks = chunk_text(combined, max_words=120)
        for c in chunks:
            records.append({"text": c, "source_question": q, "source_answer": a})

    # Embed the text
    texts = [r["text"] for r in records]
    embedder = Embeddings(backend=embed_backend, openai_api_key=openai_key)
    embeddings = embedder.encode(texts).astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product = cosine similarity
    index.add(embeddings)

    # Save index and metadata
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(records, f)

    print(f"✅ Index built: {len(records)} chunks, dim={d}")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/faqs.csv", help="Path to CSV with FAQs")
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf", help="Embeddings backend")
    parser.add_argument("--openai_key", default=None, help="OpenAI API key if using openai embeddings")
    args = parser.parse_args()

    build_index(csv_path=args.csv, embed_backend=args.backend, openai_key=args.openai_key)
