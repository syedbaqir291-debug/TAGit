"""
setup.py — Run ONCE before launching the app.

This script:
1. Downloads the sentence-transformer model (~80 MB, one time)
2. Embeds all 270+ JCI MEs into a vector index
3. Saves the index to mes_index.pkl so the app loads instantly

Usage:
    python setup.py

Takes approximately 30–60 seconds on first run.
Subsequent app launches use the cached index (instant).
"""

import json
import pickle
import time
from pathlib import Path

DB_PATH    = Path("mes_database.json")
INDEX_PATH = Path("mes_index.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("=" * 60)
    print("  JCI Observation Tagger — Setup")
    print("=" * 60)

    # Check database
    if not DB_PATH.exists():
        print(f"\n❌  {DB_PATH} not found. Make sure it's in this directory.")
        return

    with open(DB_PATH) as f:
        mes = json.load(f)
    print(f"\n✅  Loaded {len(mes)} measurable elements from {DB_PATH}")

    # Load model
    print(f"\n📥  Loading sentence-transformer model: {MODEL_NAME}")
    print("    (Downloads ~80 MB on first run, cached afterwards)")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    # Build embeddings
    print(f"\n⚙️   Embedding {len(mes)} MEs…")
    texts = [
        f"{m['standard']} {m['standard_title']} {m['text']}"
        for m in mes
    ]
    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    print(f"    Done in {time.time()-t0:.1f}s — shape: {embeddings.shape}")

    # Save
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"\n✅  Index saved to {INDEX_PATH}")

    # Quick smoke test
    print("\n🔬  Smoke test…")
    query = "crash cart not checked weekly"
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ q_vec
    import numpy as np
    top3 = np.argsort(scores)[::-1][:3]
    print(f"    Query: '{query}'")
    for i in top3:
        print(f"    → {mes[i]['id']:30s}  score={scores[i]:.3f}  {mes[i]['text'][:70]}…")

    print("\n✅  Setup complete. Launch the app with:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
