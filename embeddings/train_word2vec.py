"""
train_word2vec.py - Word2Vec Model Training
=============================================
Trains CBOW and Skip-gram Word2Vec models on the IIT Jodhpur corpus
with hyperparameter experiments.

Experiments:
- Model type: CBOW (sg=0) vs Skip-gram (sg=1)
- Embedding dimensions: 50, 100, 200
- Context window sizes: 3, 5, 7
- Negative samples: 5, 10

Usage:
    python embeddings/train_word2vec.py

Output:
    embeddings/models/             — saved Word2Vec models
    embeddings/experiment_results.csv — hyperparameter experiment table
"""

import os
import csv
import time
import itertools

from gensim.models import Word2Vec

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_FILE = os.path.join(BASE_DIR, "experiment_results.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameter grid
EMBEDDING_DIMS = [50, 100, 200]
WINDOW_SIZES = [3, 5, 7]
NEGATIVE_SAMPLES = [5, 10]
MODEL_TYPES = [
    {"name": "CBOW", "sg": 0},
    {"name": "SkipGram", "sg": 1},
]

# Fixed params
MIN_COUNT = 2     # Minimum word frequency
EPOCHS = 20       # Training epochs
WORKERS = 4       # Parallel workers

# Evaluation words for quick quality check
EVAL_WORDS = ["research", "student", "engineering", "professor", "department"]


def load_corpus(filepath):
    """Load tokenized corpus — one sentence per line, space-separated tokens."""
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                sentences.append(tokens)
    return sentences


def evaluate_model(model, eval_words):
    """Quick evaluation: average similarity of top-3 neighbors for eval words."""
    total_sim = 0
    count = 0
    for word in eval_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=3)
            for _, sim in neighbors:
                total_sim += sim
                count += 1
    return round(total_sim / max(count, 1), 4)


def train_and_evaluate():
    """Run all hyperparameter experiments."""
    print("=" * 70)
    print("WORD2VEC TRAINING — HYPERPARAMETER EXPERIMENTS")
    print("=" * 70)
    
    # Load corpus
    print(f"\nLoading corpus from: {CORPUS_FILE}")
    sentences = load_corpus(CORPUS_FILE)
    print(f"Loaded {len(sentences):,} sentences")
    
    if not sentences:
        print("ERROR: No sentences found. Run preprocess.py first!")
        return
    
    # All experiment configurations
    configs = list(itertools.product(MODEL_TYPES, EMBEDDING_DIMS, WINDOW_SIZES, NEGATIVE_SAMPLES))
    print(f"Total experiments: {len(configs)}")
    print()
    
    results = []
    best_cbow = {"avg_sim": -1, "model": None, "config": None}
    best_sg = {"avg_sim": -1, "model": None, "config": None}
    
    for i, (model_type, dim, window, neg) in enumerate(configs, 1):
        config_name = f"{model_type['name']}_d{dim}_w{window}_n{neg}"
        print(f"[{i:2d}/{len(configs)}] Training {config_name}...", end=" ", flush=True)
        
        start_time = time.time()
        
        model = Word2Vec(
            sentences=sentences,
            vector_size=dim,
            window=window,
            min_count=MIN_COUNT,
            sg=model_type["sg"],
            negative=neg,
            epochs=EPOCHS,
            workers=WORKERS,
            seed=42,
        )
        
        elapsed = time.time() - start_time
        avg_sim = evaluate_model(model, EVAL_WORDS)
        vocab_size = len(model.wv)
        
        print(f"done in {elapsed:.1f}s | vocab={vocab_size} | avg_sim={avg_sim:.4f}")
        
        result = {
            "model_type": model_type["name"],
            "embedding_dim": dim,
            "window_size": window,
            "negative_samples": neg,
            "vocab_size": vocab_size,
            "training_time_sec": round(elapsed, 2),
            "avg_similarity": avg_sim,
        }
        results.append(result)
        
        # Track best models
        if model_type["name"] == "CBOW" and avg_sim > best_cbow["avg_sim"]:
            best_cbow = {"avg_sim": avg_sim, "model": model, "config": config_name}
        elif model_type["name"] == "SkipGram" and avg_sim > best_sg["avg_sim"]:
            best_sg = {"avg_sim": avg_sim, "model": model, "config": config_name}
    
    # Save best models
    print("\n" + "=" * 70)
    print("SAVING BEST MODELS")
    print("=" * 70)
    
    if best_cbow["model"]:
        path = os.path.join(MODELS_DIR, "best_cbow.model")
        best_cbow["model"].save(path)
        print(f"  ✓ Best CBOW:     {best_cbow['config']} (avg_sim={best_cbow['avg_sim']:.4f})")
        print(f"    Saved to: {path}")
    
    if best_sg["model"]:
        path = os.path.join(MODELS_DIR, "best_skipgram.model")
        best_sg["model"].save(path)
        print(f"  ✓ Best SkipGram: {best_sg['config']} (avg_sim={best_sg['avg_sim']:.4f})")
        print(f"    Saved to: {path}")
    
    # Save experiment results to CSV
    print(f"\n  ✓ Saving experiment results to: {RESULTS_FILE}")
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Print results table
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS TABLE")
    print("=" * 70)
    print(f"{'Model':<10} {'Dim':<5} {'Win':<5} {'Neg':<5} {'Vocab':<8} {'Time(s)':<9} {'AvgSim':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r['model_type']:<10} {r['embedding_dim']:<5} {r['window_size']:<5} "
              f"{r['negative_samples']:<5} {r['vocab_size']:<8} {r['training_time_sec']:<9} "
              f"{r['avg_similarity']:<8}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
