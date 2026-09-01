"""
analysis.py - Semantic Analysis of Word Embeddings
====================================================
Performs semantic analysis on trained Word2Vec models:
1. Top-5 nearest neighbors for specified words
2. Analogy experiments (e.g., UG : BTech :: PG : ?)
3. Comparison between CBOW and Skip-gram models

Usage:
    python embeddings/analysis.py

Output:
    Prints formatted tables to stdout
    Saves results to embeddings/analysis_results.json
"""

import os
import json
from gensim.models import Word2Vec

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_FILE = os.path.join(BASE_DIR, "analysis_results.json")

# Words for nearest neighbor analysis
QUERY_WORDS = ["research", "student", "phd", "examination", "engineering"]

# Analogy experiments: (positive1, negative1, positive2) → expected analogy
# Format: word_a is to word_b as word_c is to ?
# Computed as: word_b - word_a + word_c
ANALOGIES = [
    {
        "description": "UG : B.Tech :: PG : ?",
        "positive": ["b.tech", "pg"],
        "negative": ["ug"],
        "expected": "mtech / m.tech",
    },
    {
        "description": "professor : teaching :: researcher : ?",
        "positive": ["teaching", "researcher"],
        "negative": ["professor"],
        "expected": "publication/paper",
    },
    {
        "description": "student : learning :: faculty : ?",
        "positive": ["learning", "faculty"],
        "negative": ["student"],
        "expected": "teaching",
    },
    {
        "description": "semester : examination :: thesis : ?",
        "positive": ["examination", "thesis"],
        "negative": ["semester"],
        "expected": "defense/evaluation",
    },
    {
        "description": "phd : thesis :: mtech : ?",
        "positive": ["thesis", "mtech"],
        "negative": ["phd"],
        "expected": "project/dissertation",
    },
]


def load_models():
    """Load best CBOW and Skip-gram models."""
    models = {}
    
    cbow_path = os.path.join(MODELS_DIR, "best_cbow.model")
    if os.path.exists(cbow_path):
        models["CBOW"] = Word2Vec.load(cbow_path)
        print(f"  ✓ Loaded CBOW model (vocab: {len(models['CBOW'].wv)})")
    else:
        print(f"  ✗ CBOW model not found at {cbow_path}")
    
    sg_path = os.path.join(MODELS_DIR, "best_skipgram.model")
    if os.path.exists(sg_path):
        models["SkipGram"] = Word2Vec.load(sg_path)
        print(f"  ✓ Loaded Skip-gram model (vocab: {len(models['SkipGram'].wv)})")
    else:
        print(f"  ✗ Skip-gram model not found at {sg_path}")
    
    return models


def nearest_neighbors(models, query_words):
    """Find top-5 nearest neighbors for each query word in each model."""
    print("\n" + "=" * 70)
    print("TASK 3.1: TOP-5 NEAREST NEIGHBORS (Cosine Similarity)")
    print("=" * 70)
    
    results = {}
    
    for word in query_words:
        print(f"\n  Query word: '{word}'")
        print(f"  {'─' * 60}")
        results[word] = {}
        
        for model_name, model in models.items():
            if word not in model.wv:
                print(f"  [{model_name}] Word '{word}' not in vocabulary!")
                results[word][model_name] = "NOT_IN_VOCAB"
                continue
            
            neighbors = model.wv.most_similar(word, topn=5)
            results[word][model_name] = [
                {"word": w, "similarity": round(s, 4)} for w, s in neighbors
            ]
            
            print(f"  [{model_name:>10}]  ", end="")
            for w, s in neighbors:
                print(f"{w}({s:.3f})  ", end="")
            print()
    
    return results


def analogy_experiments(models, analogies):
    """Perform analogy experiments."""
    print("\n" + "=" * 70)
    print("TASK 3.2: ANALOGY EXPERIMENTS")
    print("=" * 70)
    
    results = []
    
    for analogy in analogies:
        desc = analogy["description"]
        positive = analogy["positive"]
        negative = analogy["negative"]
        expected = analogy["expected"]
        
        print(f"\n  Analogy: {desc}")
        print(f"  Expected answer: {expected}")
        print(f"  {'─' * 55}")
        
        analogy_result = {"description": desc, "expected": expected, "results": {}}
        
        for model_name, model in models.items():
            # Check if all words are in vocabulary
            all_words = positive + negative
            missing = [w for w in all_words if w not in model.wv]
            
            if missing:
                print(f"  [{model_name:>10}]  Missing words: {missing}")
                analogy_result["results"][model_name] = f"Missing: {missing}"
                continue
            
            try:
                result = model.wv.most_similar(
                    positive=positive,
                    negative=negative,
                    topn=5
                )
                analogy_result["results"][model_name] = [
                    {"word": w, "similarity": round(s, 4)} for w, s in result
                ]
                
                print(f"  [{model_name:>10}]  Top-5: ", end="")
                for w, s in result:
                    print(f"{w}({s:.3f})  ", end="")
                print()
                
                # Check if expected answer is in top 5
                top_words = [w for w, _ in result]
                if expected in top_words:
                    print(f"  {'':>13}  ✓ Expected '{expected}' found at rank {top_words.index(expected) + 1}")
                
            except Exception as e:
                print(f"  [{model_name:>10}]  Error: {e}")
                analogy_result["results"][model_name] = f"Error: {str(e)}"
        
        results.append(analogy_result)
    
    return results


def semantic_discussion(neighbor_results, analogy_results):
    """Print a discussion of the semantic analysis."""
    print("\n" + "=" * 70)
    print("DISCUSSION")
    print("=" * 70)
    
    print("""
  NEAREST NEIGHBORS ANALYSIS:
  ─────────────────────────────
  The nearest neighbors reveal the semantic associations learned from
  the IIT Jodhpur corpus. Words like 'research' are expected to cluster
  with academic terms (publications, projects, innovation), while
  'student' should associate with academic life (courses, hostel, clubs).
  
  CBOW vs Skip-gram tend to capture different relationships:
  - CBOW: Better at capturing syntactic patterns and common collocations
  - Skip-gram: Better at capturing semantic relationships, especially for
    rare words, since it predicts context from center word
  
  ANALOGY EXPERIMENTS:
  ─────────────────────
  Analogy tasks test whether the embedding space captures meaningful
  relational structure. The 'UG:BTech :: PG:?' analogy tests whether
  the model understands degree-level relationships.
  
  Note: With a domain-specific corpus (IIT Jodhpur), analogies work best
  for domain-relevant terms. General-purpose analogies may not transfer
  well since the model has only seen academic/institutional text.
    """)


def main():
    print("=" * 70)
    print("SEMANTIC ANALYSIS OF WORD2VEC MODELS")
    print("=" * 70)
    
    # Load models
    print("\nLoading trained models...")
    models = load_models()
    
    if not models:
        print("\nERROR: No trained models found. Run train_word2vec.py first!")
        return
    
    # Task 3.1: Nearest neighbors
    neighbor_results = nearest_neighbors(models, QUERY_WORDS)
    
    # Task 3.2: Analogies
    analogy_results = analogy_experiments(models, ANALOGIES)
    
    # Discussion
    semantic_discussion(neighbor_results, analogy_results)
    
    # Save all results
    all_results = {
        "nearest_neighbors": neighbor_results,
        "analogies": analogy_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved to: {RESULTS_FILE}")
    
    print("\n" + "=" * 70)
    print("SEMANTIC ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
