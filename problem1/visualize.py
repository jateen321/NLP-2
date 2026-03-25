"""
visualize.py - Word Embedding Visualization
=============================================
Visualizes word embeddings using PCA and t-SNE:
1. Projects selected words into 2D space
2. Creates side-by-side CBOW vs Skip-gram plots
3. Color-codes words by semantic clusters

Usage:
    python problem1/visualize.py

Output:
    problem1/figures/pca_comparison.png
    problem1/figures/tsne_comparison.png
    problem1/figures/pca_cbow.png
    problem1/figures/pca_skipgram.png
    problem1/figures/tsne_cbow.png
    problem1/figures/tsne_skipgram.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Words to visualize, organized by semantic clusters
WORD_CLUSTERS = {
    "Academic Programs": [
        "btech", "mtech", "phd", "msc", "mba",
        "undergraduate", "postgraduate", "degree",
        "semester", "credits", "curriculum",
    ],
    "Departments": [
        "computer", "electrical", "mechanical", "civil",
        "chemistry", "physics", "mathematics", "engineering",
        "bioscience", "metallurgical",
    ],
    "Research": [
        "research", "publication", "journal", "paper", "conference",
        "innovation", "project", "thesis", "dissertation",
        "laboratory", "experiment",
    ],
    "People": [
        "student", "faculty", "professor", "director", "head",
        "scholar", "candidate", "researcher", "staff",
    ],
    "Academic Activities": [
        "exam", "examination", "grade", "assessment",
        "course", "lecture", "workshop", "seminar",
        "admission", "registration",
    ],
}

# Color palette for clusters
CLUSTER_COLORS = {
    "Academic Programs": "#e74c3c",    # Red
    "Departments": "#3498db",          # Blue
    "Research": "#2ecc71",             # Green
    "People": "#f39c12",              # Orange
    "Academic Activities": "#9b59b6",  # Purple
}


def load_models():
    """Load best CBOW and Skip-gram models."""
    models = {}
    for name, filename in [("CBOW", "best_cbow.model"), ("Skip-gram", "best_skipgram.model")]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = Word2Vec.load(path)
            print(f"  ✓ Loaded {name} model")
    return models


def get_embeddings(model, word_clusters):
    """Extract embeddings for words that exist in the model's vocabulary."""
    words = []
    vectors = []
    colors = []
    cluster_labels = []
    
    for cluster_name, cluster_words in word_clusters.items():
        for word in cluster_words:
            if word in model.wv:
                words.append(word)
                vectors.append(model.wv[word])
                colors.append(CLUSTER_COLORS[cluster_name])
                cluster_labels.append(cluster_name)
    
    return words, np.array(vectors), colors, cluster_labels


def plot_embeddings(coords, words, colors, cluster_labels, title, output_path):
    """Create a scatter plot of word embeddings."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points by cluster for legend
    plotted_clusters = set()
    for i, (x, y) in enumerate(coords):
        cluster = cluster_labels[i]
        label = cluster if cluster not in plotted_clusters else None
        plotted_clusters.add(cluster)
        ax.scatter(x, y, c=colors[i], s=80, alpha=0.7, edgecolors="white",
                   linewidth=0.5, label=label, zorder=3)
    
    # Add word labels
    for i, word in enumerate(words):
        ax.annotate(
            word, (coords[i, 0], coords[i, 1]),
            fontsize=8, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.7),
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_comparison(coords_dict, words_dict, colors_dict, labels_dict,
                    method_name, output_path):
    """Create side-by-side comparison plot for CBOW vs Skip-gram."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    for idx, (model_name, coords) in enumerate(coords_dict.items()):
        ax = axes[idx]
        words = words_dict[model_name]
        colors = colors_dict[model_name]
        cluster_labels = labels_dict[model_name]
        
        plotted_clusters = set()
        for i, (x, y) in enumerate(coords):
            cluster = cluster_labels[i]
            label = cluster if cluster not in plotted_clusters else None
            plotted_clusters.add(cluster)
            ax.scatter(x, y, c=colors[i], s=70, alpha=0.7,
                       edgecolors="white", linewidth=0.5, label=label, zorder=3)
        
        for i, word in enumerate(words):
            ax.annotate(
                word, (coords[i, 0], coords[i, 1]),
                fontsize=7, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="gray", alpha=0.6),
            )
        
        ax.set_title(f"{model_name} — {method_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Component 1", fontsize=10)
        ax.set_ylabel("Component 2", fontsize=10)
    
    plt.suptitle(f"Word Embedding Visualization: {method_name} Projection",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved comparison: {output_path}")


def main():
    print("=" * 70)
    print("WORD EMBEDDING VISUALIZATION")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    models = load_models()
    
    if not models:
        print("\nERROR: No models found. Run train_word2vec.py first!")
        return
    
    # Extract embeddings for each model
    all_words = {}
    all_vectors = {}
    all_colors = {}
    all_labels = {}
    
    for model_name, model in models.items():
        print(f"\n[{model_name}] Extracting embeddings...")
        words, vectors, colors, labels = get_embeddings(model, WORD_CLUSTERS)
        all_words[model_name] = words
        all_vectors[model_name] = vectors
        all_colors[model_name] = colors
        all_labels[model_name] = labels
        print(f"  Found {len(words)} words in vocabulary")
    
    # ── PCA Visualization ──
    print("\n" + "-" * 50)
    print("PCA PROJECTION")
    print("-" * 50)
    
    pca_coords = {}
    for model_name, vectors in all_vectors.items():
        if len(vectors) < 2:
            print(f"  [{model_name}] Not enough words for PCA")
            continue
        
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)
        pca_coords[model_name] = coords
        
        explained_var = pca.explained_variance_ratio_
        print(f"  [{model_name}] Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
        
        # Individual plot
        plot_embeddings(
            coords, all_words[model_name], all_colors[model_name],
            all_labels[model_name],
            f"PCA — {model_name} Word Embeddings (IIT Jodhpur Corpus)",
            os.path.join(FIGURES_DIR, f"pca_{model_name.lower().replace('-', '')}.png")
        )
    
    # Comparison plot
    if len(pca_coords) == 2:
        plot_comparison(pca_coords, all_words, all_colors, all_labels,
                        "PCA", os.path.join(FIGURES_DIR, "pca_comparison.png"))
    
    # ── t-SNE Visualization ──
    print("\n" + "-" * 50)
    print("t-SNE PROJECTION")
    print("-" * 50)
    
    tsne_coords = {}
    for model_name, vectors in all_vectors.items():
        if len(vectors) < 2:
            print(f"  [{model_name}] Not enough words for t-SNE")
            continue
        
        perplexity = min(30, len(vectors) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     max_iter=1000, learning_rate="auto", init="pca")
        coords = tsne.fit_transform(vectors)
        tsne_coords[model_name] = coords
        
        # Individual plot
        plot_embeddings(
            coords, all_words[model_name], all_colors[model_name],
            all_labels[model_name],
            f"t-SNE — {model_name} Word Embeddings (IIT Jodhpur Corpus)",
            os.path.join(FIGURES_DIR, f"tsne_{model_name.lower().replace('-', '')}.png")
        )
    
    # Comparison plot
    if len(tsne_coords) == 2:
        plot_comparison(tsne_coords, all_words, all_colors, all_labels,
                        "t-SNE", os.path.join(FIGURES_DIR, "tsne_comparison.png"))
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
