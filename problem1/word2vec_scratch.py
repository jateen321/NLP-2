"""
word2vec_scratch.py - Word2Vec Implementation From Scratch
===========================================================
Custom PyTorch implementation of Word2Vec (CBOW and Skip-gram with
Negative Sampling) for comparison with gensim.

Architecture:
- CBOW: Predicts center word from context words
- Skip-gram: Predicts context words from center word
- Negative Sampling: Efficient approximation of softmax

Usage:
    python problem1/word2vec_scratch.py

Output:
    problem1/models/scratch_cbow.pt
    problem1/models/scratch_skipgram.pt
    problem1/comparison_results.json
"""

import os
import json
import random
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_FILE = os.path.join(BASE_DIR, "comparison_results.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
EMBEDDING_DIM = 200
WINDOW_SIZE = 5
NUM_NEGATIVE = 5
MIN_COUNT = 2
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════
# VOCABULARY
# ══════════════════════════════════════════════
class Vocabulary:
    """Builds word-to-index mapping and frequency table."""

    def __init__(self, sentences, min_count=2):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Count frequencies
        for sent in sentences:
            for word in sent:
                self.word_freq[word] += 1

        # Filter by min_count and build mappings
        idx = 0
        for word, count in self.word_freq.items():
            if count >= min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        # Rebuild freq for filtered vocab
        self.freq = np.zeros(len(self.word2idx))
        for word, idx in self.word2idx.items():
            self.freq[idx] = self.word_freq[word]

        # Negative sampling distribution: P(w)^0.75
        self.neg_distribution = self.freq ** 0.75
        self.neg_distribution /= self.neg_distribution.sum()

        print(f"  Vocabulary size: {len(self.word2idx)}")
        print(f"  Total tokens (filtered): {int(self.freq.sum())}")

    def __len__(self):
        return len(self.word2idx)

    def __contains__(self, word):
        return word in self.word2idx

    def encode(self, word):
        return self.word2idx.get(word, -1)

    def decode(self, idx):
        return self.idx2word.get(idx, "<UNK>")

    def sample_negatives(self, n):
        """Sample n negative word indices."""
        return np.random.choice(len(self), size=n, p=self.neg_distribution)


# ══════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════
class CBOWDataset(Dataset):
    """Dataset for CBOW: (context_words) → center_word."""

    def __init__(self, sentences, vocab, window_size, num_negative):
        self.data = []
        self.vocab = vocab
        self.num_negative = num_negative

        for sent in sentences:
            indices = [vocab.encode(w) for w in sent if w in vocab]
            if len(indices) < 2 * window_size + 1:
                continue

            for i in range(window_size, len(indices) - window_size):
                context = []
                for j in range(i - window_size, i + window_size + 1):
                    if j != i:
                        context.append(indices[j])
                center = indices[i]
                self.data.append((context, center))

        print(f"  CBOW dataset: {len(self.data):,} training pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, center = self.data[idx]
        # Sample negatives
        negatives = self.vocab.sample_negatives(self.num_negative).tolist()
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(center, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram: center_word → context_word."""

    def __init__(self, sentences, vocab, window_size, num_negative):
        self.data = []
        self.vocab = vocab
        self.num_negative = num_negative

        for sent in sentences:
            indices = [vocab.encode(w) for w in sent if w in vocab]
            for i in range(len(indices)):
                center = indices[i]
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        self.data.append((center, indices[j]))

        print(f"  Skip-gram dataset: {len(self.data):,} training pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        negatives = self.vocab.sample_negatives(self.num_negative).tolist()
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )


# ══════════════════════════════════════════════
# MODELS (From Scratch)
# ══════════════════════════════════════════════
class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) with Negative Sampling.

    Architecture:
        Input:  context word indices (2*window_size words)
        Embed:  Average of context word embeddings  →  R^d
        Output: Dot product with center word embedding + negative samples
        Loss:   Binary cross-entropy (positive=1, negatives=0)
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Two embedding matrices: input (context) and output (center)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)

    def forward(self, context, center, negatives):
        """
        Args:
            context:   (batch, 2*window)  — context word indices
            center:    (batch,)           — center word index
            negatives: (batch, num_neg)   — negative sample indices

        Returns:
            loss: scalar
        """
        # Context embedding: average of context word embeddings
        ctx_embed = self.in_embeddings(context).mean(dim=1)  # (batch, dim)

        # Positive score: dot product with center word output embedding
        center_embed = self.out_embeddings(center)  # (batch, dim)
        pos_score = (ctx_embed * center_embed).sum(dim=1)  # (batch,)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)

        # Negative scores
        neg_embed = self.out_embeddings(negatives)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_embed, ctx_embed.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def get_word_vector(self, word_idx):
        """Get the learned embedding for a word."""
        return self.in_embeddings.weight[word_idx].detach().cpu().numpy()


class SkipGramModel(nn.Module):
    """
    Skip-gram with Negative Sampling (SGNS).

    Architecture:
        Input:  center word index
        Embed:  Center word embedding  →  R^d
        Output: Dot product with context word embedding + negative samples
        Loss:   Binary cross-entropy (positive=1, negatives=0)
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)

    def forward(self, center, context, negatives):
        """
        Args:
            center:    (batch,)           — center word index
            context:   (batch,)           — context word index
            negatives: (batch, num_neg)   — negative sample indices

        Returns:
            loss: scalar
        """
        center_embed = self.in_embeddings(center)  # (batch, dim)

        # Positive
        ctx_embed = self.out_embeddings(context)  # (batch, dim)
        pos_score = (center_embed * ctx_embed).sum(dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)

        # Negatives
        neg_embed = self.out_embeddings(negatives)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def get_word_vector(self, word_idx):
        return self.in_embeddings.weight[word_idx].detach().cpu().numpy()


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
def train_model(model, dataset, epochs, batch_size, lr, model_name):
    """Train a Word2Vec model."""
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training {model_name} on {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

    loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = [t.to(device) for t in batch]
            loss = model(*batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    model = model.cpu()
    total_time = time.time() - start_time
    print(f"  ✓ Training complete in {total_time:.1f}s | Final loss: {loss_history[-1]:.4f}")

    return model, loss_history


# ══════════════════════════════════════════════
# EVALUATION & COMPARISON
# ══════════════════════════════════════════════
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_nearest_neighbors(model, vocab, word, topn=5):
    """Find top-N nearest neighbors for a word."""
    if word not in vocab:
        return None

    word_idx = vocab.encode(word)
    word_vec = model.get_word_vector(word_idx)

    # Compute similarity with all words
    all_vecs = model.in_embeddings.weight.detach().cpu().numpy()
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = all_vecs / norms
    word_norm = word_vec / max(np.linalg.norm(word_vec), 1e-8)

    sims = normalized @ word_norm
    top_indices = np.argsort(sims)[::-1][1:topn+1]  # Skip self

    return [(vocab.decode(idx), float(sims[idx])) for idx in top_indices]


def analogy(model, vocab, pos_words, neg_words, topn=5):
    """Solve word analogy: pos - neg."""
    all_words = pos_words + neg_words
    for w in all_words:
        if w not in vocab:
            return None, f"'{w}' not in vocabulary"

    # Compute analogy vector
    vec = np.zeros(EMBEDDING_DIM)
    for w in pos_words:
        vec += model.get_word_vector(vocab.encode(w))
    for w in neg_words:
        vec -= model.get_word_vector(vocab.encode(w))

    # Find nearest
    all_vecs = model.in_embeddings.weight.detach().cpu().numpy()
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = all_vecs / norms
    vec_norm = vec / max(np.linalg.norm(vec), 1e-8)

    sims = normalized @ vec_norm
    # Exclude input words
    exclude_indices = {vocab.encode(w) for w in all_words}
    for idx in exclude_indices:
        sims[idx] = -1

    top_indices = np.argsort(sims)[::-1][:topn]
    return [(vocab.decode(idx), float(sims[idx])) for idx in top_indices], None


def compare_with_gensim(scratch_models, vocab):
    """Compare scratch models with gensim models."""
    from gensim.models import Word2Vec as GensimW2V

    print("\n" + "=" * 70)
    print("COMPARISON: SCRATCH vs GENSIM")
    print("=" * 70)

    gensim_models = {}
    gensim_cbow = os.path.join(MODELS_DIR, "best_cbow.model")
    gensim_sg = os.path.join(MODELS_DIR, "best_skipgram.model")

    if os.path.exists(gensim_cbow):
        gensim_models["CBOW"] = GensimW2V.load(gensim_cbow)
    if os.path.exists(gensim_sg):
        gensim_models["SkipGram"] = GensimW2V.load(gensim_sg)

    query_words = ["research", "student", "phd", "examination", "engineering"]
    comparison = {}

    for word in query_words:
        print(f"\n  Query: '{word}'")
        print(f"  {'─' * 65}")
        comparison[word] = {}

        for model_name in ["CBOW", "SkipGram"]:
            # Scratch results
            if model_name in scratch_models:
                neighbors = get_nearest_neighbors(scratch_models[model_name], vocab, word, topn=5)
                if neighbors:
                    comparison[word][f"Scratch_{model_name}"] = neighbors
                    print(f"  [Scratch {model_name:>8}]  ", end="")
                    for w, s in neighbors:
                        print(f"{w}({s:.3f})  ", end="")
                    print()

            # Gensim results
            if model_name in gensim_models:
                gm = gensim_models[model_name]
                if word in gm.wv:
                    g_neighbors = gm.wv.most_similar(word, topn=5)
                    comparison[word][f"Gensim_{model_name}"] = [
                        {"word": w, "sim": round(s, 4)} for w, s in g_neighbors
                    ]
                    print(f"  [Gensim {model_name:>8}]  ", end="")
                    for w, s in g_neighbors:
                        print(f"{w}({s:.3f})  ", end="")
                    print()

    # Analogy comparison
    print(f"\n{'=' * 70}")
    print("ANALOGY COMPARISON")
    print("=" * 70)

    analogies = [
        {"desc": "UG : B.Tech :: PG : ?", "pos": ["b.tech", "pg"], "neg": ["ug"]},
        {"desc": "student : learning :: faculty : ?", "pos": ["learning", "faculty"], "neg": ["student"]},
        {"desc": "semester : examination :: thesis : ?", "pos": ["examination", "thesis"], "neg": ["semester"]},
    ]

    analogy_comparison = {}
    for a in analogies:
        print(f"\n  {a['desc']}")
        print(f"  {'─' * 55}")
        analogy_comparison[a["desc"]] = {}

        for model_name in ["CBOW", "SkipGram"]:
            # Scratch
            if model_name in scratch_models:
                result, err = analogy(scratch_models[model_name], vocab, a["pos"], a["neg"])
                if result:
                    analogy_comparison[a["desc"]][f"Scratch_{model_name}"] = result
                    print(f"  [Scratch {model_name:>8}]  ", end="")
                    for w, s in result[:3]:
                        print(f"{w}({s:.3f})  ", end="")
                    print()
                elif err:
                    print(f"  [Scratch {model_name:>8}]  {err}")

            # Gensim
            if model_name in gensim_models:
                gm = gensim_models[model_name]
                all_words = a["pos"] + a["neg"]
                missing = [w for w in all_words if w not in gm.wv]
                if not missing:
                    result = gm.wv.most_similar(positive=a["pos"], negative=a["neg"], topn=3)
                    analogy_comparison[a["desc"]][f"Gensim_{model_name}"] = [
                        {"word": w, "sim": round(s, 4)} for w, s in result
                    ]
                    print(f"  [Gensim {model_name:>8}]  ", end="")
                    for w, s in result:
                        print(f"{w}({s:.3f})  ", end="")
                    print()
                else:
                    print(f"  [Gensim {model_name:>8}]  Missing: {missing}")

    return {"neighbors": comparison, "analogies": analogy_comparison}


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def load_corpus():
    sentences = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                sentences.append(tokens)
    return sentences


def main():
    print("=" * 70)
    print("WORD2VEC FROM SCRATCH (PyTorch)")
    print("=" * 70)
    print(f"\n  Config: dim={EMBEDDING_DIM}, window={WINDOW_SIZE}, "
          f"neg={NUM_NEGATIVE}, epochs={EPOCHS}, lr={LEARNING_RATE}")

    # Load corpus
    print("\n[1/5] Loading corpus...")
    sentences = load_corpus()
    print(f"  Loaded {len(sentences):,} sentences")

    # Build vocabulary
    print("\n[2/5] Building vocabulary...")
    vocab = Vocabulary(sentences, min_count=MIN_COUNT)

    # Create datasets
    print("\n[3/5] Creating datasets...")
    cbow_dataset = CBOWDataset(sentences, vocab, WINDOW_SIZE, NUM_NEGATIVE)
    sg_dataset = SkipGramDataset(sentences, vocab, WINDOW_SIZE, NUM_NEGATIVE)

    # Train CBOW
    print("\n[4/5] Training models...")
    cbow_model = CBOWModel(len(vocab), EMBEDDING_DIM)
    cbow_model, cbow_loss = train_model(
        cbow_model, cbow_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, "CBOW"
    )

    # Train Skip-gram
    sg_model = SkipGramModel(len(vocab), EMBEDDING_DIM)
    sg_model, sg_loss = train_model(
        sg_model, sg_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, "Skip-gram"
    )

    # Save models
    torch.save({
        "model_state": cbow_model.state_dict(),
        "loss_history": cbow_loss,
        "config": {"dim": EMBEDDING_DIM, "window": WINDOW_SIZE, "neg": NUM_NEGATIVE},
    }, os.path.join(MODELS_DIR, "scratch_cbow.pt"))
    print(f"\n  ✓ Saved scratch CBOW model")

    torch.save({
        "model_state": sg_model.state_dict(),
        "loss_history": sg_loss,
        "config": {"dim": EMBEDDING_DIM, "window": WINDOW_SIZE, "neg": NUM_NEGATIVE},
    }, os.path.join(MODELS_DIR, "scratch_skipgram.pt"))
    print(f"  ✓ Saved scratch Skip-gram model")

    # Compare with gensim
    print("\n[5/5] Comparing with gensim models...")
    scratch_models = {"CBOW": cbow_model, "SkipGram": sg_model}
    comparison = compare_with_gensim(scratch_models, vocab)

    # Save comparison results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(comparison, f, indent=2, default=convert)
    print(f"\n  ✓ Comparison saved to: {RESULTS_FILE}")

    # Print parameter counts
    print(f"\n{'=' * 70}")
    print("MODEL PARAMETER COUNTS")
    print(f"{'=' * 70}")
    cbow_params = sum(p.numel() for p in cbow_model.parameters())
    sg_params = sum(p.numel() for p in sg_model.parameters())
    print(f"  CBOW:      {cbow_params:,} trainable parameters")
    print(f"  Skip-gram: {sg_params:,} trainable parameters")
    print(f"  (Both: 2 × {len(vocab)} × {EMBEDDING_DIM} = {2 * len(vocab) * EMBEDDING_DIM:,})")

    print(f"\n{'=' * 70}")
    print("FROM-SCRATCH TRAINING COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
