"""
train.py - Training Loop for All Character-Level Models
========================================================
Trains Vanilla RNN, BLSTM, and RNN+Attention models for
character-level name generation.

Usage:
    python name_generation/train.py

Output:
    name_generation/checkpoints/vanilla_rnn.pt
    name_generation/checkpoints/blstm.pt
    name_generation/checkpoints/attention_rnn.pt
    name_generation/figures/training_loss.png
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from models import VanillaRNN, BLSTM, AttentionRNN, print_model_summary

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
NAMES_FILE = os.path.join(BASE_DIR, "TrainingNames.txt")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hyperparameters
EMBED_DIM = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.003
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════
# CHARACTER VOCABULARY
# ══════════════════════════════════════════════
class CharVocab:
    """Character-level vocabulary with special tokens."""

    def __init__(self, names):
        chars = set()
        for name in names:
            for c in name:
                chars.add(c)

        self.chars = sorted(chars)
        self.char2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        for i, c in enumerate(self.chars):
            self.char2idx[c] = i + 3

        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2

    def encode(self, name):
        """Encode name to indices: <SOS> + chars + <EOS>."""
        return [self.sos_idx] + [self.char2idx[c] for c in name] + [self.eos_idx]

    def decode(self, indices):
        """Decode indices to string, stopping at <EOS>."""
        chars = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            chars.append(self.idx2char.get(idx, "?"))
        return "".join(chars)

    def __len__(self):
        return len(self.char2idx)


# ══════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════
class NamesDataset(Dataset):
    """Character-level name dataset."""

    def __init__(self, names, vocab):
        self.data = []
        for name in names:
            encoded = vocab.encode(name)
            # Input: <SOS> + chars  |  Target: chars + <EOS>
            self.data.append(torch.tensor(encoded, dtype=torch.long))
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = seq[:-1]   # Input: everything except last
        y = seq[1:]     # Target: everything except first
        return x, y


def collate_fn(batch):
    """Pad sequences to same length in a batch."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    x_padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_padded = torch.full((len(ys), max_len), -100, dtype=torch.long)  # -100 = ignore in CE loss

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, :len(x)] = x
        y_padded[i, :len(y)] = y

    return x_padded, y_padded


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
def train_model(model, dataloader, epochs, lr, device, model_tag):
    """Train a single model and return loss history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(x_batch)
            # Reshape: (batch * seq_len, vocab) vs (batch * seq_len,)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.5f} | Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    model = model.cpu()

    # Save checkpoint
    save_path = os.path.join(CHECKPOINTS_DIR, f"{model_tag}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "loss_history": loss_history,
        "config": {
            "embed_dim": EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        }
    }, save_path)
    print(f"  ✓ Saved to {save_path} | Total time: {total_time:.1f}s")

    return model, loss_history


def plot_training_curves(loss_dict, output_path):
    """Plot training loss curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Vanilla RNN": "#e74c3c", "Bidirectional LSTM": "#3498db",
              "RNN + Attention": "#2ecc71"}

    for model_name, losses in loss_dict.items():
        color = colors.get(model_name, "#333")
        ax.plot(range(1, len(losses) + 1), losses, label=model_name,
                color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss — Character-Level Name Generation Models",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(list(loss_dict.values())[0]))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved to: {output_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CHARACTER-LEVEL NAME GENERATION — TRAINING")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Hyperparameters: embed={EMBED_DIM}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, lr={LEARNING_RATE}, epochs={EPOCHS}")

    # Load names
    print("\n[1/5] Loading names...")
    with open(NAMES_FILE) as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(names)} names")

    # Build vocabulary
    print("\n[2/5] Building character vocabulary...")
    vocab = CharVocab(names)
    print(f"  Vocabulary size: {len(vocab)} (including {PAD_TOKEN}, {SOS_TOKEN}, {EOS_TOKEN})")
    print(f"  Characters: {''.join(vocab.chars)}")

    # Save vocab for later use
    vocab_path = os.path.join(CHECKPOINTS_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": vocab.char2idx, "idx2char": {str(k): v for k, v in vocab.idx2char.items()}}, f)

    # Create dataset
    print("\n[3/5] Creating dataset...")
    dataset = NamesDataset(names, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, drop_last=False)
    print(f"  Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")

    # Create models
    print("\n[4/5] Creating and training models...")
    vocab_size = len(vocab)

    models_config = [
        ("vanilla_rnn", VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("blstm", BLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("attention_rnn", AttentionRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
    ]

    all_losses = {}

    for tag, model in models_config:
        print(f"\n{'─' * 60}")
        print_model_summary(model)

        model, losses = train_model(model, dataloader, EPOCHS, LEARNING_RATE, device, tag)
        all_losses[model.model_name] = losses

    # Plot training curves
    print(f"\n[5/5] Plotting training curves...")
    plot_training_curves(all_losses, os.path.join(FIGURES_DIR, "training_loss.png"))

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
