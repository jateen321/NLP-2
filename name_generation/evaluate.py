"""
evaluate.py - Quantitative Evaluation & Qualitative Analysis
=============================================================
Generates names from all trained models and computes:
- Novelty Rate: % of generated names not in training set
- Diversity: unique generated / total generated
- Representative samples and failure mode analysis

Usage:
    python name_generation/evaluate.py

Output:
    name_generation/generated_vanilla_rnn.txt
    name_generation/generated_blstm.txt
    name_generation/generated_attention_rnn.txt
    name_generation/evaluation_results.json
    name_generation/figures/evaluation_comparison.png
"""

import os
import sys
import json
import random
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from models import VanillaRNN, BLSTM, AttentionRNN
from train import CharVocab, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
NAMES_FILE = os.path.join(BASE_DIR, "TrainingNames.txt")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

NUM_GENERATE = 500   # Number of names to generate per model
MAX_LENGTH = 15      # Maximum name length
TEMPERATURE = 0.8    # Sampling temperature
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════
# NAME GENERATION
# ══════════════════════════════════════════════
def generate_name(model, vocab, temperature=0.8, max_length=20):
    """Generate a single name autoregressively."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Start with <SOS>
        current_char = torch.tensor([[vocab.sos_idx]], dtype=torch.long, device=device)
        hidden = None
        name_chars = []

        for _ in range(max_length):
            if isinstance(model, BLSTM):
                logits, hidden = model.generate_forward(current_char, hidden)
            else:
                logits, hidden = model(current_char, hidden)

            # Get logits for last timestep
            logits = logits[:, -1, :] / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab.eos_idx:
                break
            if next_idx == vocab.pad_idx:
                continue

            char = vocab.idx2char.get(next_idx, "?")
            if char in ("<SOS>", "<PAD>", "<EOS>"):
                continue
            name_chars.append(char)
            current_char = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return "".join(name_chars)


def generate_names(model, vocab, n=500, temperature=0.8, max_length=20):
    """Generate n names from a model."""
    names = []
    for _ in range(n):
        name = generate_name(model, vocab, temperature, max_length)
        if name:  # Skip empty names
            names.append(name)
    return names


# ══════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════
def compute_metrics(generated_names, training_names):
    """Compute novelty rate and diversity."""
    training_set = set(n.lower() for n in training_names)

    total = len(generated_names)
    if total == 0:
        return {"novelty_rate": 0, "diversity": 0, "total": 0}

    # Novelty: % not in training set
    novel = sum(1 for n in generated_names if n.lower() not in training_set)
    novelty_rate = novel / total * 100

    # Diversity: unique / total
    unique = len(set(generated_names))
    diversity = unique / total * 100

    return {
        "total_generated": total,
        "novel_count": novel,
        "novelty_rate": round(novelty_rate, 2),
        "unique_count": unique,
        "diversity": round(diversity, 2),
    }


def analyze_failure_modes(generated_names):
    """Identify common failure modes."""
    failures = {
        "too_short": [],     # < 2 chars
        "too_long": [],      # > 15 chars
        "repetitive": [],    # Repeated patterns
        "nonsense": [],      # All same char
        "no_vowels": [],     # No vowels
    }

    vowels = set("aeiouAEIOU")

    for name in generated_names:
        if len(name) < 2:
            failures["too_short"].append(name)
        elif len(name) > 15:
            failures["too_long"].append(name)

        # Check for repetitive patterns (same 2+ chars repeated)
        if len(name) >= 4:
            for i in range(len(name) - 3):
                if name[i:i+2] == name[i+2:i+4]:
                    failures["repetitive"].append(name)
                    break

        # Check all same character
        if len(set(name.lower())) == 1 and len(name) > 1:
            failures["nonsense"].append(name)

        # Check no vowels
        if not any(c in vowels for c in name):
            failures["no_vowels"].append(name)

    return {k: {"count": len(v), "examples": v[:5]} for k, v in failures.items()}


# ══════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════
def plot_evaluation(results, output_path):
    """Plot evaluation metrics comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = list(results.keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # Novelty Rate
    ax = axes[0]
    novelty = [results[m]["metrics"]["novelty_rate"] for m in model_names]
    bars = ax.bar(model_names, novelty, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Novelty Rate (%)", fontsize=12)
    ax.set_title("Novelty Rate", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, novelty):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Diversity
    ax = axes[1]
    diversity = [results[m]["metrics"]["diversity"] for m in model_names]
    bars = ax.bar(model_names, diversity, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Diversity (%)", fontsize=12)
    ax.set_title("Diversity (Unique / Total)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, diversity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Quantitative Evaluation — Character-Level Name Generation",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Evaluation plot saved to: {output_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CHARACTER-LEVEL NAME GENERATION — EVALUATION")
    print("=" * 70)

    # Load training names
    with open(NAMES_FILE) as f:
        training_names = [line.strip() for line in f if line.strip()]
    print(f"\n  Training names: {len(training_names)}")

    # Build vocab
    vocab = CharVocab(training_names)
    vocab_size = len(vocab)

    # Models to evaluate
    model_configs = [
        ("Vanilla RNN", "vanilla_rnn", VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("Bidirectional LSTM", "blstm", BLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("RNN + Attention", "attention_rnn", AttentionRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
    ]

    all_results = {}

    for model_name, tag, model in model_configs:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─' * 60}")

        # Load checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{tag}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Generate names
        print(f"  Generating {NUM_GENERATE} names...")
        generated = generate_names(model, vocab, NUM_GENERATE, TEMPERATURE, MAX_LENGTH)
        print(f"  Generated: {len(generated)} names")

        # Save generated names
        gen_path = os.path.join(BASE_DIR, f"generated_{tag}.txt")
        with open(gen_path, "w") as f:
            for name in generated:
                f.write(name + "\n")
        print(f"  ✓ Saved to: {gen_path}")

        # Compute metrics
        metrics = compute_metrics(generated, training_names)
        print(f"  Novelty Rate: {metrics['novelty_rate']:.1f}%")
        print(f"  Diversity:    {metrics['diversity']:.1f}%")

        # Failure analysis
        failures = analyze_failure_modes(generated)

        # Representative samples
        sample_size = min(30, len(generated))
        samples = random.sample(generated, sample_size)

        all_results[model_name] = {
            "metrics": metrics,
            "failures": failures,
            "samples": samples,
            "parameters": model.count_parameters(),
        }

        print(f"\n  Sample generated names:")
        for i, name in enumerate(samples[:10]):
            print(f"    {i+1:2d}. {name}")

        print(f"\n  Failure modes:")
        for mode, info in failures.items():
            if info["count"] > 0:
                print(f"    {mode}: {info['count']} ({', '.join(info['examples'][:3])})")

    # Save all results
    results_path = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved to: {results_path}")

    # Plot comparison
    if all_results:
        plot_evaluation(all_results, os.path.join(FIGURES_DIR, "evaluation_comparison.png"))

    # Summary table
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} {'Params':>10s} {'Novelty%':>10s} {'Diversity%':>12s}")
    print(f"{'─' * 60}")
    for model_name, res in all_results.items():
        print(f"{model_name:25s} {res['parameters']:>10,} "
              f"{res['metrics']['novelty_rate']:>9.1f}% "
              f"{res['metrics']['diversity']:>11.1f}%")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
