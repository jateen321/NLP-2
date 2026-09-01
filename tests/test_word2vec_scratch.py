"""
Correctness checks for the from-scratch Word2Vec implementation.

The negative sampler and the min_count filter are the two places where a
subtle error produces embeddings that look plausible but are wrong, so they
are asserted on directly.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embeddings"))

from word2vec_scratch import (  # noqa: E402
    CBOWDataset, CBOWModel, SkipGramDataset, SkipGramModel, Vocabulary,
)

SENTENCES = [
    ["the", "institute", "offers", "research", "programmes"],
    ["the", "institute", "offers", "engineering", "programmes"],
    ["research", "and", "engineering", "at", "the", "institute"],
    ["students", "join", "the", "research", "programmes"],
]
DIM = 16


@pytest.fixture
def vocab():
    return Vocabulary(SENTENCES, min_count=2)


# ── Vocabulary ─────────────────────────────────────────────────────

def test_min_count_filters_rare_words(vocab):
    assert "the" in vocab          # appears 4 times
    assert "institute" in vocab    # appears 3 times
    assert "students" not in vocab  # appears once
    assert "join" not in vocab      # appears once


def test_encode_and_decode_round_trip(vocab):
    for word in ("the", "research", "programmes"):
        assert vocab.decode(vocab.encode(word)) == word


def test_encode_returns_sentinel_for_out_of_vocabulary_word(vocab):
    assert vocab.encode("thisworddoesnotexist") == -1


# ── Negative sampling ──────────────────────────────────────────────

def test_negative_sampling_distribution_is_normalised(vocab):
    assert vocab.neg_distribution.shape == (len(vocab),)
    assert np.isclose(vocab.neg_distribution.sum(), 1.0)
    assert np.all(vocab.neg_distribution > 0)


def test_negative_distribution_uses_three_quarter_power(vocab):
    """P(w) ∝ f(w)^0.75 — the exponent that flattens the unigram distribution."""
    expected = vocab.freq ** 0.75
    expected = expected / expected.sum()
    assert np.allclose(vocab.neg_distribution, expected)


def test_three_quarter_power_dampens_frequent_words(vocab):
    """The whole point of the exponent: frequent words are sampled less often
    than their raw frequency share would dictate."""
    raw = vocab.freq / vocab.freq.sum()
    most_frequent = int(np.argmax(vocab.freq))
    assert vocab.neg_distribution[most_frequent] < raw[most_frequent]


def test_sampled_negatives_are_valid_in_vocabulary_indices(vocab):
    negatives = vocab.sample_negatives(64)
    assert negatives.shape == (64,)
    assert negatives.min() >= 0
    assert negatives.max() < len(vocab)


# ── Datasets ───────────────────────────────────────────────────────

def test_skipgram_yields_more_pairs_than_cbow(vocab):
    """Skip-gram treats each (target, context) pair separately, so for the same
    corpus and window it must produce strictly more training examples."""
    cbow = CBOWDataset(SENTENCES, vocab, window_size=2, num_negative=5)
    skipgram = SkipGramDataset(SENTENCES, vocab, window_size=2, num_negative=5)
    assert len(skipgram) > len(cbow)


@pytest.mark.parametrize("dataset_cls", [CBOWDataset, SkipGramDataset])
def test_dataset_items_have_expected_negative_count(vocab, dataset_cls):
    dataset = dataset_cls(SENTENCES, vocab, window_size=2, num_negative=5)
    *_, negatives = dataset[0]
    assert negatives.shape == (5,)


# ── Models ─────────────────────────────────────────────────────────

def test_models_keep_separate_input_and_output_embedding_tables(vocab):
    """Tying the tables would make every word its own nearest neighbour."""
    for cls in (CBOWModel, SkipGramModel):
        model = cls(len(vocab), DIM)
        assert model.in_embeddings.weight is not model.out_embeddings.weight
        assert model.in_embeddings.weight.shape == (len(vocab), DIM)
        assert model.out_embeddings.weight.shape == (len(vocab), DIM)


def test_loss_is_scalar_finite_and_differentiable(vocab):
    """Negative-sampling BCE must reduce to a scalar with usable gradients."""
    for cls, dataset_cls in ((CBOWModel, CBOWDataset), (SkipGramModel, SkipGramDataset)):
        torch.manual_seed(0)
        dataset = dataset_cls(SENTENCES, vocab, window_size=2, num_negative=5)
        model = cls(len(vocab), DIM)

        first, second, negatives = dataset[0]
        loss = model(first.unsqueeze(0), second.unsqueeze(0), negatives.unsqueeze(0))

        assert loss.dim() == 0, f"{cls.__name__} loss is not a scalar"
        assert torch.isfinite(loss), f"{cls.__name__} loss is not finite"

        loss.backward()
        assert model.in_embeddings.weight.grad is not None
        assert torch.isfinite(model.in_embeddings.weight.grad).all()
