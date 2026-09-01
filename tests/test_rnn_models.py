"""
Correctness checks for the from-scratch recurrent cells and models.

These do not assert on training quality — they assert on the things that
silently break when recurrent code is written by hand: tensor shapes,
gradient flow through every parameter, the LSTM forget-gate bias
initialisation, and the causality of the attention mask.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "name_generation"))

from models import AttentionRNN, BLSTM, LSTMCell, VanillaRNN, VanillaRNNCell  # noqa: E402

VOCAB, EMBED, HIDDEN, LAYERS = 49, 32, 128, 2
BATCH, SEQ = 4, 7


@pytest.fixture(params=[VanillaRNN, BLSTM, AttentionRNN])
def model(request):
    torch.manual_seed(0)
    return request.param(VOCAB, EMBED, HIDDEN, num_layers=LAYERS, dropout=0.1)


# ── Shapes ─────────────────────────────────────────────────────────

def test_forward_returns_logits_over_vocab_for_every_timestep(model):
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits, _ = model(x)
    assert logits.shape == (BATCH, SEQ, VOCAB)


def test_vanilla_rnn_cell_preserves_batch_and_hidden_dims():
    cell = VanillaRNNCell(EMBED, HIDDEN)
    h = cell(torch.randn(BATCH, EMBED), torch.zeros(BATCH, HIDDEN))
    assert h.shape == (BATCH, HIDDEN)
    assert torch.all(h.abs() <= 1.0), "tanh output must lie in [-1, 1]"


def test_lstm_cell_returns_hidden_and_cell_state():
    cell = LSTMCell(EMBED, HIDDEN)
    h, c = cell(torch.randn(BATCH, EMBED),
                torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN))
    assert h.shape == (BATCH, HIDDEN)
    assert c.shape == (BATCH, HIDDEN)


def test_models_handle_single_step_and_single_example(model):
    logits, _ = model(torch.randint(0, VOCAB, (1, 1)))
    assert logits.shape == (1, 1, VOCAB)


# ── Gradients ──────────────────────────────────────────────────────

def test_every_parameter_receives_gradient(model):
    """A hand-written cell that forgets to use a weight trains silently and wrongly."""
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits, _ = model(x)
    logits.sum().backward()

    dead = [name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, f"parameters received no gradient: {dead}"


def test_gradients_are_finite(model):
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits, _ = model(x)
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), torch.randint(0, VOCAB, (BATCH * SEQ,))
    ).backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


# ── Initialisation ─────────────────────────────────────────────────

def test_lstm_forget_gate_bias_initialised_to_one():
    """Standard trick: forget bias = 1 keeps the cell state open early in training."""
    cell = LSTMCell(EMBED, HIDDEN)
    forget_bias = cell.bias[HIDDEN:2 * HIDDEN]
    assert torch.allclose(forget_bias, torch.ones(HIDDEN))


def test_parameter_counts_match_reported_results():
    """Guards the parameter counts quoted in RESULTS.md against silent drift."""
    counts = {
        VanillaRNN: 61_393,
        BLSTM: 442_193,
        AttentionRNN: 271_185,
    }
    for cls, expected in counts.items():
        model = cls(VOCAB, EMBED, HIDDEN, num_layers=LAYERS, dropout=0.1)
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert actual == expected, f"{cls.__name__}: {actual} != {expected}"


# ── Causality ──────────────────────────────────────────────────────

def test_attention_output_at_step_t_ignores_future_characters():
    """
    Luong attention here scores only j < t. If future positions leaked in, the
    model would be non-causal and generation would be invalid.
    """
    torch.manual_seed(0)
    model = AttentionRNN(VOCAB, EMBED, HIDDEN, num_layers=LAYERS, dropout=0.0)
    model.eval()

    a = torch.randint(0, VOCAB, (1, SEQ))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % VOCAB  # perturb only the final character

    with torch.no_grad():
        out_a, _ = model(a)
        out_b, _ = model(b)

    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-6), \
        "changing the last character altered earlier outputs — model is not causal"
