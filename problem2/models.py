"""
models.py - Character-Level RNN Models (From Scratch)
======================================================
Implements three recurrent neural architectures for character-level
name generation:

1. Vanilla RNN
2. Bidirectional LSTM (BLSTM)
3. RNN with Basic Attention Mechanism

All models implemented from scratch using PyTorch nn.Module with
manual cell computations (no nn.RNN/nn.LSTM built-ins).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════
# 1. VANILLA RNN
# ══════════════════════════════════════════════
class VanillaRNNCell(nn.Module):
    """
    Single RNN cell: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    Implemented from scratch — not using nn.RNNCell.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Input-to-hidden weights
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        # Hidden-to-hidden weights
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # Bias
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Xavier initialization
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x, h_prev):
        """
        Args:
            x:      (batch, input_size)
            h_prev: (batch, hidden_size)
        Returns:
            h_next: (batch, hidden_size)
        """
        h_next = torch.tanh(
            x @ self.W_ih.t() + h_prev @ self.W_hh.t() + self.b_h
        )
        return h_next


class VanillaRNN(nn.Module):
    """
    Vanilla RNN for character-level name generation.

    Architecture:
        Char embedding → RNN cells (stacked layers) → Linear → Softmax
        
    Training: Teacher forcing — feed ground-truth char at each step.
    Generation: Autoregressive — feed predicted char at each step.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "Vanilla RNN"

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack of RNN cells
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(input_dim, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x:      (batch, seq_len) — character indices
            hidden: list of (batch, hidden_size) for each layer, or None
        Returns:
            output: (batch, seq_len, vocab_size)
            hidden: list of final hidden states
        """
        batch_size, seq_len = x.size()

        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                      for _ in range(self.num_layers)]

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs = []

        for t in range(seq_len):
            inp = embedded[:, t, :]  # (batch, embed_dim)
            new_hidden = []
            for i, cell in enumerate(self.rnn_cells):
                h = cell(inp, hidden[i])
                inp = self.dropout(h) if i < self.num_layers - 1 else h
                new_hidden.append(h)
            hidden = new_hidden
            outputs.append(hidden[-1])

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.fc_out(outputs)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════
# 2. BIDIRECTIONAL LSTM (BLSTM)
# ══════════════════════════════════════════════
class LSTMCell(nn.Module):
    """
    Single LSTM cell — implemented from scratch.
    
    Gates:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     — forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     — input gate
        g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  — candidate cell
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     — output gate
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency (4 gates)
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # Initialize
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        # Forget gate bias = 1 (important for training stability)
        nn.init.constant_(self.bias[hidden_size:2*hidden_size], 1.0)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x:      (batch, input_size)
            h_prev: (batch, hidden_size)
            c_prev: (batch, hidden_size)
        Returns:
            h_next: (batch, hidden_size)
            c_next: (batch, hidden_size)
        """
        gates = x @ self.W_ih.t() + h_prev @ self.W_hh.t() + self.bias  # (batch, 4*hidden)

        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        i_gate = torch.sigmoid(i_gate)   # Input gate
        f_gate = torch.sigmoid(f_gate)   # Forget gate
        g_gate = torch.tanh(g_gate)      # Candidate
        o_gate = torch.sigmoid(o_gate)   # Output gate

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next


class BLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level name generation.

    Architecture:
        Char embedding → Forward LSTM + Backward LSTM
        → Concatenate hidden states → Linear → Softmax

    Note: For generation, we use the forward direction primarily.
    The bidirectional encoding helps learn better representations
    during training, and the combined hidden state is used for
    the output projection.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "Bidirectional LSTM"

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Forward LSTM cells (each direction stacks independently)
        self.fwd_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_size
            self.fwd_cells.append(LSTMCell(input_dim, hidden_size))

        # Backward LSTM cells
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_size
            self.bwd_cells.append(LSTMCell(input_dim, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(2 * hidden_size, vocab_size)  # Concat fwd+bwd

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len)
        Returns:
            output: (batch, seq_len, vocab_size)
            hidden: tuple of final states
        """
        batch_size, seq_len = x.size()
        device = x.device

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # ── Forward pass ──
        fwd_h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        fwd_c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]

        fwd_outputs = []
        for t in range(seq_len):
            inp = embedded[:, t, :]
            for i, cell in enumerate(self.fwd_cells):
                fwd_h[i], fwd_c[i] = cell(inp, fwd_h[i], fwd_c[i])
                inp = self.dropout(fwd_h[i]) if i < self.num_layers - 1 else fwd_h[i]
            fwd_outputs.append(fwd_h[-1])

        # ── Backward pass ──
        bwd_h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        bwd_c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]

        bwd_outputs = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            inp = embedded[:, t, :]
            for i, cell in enumerate(self.bwd_cells):
                bwd_h[i], bwd_c[i] = cell(inp, bwd_h[i], bwd_c[i])
                inp = self.dropout(bwd_h[i]) if i < self.num_layers - 1 else bwd_h[i]
            bwd_outputs[t] = bwd_h[-1]

        # Concatenate forward and backward
        fwd_stack = torch.stack(fwd_outputs, dim=1)  # (batch, seq, hidden)
        bwd_stack = torch.stack(bwd_outputs, dim=1)   # (batch, seq, hidden)
        combined = torch.cat([fwd_stack, bwd_stack], dim=2)  # (batch, seq, 2*hidden)

        logits = self.fc_out(combined)  # (batch, seq_len, vocab_size)

        return logits, (fwd_h, fwd_c)

    def generate_forward(self, x, hidden=None):
        """
        Forward-only pass for autoregressive generation.
        """
        batch_size, seq_len = x.size()
        device = x.device

        if hidden is None:
            fwd_h = [torch.zeros(batch_size, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
            fwd_c = [torch.zeros(batch_size, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
        else:
            fwd_h, fwd_c = hidden

        embedded = self.embedding(x)
        outputs = []

        for t in range(seq_len):
            inp = embedded[:, t, :]
            for i, cell in enumerate(self.fwd_cells):
                fwd_h[i], fwd_c[i] = cell(inp, fwd_h[i], fwd_c[i])
                inp = fwd_h[i]
            # Use forward hidden only, zero-pad backward for fc_out
            combined = torch.cat([fwd_h[-1], torch.zeros_like(fwd_h[-1])], dim=1)
            outputs.append(combined)

        outputs = torch.stack(outputs, dim=1)
        logits = self.fc_out(outputs)
        return logits, (fwd_h, fwd_c)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════
# 3. RNN WITH BASIC ATTENTION
# ══════════════════════════════════════════════
class AttentionRNN(nn.Module):
    """
    RNN with Basic Attention Mechanism for character-level name generation.

    Architecture:
        Char embedding → RNN encoder (stores all hidden states)
        → Attention over all previous hidden states
        → Context vector + current hidden → Linear → Softmax

    Attention:
        score(h_t, h_j) = h_t^T W_attn h_j     (Luong general attention)
        α = softmax(scores)
        context = Σ α_j * h_j
        output = tanh(W_c [context; h_t])
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "RNN + Attention"

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN cells (using LSTM cells for better gradient flow)
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_size
            self.rnn_cells.append(LSTMCell(input_dim, hidden_size))

        # Attention weights
        self.W_attn = nn.Linear(hidden_size, hidden_size, bias=False)

        # Combine context + hidden
        self.W_combine = nn.Linear(2 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def attention(self, query, keys):
        """
        Compute attention weights and context vector.

        Args:
            query: (batch, hidden_size) — current hidden state
            keys:  (batch, num_steps, hidden_size) — all previous hidden states

        Returns:
            context: (batch, hidden_size) — weighted sum of keys
            weights: (batch, num_steps) — attention weights
        """
        # Luong general attention: score = query^T * W * key
        transformed = self.W_attn(keys)  # (batch, num_steps, hidden)
        scores = torch.bmm(transformed, query.unsqueeze(2)).squeeze(2)  # (batch, num_steps)

        weights = F.softmax(scores, dim=1)  # (batch, num_steps)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (batch, hidden)

        return context, weights

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len)
        Returns:
            output: (batch, seq_len, vocab_size)
            hidden: (h_list, c_list)
        """
        batch_size, seq_len = x.size()
        device = x.device

        if hidden is None:
            h_list = [torch.zeros(batch_size, self.hidden_size, device=device)
                      for _ in range(self.num_layers)]
            c_list = [torch.zeros(batch_size, self.hidden_size, device=device)
                      for _ in range(self.num_layers)]
        else:
            h_list, c_list = hidden

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Collect all hidden states for attention
        all_hidden = []
        outputs = []

        for t in range(seq_len):
            inp = embedded[:, t, :]

            # Run through RNN layers
            for i, cell in enumerate(self.rnn_cells):
                h_list[i], c_list[i] = cell(inp, h_list[i], c_list[i])
                inp = self.dropout(h_list[i]) if i < self.num_layers - 1 else h_list[i]

            current_h = h_list[-1]  # (batch, hidden)
            all_hidden.append(current_h)

            # Apply attention over all previous hidden states
            if len(all_hidden) > 1:
                keys = torch.stack(all_hidden[:-1], dim=1)  # (batch, t, hidden)
                context, _ = self.attention(current_h, keys)
                # Combine context with current hidden
                combined = torch.cat([context, current_h], dim=1)
                output = torch.tanh(self.W_combine(combined))
            else:
                # First timestep — no attention possible
                output = current_h

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.fc_out(outputs)  # (batch, seq_len, vocab_size)

        return logits, (h_list, c_list)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════
# UTILITY: Print model architecture summary
# ══════════════════════════════════════════════
def print_model_summary(model):
    """Print architecture details and parameter count."""
    print(f"\n  Model: {model.model_name}")
    print(f"  {'─' * 50}")
    print(f"  Vocab size:     {model.vocab_size}")
    print(f"  Embedding dim:  {model.embed_dim}")
    print(f"  Hidden size:    {model.hidden_size}")
    print(f"  Num layers:     {model.num_layers}")
    print(f"  Total params:   {model.count_parameters():,}")
    print(f"  {'─' * 50}")

    # Detailed layer breakdown
    for name, param in model.named_parameters():
        print(f"    {name:40s} → {str(list(param.shape)):>20s} = {param.numel():>8,}")
    print()
