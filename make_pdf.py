"""
make_pdf.py - Generate a high-quality report.pdf
=================================================
Creates a well-formatted, multi-page PDF report using ReportLab-like
layout via Matplotlib's PDF backend.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import os

# ── Data ─────────────────────────────────────────────────────────────
STUDENT_VECTOR = ("-0.4817, 0.3603, 0.0493, 0.4250, -0.3770, -0.0673, 0.0594, 0.3550, 0.4441, 0.0108, "
                  "0.4734, 0.2026, -0.0270, 0.5809, -0.2244, -0.5872, 0.0789, -0.1344, -0.1896, -0.5505, "
                  "0.3547, 0.5694, -0.2025, 0.3512, 0.2020, -0.0790, -0.4439, -0.6036, 0.4092, 0.0152, "
                  "-0.8031, -0.1061, 0.2068, -0.0710, 0.1819, -0.1978, 0.1547, 0.2107, -0.0920, -0.2565, "
                  "0.3691, -0.9547, -0.2578, -0.0465, 0.6128, -0.1441, 0.2610, -0.0176, -0.5609, 0.1823, "
                  "-0.1211, -0.2968, -0.2824, -0.4080, 0.1438, -0.1003, 0.1738, -0.1842, 0.6324, 0.2406, "
                  "0.1792, 0.3348, 0.2112, -0.3231, -0.0153, 0.3718, -0.3385, 0.1989, 0.0410, 1.1182, "
                  "0.2469, 0.2355, -0.7243, -0.1846, -0.7783, -0.2690, -0.2737, 0.1185, -0.1831, 0.0040, "
                  "-0.3267, 0.3263, 0.1461, -0.3609, -0.2161, -0.4403, 0.5465, -0.3295, -0.0014, -0.1480, "
                  "-0.3047, -0.0644, -0.4163, -0.2091, -0.4770, -0.1277, -0.3629, 0.3530, 0.2578, 0.4835, "
                  "0.2564, 0.3275, 0.7638, -0.5899, -0.1397, -0.1678, 0.3567, -0.2041, -0.4595, -0.1959, "
                  "-0.0501, 0.2504, -0.0106, -0.1677, -0.4346, 0.0239, 0.1260, 0.1062, -0.7511, 0.0171, "
                  "0.2365, 0.0898, -0.6174, -0.2059, -0.2231, 0.0275, -0.5134, 0.5352, 0.0127, 0.0232, "
                  "-0.1253, -0.1755, 0.2097, -0.4404, -0.4178, -0.0805, -0.3315, -0.0120, -0.2659, -0.5955, "
                  "0.0000, 0.1635, -0.2164, -0.0874, -0.1708, -0.0451, 0.7730, -0.0028, 0.0360, -0.3441, "
                  "0.0187, -0.3389, 0.2444, 0.3140, -0.2221, -0.4685, -0.4509, 0.3447, 0.2380, -0.0195, "
                  "-0.1323, -0.0985, -0.0388, 0.0161, -0.5808, 0.0477, 0.2680, -0.2245, 0.5469, 0.2117, "
                  "-0.5573, 0.4815, 0.3396, -0.0683, 0.0381, -0.5191, 0.0503, -0.1327, -0.0704, 0.1641, "
                  "-0.1939, -0.2861, -0.6366, 0.1738, -0.6682, -0.1642, -0.0663, -0.2858, -0.1228, -0.1536, "
                  "-0.0461, -0.6755, -0.1673, -0.1956, -0.3070, -0.3313, -0.1935, 0.1770, 0.0497, -0.4135, "
                  "0.5211, -0.6660, -0.3345, -0.0580, -0.5002, -0.3094, -0.1044, -0.2882, 0.0873, 0.3455, "
                  "-0.2908, 0.1523, 0.1844, 0.3205, 0.2177, 0.3667, -0.1245, -0.2883, 0.5046, 0.6984, "
                  "-0.4036, 0.1387, 0.5568, 0.0524, 0.2946, 0.1458, 0.5299, -0.0657, -0.7453, 0.0522, "
                  "0.3259, 0.0074, 0.2894, 0.0353, -0.5590, 0.0846, 0.2497, -0.2185, 0.2897, -0.1043, "
                  "-0.5683, 0.3164, -0.0470, 0.0958, -0.2009, -0.5172, -0.0821, -0.0654, 0.6564, -0.8608, "
                  "0.1632, 0.1346, -0.0572, 0.0744, -0.2215, -0.0736, 0.1789, -0.0172, 0.0034, 0.3277, "
                  "0.4270, -0.2283, -0.4266, 0.3323, 0.3555, 0.3139, -0.1971, -0.0703, -0.5721, 0.5277, "
                  "0.2169, 0.2295, 0.0806, -0.0821, -0.0810, -0.0473, 0.0258, 0.2180, -1.0748, -0.3741, "
                  "-0.4136, -0.2667, 0.0858, -0.1651, 0.4323, -0.6172, 0.3687, 0.5480, 0.1269, 0.4493, "
                  "-0.1455, -0.6407, 0.5454, -0.4102, -0.3061, 0.0469, 0.3951, 0.2586, -0.2073, -0.1485")


# ── Helpers ───────────────────────────────────────────────────────────
def new_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_page(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def header_rect(ax, color="#1a3a5c"):
    rect = FancyBboxPatch((0, 0.93), 1, 0.07, boxstyle="square,pad=0",
                          facecolor=color, transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)


def section_bg(ax, y, height=0.032, color="#e8eef5"):
    rect = FancyBboxPatch((0.02, y - 0.005), 0.96, height,
                          boxstyle="round,pad=0.005", facecolor=color,
                          edgecolor="#c0cce0", linewidth=0.5,
                          transform=ax.transAxes)
    ax.add_patch(rect)


def hline(ax, y, color="#c0cce0"):
    ax.plot([0.02, 0.98], [y, y], color=color, linewidth=0.7,
            transform=ax.transAxes)


def t(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, **kw)


def wrap_text(text, width=88):
    """Wrap text to given width."""
    words = text.split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


# ── Page 1: Cover + Problem 1 ─────────────────────────────────────────
def page1(pdf):
    fig, ax = new_page(pdf)

    # Header bar
    header_rect(ax, "#1a3a5c")
    t(ax, 0.03, 0.965, "NLP Assignment 2 — B22CS026", fontsize=14,
      fontweight="bold", color="white")
    t(ax, 0.97, 0.965, "Jateen", fontsize=11, color="#aac8e8", ha="right")

    # Title block
    t(ax, 0.5, 0.875, "Problem 1: Word Embeddings from IIT Jodhpur Data",
      fontsize=13, fontweight="bold", ha="center", color="#1a3a5c")
    hline(ax, 0.86)

    y = 0.840

    # ── P1 Q1: Corpus size ──────────────────────────────────────────
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "P1 ▸  Corpus File Size", fontsize=10,
      fontweight="bold", color="#1a3a5c")
    y -= 0.038
    t(ax, 0.06, y, "0.937 MB  (982,830 bytes)", fontsize=10, color="#333333")
    t(ax, 0.55, y, "Documents: 50 | Tokens: 153,804 | Vocab: 9,970",
      fontsize=9, color="#555555")
    y -= 0.035

    # ── P1 Q2: Preprocessing steps ─────────────────────────────────
    hline(ax, y + 0.010)
    y -= 0.005
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "P1 ▸  Corpus Curation & Preprocessing Steps",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.038

    steps = [
        ("Step-1", "Data Collection",
         "Scraped 36 pages from IIT Jodhpur website (Departments, Academics, Research,\n"
         "          Faculty profiles, Announcements) using Requests + BeautifulSoup."),
        ("Step-2", "Boilerplate Removal",
         "Removed website navigation, headers, footers, and non-English lines using\n"
         "          ASCII character proportion heuristic (>70% ASCII = English)."),
        ("Step-3", "Text Cleaning",
         "Removed URLs, email addresses, LaTeX fragments, excessive punctuation,\n"
         "          numbers and non-textual content using regular expressions."),
        ("Step-4", "Lowercasing",
         "Converted all text to lower-case for vocabulary normalization."),
        ("Step-5", "Tokenization",
         "Tokenized into sentences using NLTK's sent_tokenize, then into words\n"
         "          using word_tokenize."),
        ("Step-6", "Frequency Filtering",
         "Removed words with frequency < 2 (used as min_count in Word2Vec training)\n"
         "          to reduce noise from rare/misspelled tokens."),
    ]

    for step_id, title, desc in steps:
        t(ax, 0.06, y, f"• {step_id}: {title} —", fontsize=9,
          fontweight="bold", color="#1a3a5c")
        for i, line in enumerate(desc.split("\n")):
            t(ax, 0.06 if i == 0 else 0.10, y - (0.0 if i == 0 else 0.018),
              ("" if i == 0 else "") + line.strip(), fontsize=9, color="#333333")
        y -= 0.042

    hline(ax, y + 0.005)
    y -= 0.015

    # ── P1 Q4: Top-10 words ─────────────────────────────────────────
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "P1 ▸  Top-10 Words (Frequency-wise)", fontsize=10,
      fontweight="bold", color="#1a3a5c")
    y -= 0.038

    top10 = [("the",6154),("of",4758),("and",4423),("in",2181),
              ("to",2174),("for",1805),("a",1677),("ug",1014),
              ("iit",951),("students",951)]

    col_w = 0.18
    for i, (word, freq) in enumerate(top10):
        col = i % 5
        row = i // 5
        xp = 0.06 + col * col_w
        yp = y - row * 0.025
        t(ax, xp, yp, f"{i+1}. {word}", fontsize=9.5,
          fontweight="bold", color="#1a3a5c")
        t(ax, xp + 0.08, yp, f"{freq:,}", fontsize=9, color="#555555")
    y -= 0.058

    hline(ax, y + 0.005)
    y -= 0.015

    # ── P1 Q5: Analogy ──────────────────────────────────────────────
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "P1 ▸  Interesting Analogy Found", fontsize=10,
      fontweight="bold", color="#1a3a5c")
    y -= 0.038
    analogy_lines = [
        "  semester : examination  ::  thesis : defense",
        "  Interpretation: Just as a semester culminates in an examination, a PhD thesis",
        "  culminates in a 'defense' (viva-voce). The model also resolved:",
        "  UG : B.Tech  ::  PG : M.Tech    and    student : learning  ::  faculty : teaching",
    ]
    for line in analogy_lines:
        t(ax, 0.04, y, line, fontsize=9.2,
          color="#1a3a5c" if "::" in line else "#333333",
          fontweight="bold" if "::" in line else "normal",
          fontfamily="monospace" if "::" in line else "sans-serif")
        y -= 0.020

    add_page(pdf, fig)


# ── Page 2: Embedding Vector ──────────────────────────────────────────
def page2(pdf):
    fig, ax = new_page(pdf)
    header_rect(ax)
    t(ax, 0.03, 0.965, "P1 ▸  300-Dimensional Embedding Vector for 'student'",
      fontsize=12, fontweight="bold", color="white")
    t(ax, 0.97, 0.965, "B22CS026", fontsize=10, color="#aac8e8", ha="right")

    note = ("Trained using Word2Vec Skip-gram (vector_size=300, window=5, "
            "negative_sampling=5, min_count=2, epochs=20) on the IITJ corpus.")
    t(ax, 0.03, 0.920, note, fontsize=8.5, color="#555555", style="italic")
    hline(ax, 0.910)

    t(ax, 0.03, 0.893, "student  — ", fontsize=10,
      fontweight="bold", color="#1a3a5c")

    # Wrap the vector into lines of ~90 chars
    vec_lines = wrap_text(STUDENT_VECTOR, width=90)
    y = 0.875
    for line in vec_lines:
        t(ax, 0.03, y, line, fontsize=7.8, color="#222222",
          fontfamily="monospace")
        y -= 0.025

    add_page(pdf, fig)


# ── Page 3: Problem 2 ─────────────────────────────────────────────────
def page3(pdf):
    fig, ax = new_page(pdf)
    header_rect(ax, "#1a3a5c")
    t(ax, 0.03, 0.965, "Problem 2: Character-Level Name Generation using RNN Variants",
      fontsize=12, fontweight="bold", color="white")
    t(ax, 0.97, 0.965, "B22CS026", fontsize=10, color="#aac8e8", ha="right")

    y = 0.920

    # Dataset
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "Dataset (Task-0)", fontsize=10,
      fontweight="bold", color="#1a3a5c")
    y -= 0.038
    t(ax, 0.06, y,
      "1,000 unique Indian names curated from North Indian, South Indian, Bengali,\n"
      "Gujarati, Marathi, and pan-Indian name pools (stored in TrainingNames.txt).",
      fontsize=9.5, color="#333333")
    y -= 0.045

    # Architecture table
    hline(ax, y + 0.005)
    y -= 0.010
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "Model Architectures (Task-1) — All implemented from scratch using PyTorch",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.042

    headers = ["Model", "Architecture Description", "Params", "Size"]
    col_x = [0.03, 0.22, 0.73, 0.88]
    for hdr, xp in zip(headers, col_x):
        t(ax, xp, y, hdr, fontsize=9, fontweight="bold", color="#1a3a5c")
    y -= 0.004
    hline(ax, y)
    y -= 0.020

    rows = [
        ("Vanilla RNN", "Manual VanillaRNNCell (tanh activation). 2 stacked\n"
         "           layers. No gating mechanism. Simple recurrence:\n"
         "           h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)",
         "61,393", "0.24 MB"),
        ("Bi-LSTM",    "Manual LSTMCell with 4 gates (i/f/o/g). Forward\n"
         "           + Backward passes independently stacked (2 layers\n"
         "           each). Outputs concatenated for projection.",
         "442,193", "1.69 MB"),
        ("Attention\n+RNN",
         "Manual LSTMCell (2 layers) + Luong general attention.\n"
         "           At each step: score = h_t^T·W·h_j; context =\n"
         "           Σα_j·h_j; output = tanh(W_c·[context;h_t])",
         "271,185", "1.04 MB"),
    ]

    for name, desc, params, sz in rows:
        t(ax, col_x[0], y, name, fontsize=9, fontweight="bold", color="#333333")
        for i, dl in enumerate(desc.split("\n")):
            t(ax, col_x[1], y - i * 0.018, dl.strip(), fontsize=8.5, color="#333333")
        t(ax, col_x[2], y, params, fontsize=9, color="#333333")
        t(ax, col_x[3], y, sz, fontsize=9, color="#333333")
        y -= 0.056
        hline(ax, y + 0.004, color="#e0e8f0")

    y -= 0.010

    # Hyperparameters
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "Hyperparameters (shared across all models)",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.038
    hp = ("Embedding dim: 32  |  Hidden size: 128  |  Layers: 2  |  "
          "Learning rate: 0.003  |  Epochs: 100 (Vanilla) / 60 (Bi-LSTM, Attention)  |  "
          "Dropout: 0.2 (Vanilla), 0.4 (Bi-LSTM), 0.3 (Attention)  |  "
          "Optimizer: Adam  |  LR scheduler: StepLR (γ=0.5)")
    for line in wrap_text(hp, 90):
        t(ax, 0.06, y, line, fontsize=9, color="#333333")
        y -= 0.020

    y -= 0.010

    # Quantitative results
    hline(ax, y + 0.005)
    y -= 0.010
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "Quantitative Evaluation (Task-2)",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.040

    # Mini table
    t(ax, 0.06, y, "Model", fontsize=9, fontweight="bold", color="#1a3a5c")
    t(ax, 0.35, y, "Novelty Rate", fontsize=9, fontweight="bold", color="#1a3a5c")
    t(ax, 0.58, y, "Diversity", fontsize=9, fontweight="bold", color="#1a3a5c")
    t(ax, 0.75, y, "Sample Names Generated", fontsize=9, fontweight="bold", color="#1a3a5c")
    y -= 0.004
    hline(ax, y, color="#c0cce0")
    y -= 0.022

    evals = [
        ("Vanilla RNN",  "24.6%",  "80.4%",  "Chaitanya, Namit, Vasuki, Sahan, Shubh, Devi"),
        ("Bi-LSTM",      "100.0%", "97.6%",  "Ojw, Yegi, Tookst, Yotas, Gokgs, Hgoy"),
        ("Att+RNN",      "100.0%", "100.0%", "Diverse but less realistic character strings"),
    ]
    for model, nov, div, samples in evals:
        t(ax, 0.06, y, model, fontsize=9, color="#333333")
        t(ax, 0.35, y, nov, fontsize=9, color="#333333",
          fontweight="bold" if nov == "24.6%" else "normal")
        t(ax, 0.58, y, div, fontsize=9, color="#333333")
        t(ax, 0.75, y, samples, fontsize=8, color="#555555")
        y -= 0.022

    y -= 0.012
    hline(ax, y + 0.005)
    y -= 0.018

    # P2 question answers
    section_bg(ax, y, 0.030)
    t(ax, 0.03, y + 0.012, "P2 ▸  Which model works better, and why?",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.040

    qual_text = [
        "Vanilla RNN performs best for realistic name generation on this dataset.",
        "",
        "• Vanilla RNN excels because Indian names follow short, consistent phonotactic patterns",
        "  (avg 6.1 chars). Its simple recurrence h_t = tanh(W_ih·x + W_hh·h_{t-1}) captures these",
        "  local character transitions effectively without over-parameterization.",
        "",
        "• Bi-LSTM (442K params) massively over-fitted the 1000-name training set during teacher",
        "  forcing (training loss → 0.003) making its autoregressive generation unreliable, since",
        "  the bidirectional context used during training is unavailable at inference time.",
        "",
        "• Attention+RNN had the highest diversity (100%) but generated less realistic names,",
        "  as the attention mechanism requires longer sequences to learn meaningful alignments;",
        "  with short names (avg ~6 chars) there is insufficient context to attend over.",
        "",
        "  Conclusion: For short, domain-specific sequences with a small training set,",
        "  simpler models generalize better than over-parameterized ones.",
    ]
    for line in qual_text:
        t(ax, 0.04, y, line, fontsize=8.8, color="#222222" if line else "#333333")
        y -= 0.018

    y -= 0.008

    section_bg(ax, y, 0.028)
    t(ax, 0.03, y + 0.010, "P2 ▸  Vanilla RNN — Number of Parameters & Model Size",
      fontsize=10, fontweight="bold", color="#1a3a5c")
    y -= 0.032
    param_lines = [
        "• Trainable Parameters: 61,393",
        "  Breakdown: embedding (1,568) + rnn_cell_0.W_ih (4,096) + rnn_cell_0.W_hh (16,384)",
        "             + rnn_cell_0.b (128) + rnn_cell_1.W_ih (16,384) + rnn_cell_1.W_hh (16,384)",
        "             + rnn_cell_1.b (128) + fc_out.weight (6,272) + fc_out.bias (49)",
        "• Model Size on Disk: 0.24 MB  (checkpoint including optimizer state)",
    ]
    for line in param_lines:
        t(ax, 0.04, y, line, fontsize=8.8, color="#333333")
        y -= 0.018

    add_page(pdf, fig)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    output = "report.pdf"
    with PdfPages(output) as pdf:
        # Page metadata
        d = pdf.infodict()
        d['Title'] = 'NLP Assignment 2 Report'
        d['Author'] = 'Jateen — B22CS026'
        d['Subject'] = 'Word Embeddings and Character-Level Name Generation'

        page1(pdf)
        page2(pdf)
        page3(pdf)

    size_kb = os.path.getsize(output) / 1024
    print(f"✓ {output} generated ({size_kb:.1f} KB, 3 pages)")


if __name__ == "__main__":
    main()
