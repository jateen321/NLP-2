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
STUDENT_VECTOR = ("-0.0332, 0.0768, -0.0540, 0.0166, 0.1287, -0.0581, 0.1325, 0.3132, 0.0152, -0.2531, "
                  "0.1639, -0.3321, 0.0136, 0.1670, -0.0625, 0.0303, 0.2647, 0.0302, 0.0870, -0.2013, "
                  "-0.0363, -0.0993, -0.0402, -0.0409, 0.2379, -0.0981, -0.0079, 0.1399, -0.0009, -0.1955, "
                  "0.2793, 0.2148, 0.2573, 0.0491, 0.0061, -0.0867, 0.0691, -0.1670, -0.0555, -0.2244, "
                  "0.0210, -0.0878, 0.1224, -0.1436, 0.2938, 0.0395, -0.0657, 0.1391, 0.2132, 0.1279, "
                  "-0.0335, 0.2173, -0.0258, 0.4945, 0.0994, 0.3428, 0.1370, 0.0780, 0.0195, 0.0295, "
                  "0.0804, 0.2847, 0.1784, 0.0062, -0.2059, 0.1322, 0.0449, 0.0781, -0.0034, 0.1986, "
                  "-0.0058, 0.1291, 0.1675, -0.2651, 0.2191, 0.0783, -0.1050, 0.0214, -0.1892, -0.0813, "
                  "-0.0829, 0.1267, -0.0512, 0.3153, 0.1579, 0.1999, 0.0993, -0.1966, 0.1001, 0.3156, "
                  "0.1392, -0.0833, -0.1254, 0.1173, 0.0844, -0.0081, 0.0483, -0.1338, 0.2399, -0.0644, "
                  "-0.0824, -0.0188, -0.1603, 0.0743, -0.0908, -0.1876, -0.0464, 0.1061, -0.0887, -0.0982, "
                  "-0.2921, 0.0763, 0.0154, 0.2368, -0.0027, 0.0362, 0.0480, -0.0192, 0.0926, -0.3540, "
                  "0.1749, -0.1050, 0.0442, -0.0509, -0.0589, 0.0479, -0.1219, -0.2245, 0.2742, 0.2679, "
                  "0.0255, -0.1560, -0.0653, -0.1290, 0.0717, 0.4318, -0.0419, -0.1544, -0.1650, -0.0917, "
                  "0.0400, -0.2401, -0.1098, -0.0277, -0.1026, -0.2030, -0.0642, 0.0018, 0.0783, -0.2667, "
                  "-0.0960, -0.1297, 0.0692, 0.0572, 0.0013, 0.0187, -0.2271, -0.0971, -0.0112, -0.0152, "
                  "-0.1142, 0.0045, -0.0459, 0.0722, 0.1139, 0.3181, 0.0746, -0.0624, -0.0267, 0.1068, "
                  "-0.2320, -0.0301, -0.0596, 0.1046, -0.0659, 0.0053, -0.0784, -0.0829, -0.0141, 0.1260, "
                  "0.0115, 0.0642, 0.0477, 0.0450, -0.1304, 0.1254, 0.1332, 0.0799, 0.1418, 0.0242, "
                  "0.3049, -0.0933, -0.1628, 0.1725, -0.0006, -0.3005, 0.0397, 0.0649, -0.1059, 0.0509, "
                  "-0.0735, -0.0505, -0.1170, 0.0720, 0.1789, -0.0574, 0.1407, 0.1869, 0.1582, 0.0563, "
                  "-0.1839, 0.0663, -0.0328, -0.2670, -0.0342, -0.0568, 0.0133, -0.4136, -0.1195, -0.1311, "
                  "-0.0663, 0.0298, -0.1353, -0.0021, -0.1324, -0.1946, -0.0774, 0.1757, -0.0338, -0.0149, "
                  "0.1579, -0.0143, 0.0173, 0.1348, -0.1733, 0.0279, -0.1201, -0.0066, 0.0334, -0.1462, "
                  "0.1127, -0.1394, -0.1055, 0.1111, 0.0273, -0.0760, 0.2463, -0.1008, 0.2680, 0.3570, "
                  "0.0466, -0.1980, 0.1632, -0.0232, -0.2050, -0.0057, 0.1749, -0.0948, -0.5011, -0.1659, "
                  "-0.1496, -0.0508, 0.1911, -0.1177, -0.4334, 0.1428, 0.0994, 0.0340, -0.0555, 0.0211, "
                  "0.0251, 0.0851, 0.2171, 0.1014, -0.0047, 0.0674, 0.1998, 0.0234, 0.0488, -0.1421, "
                  "0.0388, -0.0111, -0.0579, 0.3688, 0.0446, -0.2802, -0.1295, -0.0330, -0.0471, 0.1167, "
                  "0.2951, 0.0938, 0.1564, -0.0060, 0.1240, 0.1138, 0.0743, 0.0030, 0.0674, -0.2208")


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
    t(ax, 0.06, y, "0.168 MB  (175,925 bytes)", fontsize=10, color="#333333")
    t(ax, 0.55, y, "Documents: 36 | Tokens: 25,964 | Vocab: 3,547",
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

    top10 = [("the",1552),("of",1088),("and",773),("to",640),
              ("for",542),("in",482),("a",468),("be",295),
              ("student",238),("by",232)]

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
