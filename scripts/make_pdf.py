"""
make_pdf.py - Premium Report Generator
=======================================
Generates a multi-page, visually-rich PDF report including:
- Page 1: Professional Cover Page
- Page 2: Problem 1 Analysis (Stats + Word Cloud)
- Page 3: Problem 1 Embedding Vector (300D)
- Page 4: Problem 2 Implementation (RNN vs LSTM vs Attention)
- Page 5: Problem 2 Results (Training Loss + Evaluation Metrics)

Uses Matplotlib for all rendering to ensure zero external PDF dependencies
other than the standard library and Matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

# --- Student Info ---
NAME = "Jateen"
ROLL = "B22CS026"
COURSE = "NLP (IIT Jodhpur)"
ASSIGNMENT = "Assignment 2: Word Embeddings & Name Generation"

# --- Image Paths ---
IMG_WORDCLOUD = "problem1/figures/wordcloud.png"
IMG_LOSS = "problem2/figures/training_loss.png"
IMG_EVAL = "problem2/figures/evaluation_comparison.png"

# --- Vector Data ---
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

COLOR_PRIMARY = "#1A3A5C"  # Dark Blue
COLOR_SECONDARY = "#E86E24"  # Orange
COLOR_TEXT = "#2C3E50"
COLOR_MUTED = "#7F8C8D"

font_header = {'fontsize': 24, 'fontweight': 'bold', 'color': COLOR_PRIMARY}
font_section = {'fontsize': 16, 'fontweight': 'bold', 'color': COLOR_PRIMARY}
font_body = {'fontsize': 11, 'color': COLOR_TEXT}
font_code = {'fontsize': 9, 'color': COLOR_TEXT, 'family': 'monospace'}

def create_page():
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    plt.axis('off')
    return fig

def add_header(ax, text):
    ax.text(0.05, 0.95, text, **font_section, transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.935, 0.935], color=COLOR_PRIMARY, lw=2, transform=ax.transAxes)

def wrap_text(text, width=80):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    lines.append(" ".join(current_line))
    return lines

def main():
    print("Generating premium report.pdf...")
    with PdfPages('report.pdf') as pdf:
        
        # --- PAGE 1: COVER ---
        fig = create_page()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Decorative stripes
        ax.add_patch(Rectangle((0, 0.8), 1, 0.2, color=COLOR_PRIMARY, alpha=0.1))
        ax.add_patch(Rectangle((0, 0), 1, 0.05, color=COLOR_PRIMARY))
        
        ax.text(0.5, 0.65, "NLP ASSIGNMENT 2", ha='center', **font_header)
        ax.text(0.5, 0.60, ASSIGNMENT.split(":")[1].strip().upper(), ha='center', 
                fontsize=14, fontweight='bold', color=COLOR_SECONDARY)
        
        ax.plot([0.3, 0.7], [0.55, 0.55], color=COLOR_MUTED, lw=1)
        
        ax.text(0.5, 0.45, f"Student: {NAME}", ha='center', fontsize=16, color=COLOR_TEXT)
        ax.text(0.5, 0.40, f"Roll Number: {ROLL}", ha='center', fontsize=16, color=COLOR_TEXT)
        ax.text(0.5, 0.35, COURSE, ha='center', fontsize=12, color=COLOR_MUTED)
        
        pdf.savefig(fig)
        plt.close(fig)

        # --- PAGE 2: PROBLEM 1 OVERVIEW ---
        fig = create_page()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        add_header(ax, "Problem 1: Word Embeddings from IIT Jodhpur Data")
        
        y = 0.88
        ax.text(0.05, y, "Dataset Statistics", fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
        y -= 0.04
        stats = [
            ("Corpus File Size", "0.937 MB"),
            ("Total Documents", "50 scraped pages"),
            ("Total Sentences", "5,228"),
            ("Total Tokens", "153,804"),
            ("Vocabulary Size", "9,970 unique tokens")
        ]
        for label, val in stats:
            ax.text(0.08, y, f"• {label}:", fontsize=11, fontweight='bold', color=COLOR_TEXT)
            ax.text(0.30, y, val, fontsize=11, color=COLOR_TEXT)
            y -= 0.025
            
        y -= 0.02
        ax.text(0.05, y, "Top-10 Words (Frequency-wise)", fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
        y -= 0.03
        top_words = "the, 6154, of, 4758, and, 4423, in, 2181, to, 2174, for, 1805, a, 1677, ug, 1014, iit, 951, students, 951"
        ax.text(0.08, y, top_words, fontsize=10, color=COLOR_TEXT, style='italic')
        
        y -= 0.08
        ax.text(0.05, y, "Corpus Curation & Preprocessing", fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
        y -= 0.04
        steps = [
            "Step-1: Data Collection — Scraped 50+ HTML/PDF sources using BeautifulSoup/PyPDF2.",
            "Step-2: Language Filtering — Heuristic removed non-English fragments.",
            "Step-3: Text Cleaning — Removed URLs, symbols, and formatting artifacts using regex.",
            "Step-4: Tokenization — Sentence and word level expansion via NLTK.",
            "Step-5: Normalization — Lowercasing and frequency filtering (min_count=2)."
        ]
        for s in steps:
            ax.text(0.08, y, s, fontsize=10, color=COLOR_TEXT)
            y -= 0.022
            
        # Add WordCloud Image
        if os.path.exists(IMG_WORDCLOUD):
            img = mpimg.imread(IMG_WORDCLOUD)
            # Center the image at the bottom
            ax_img = fig.add_axes([0.15, 0.05, 0.7, 0.35]) 
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title("Visualized Corpus Frequency (Word Cloud)", fontsize=10, color=COLOR_MUTED)

        pdf.savefig(fig)
        plt.close(fig)

        # --- PAGE 3: EMBEDDING VECTOR ---
        fig = create_page()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        add_header(ax, "Problem 1: 300-Dimensional Embedding Vector")
        
        y = 0.88
        ax.text(0.05, y, "Selected Word: 'student'", fontsize=13, fontweight='bold', color=COLOR_SECONDARY)
        y -= 0.03
        ax.text(0.05, y, "Trained via Skip-Gram (vector_size=300, window=5, negative=5, epochs=20)", fontsize=10, color=COLOR_MUTED)
        
        y -= 0.05
        # Box for vector
        rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.75, boxstyle="round,pad=0.02", ec=COLOR_MUTED, fc="#F8F9F9", lw=0.5)
        ax.add_patch(rect)
        
        wrapped_vec = wrap_text(STUDENT_VECTOR, width=95)
        vy = 0.83
        for line in wrapped_vec[:35]: # Limit to one page worth
            ax.text(0.07, vy, line, **font_code)
            vy -= 0.02
            
        pdf.savefig(fig)
        plt.close(fig)

        # --- PAGE 4: PROBLEM 2 ARCHITECTURE ---
        fig = create_page()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        add_header(ax, "Problem 2: Character-Level Name Generation")
        
        y = 0.88
        ax.text(0.05, y, "Model Implementation Details (From Scratch)", fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
        y -= 0.04
        
        models = [
            ("Vanilla RNN", "2-layer manual RNNCell stacking. Captures local dependencies.", "61,393"),
            ("Bidirectional LSTM", "Manual LSTMCell with forward/backward stacks. Captures both directions.", "442,193"),
            ("Attention + RNN", "LSTM base with Luong-style general attention mechanism.", "271,185")
        ]
        
        # Table Header
        ax.text(0.08, y, "Model Variant", fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        ax.text(0.30, y, "Architectural Approach", fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        ax.text(0.80, y, "Parameters", fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        y -= 0.015
        ax.plot([0.08, 0.92], [y, y], color=COLOR_MUTED, lw=0.5)
        y -= 0.03
        
        for name, desc, params in models:
            ax.text(0.08, y, name, fontsize=10, fontweight='bold', color=COLOR_TEXT)
            ax.text(0.30, y, desc, fontsize=10, color=COLOR_TEXT)
            ax.text(0.80, y, params, fontsize=10, color=COLOR_TEXT)
            y -= 0.04
            
        y -= 0.04
        ax.text(0.05, y, "Analysis: Which model works better?", fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
        y -= 0.03
        analysis = [
            "Vanilla RNN provided the best 'Indian' realism despite having the fewest parameters.",
            "The model successfully learned names like 'Chaitanya', 'Namit', and 'Vasuki'.",
            "",
            "Why? Complex models like BLSTM tended to overfit the small dataset (1000 names),",
            "memorizing patterns that didn't generalize to pronounceable new names. Vanilla RNN's",
            "simplicity acted as a regularizer, capturing the phonotactics of Indian names more robustly."
        ]
        for line in analysis:
            ax.text(0.08, y, line, fontsize=11, color=COLOR_TEXT)
            y -= 0.025

        # Small footer info
        ax.text(0.05, 0.05, f"Vanilla RNN Model Size: 0.24 MB on disk", fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

        pdf.savefig(fig)
        plt.close(fig)

        # --- PAGE 5: VISUAL RESULTS ---
        fig = create_page()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        add_header(ax, "Experimental Results & Quantitative Analysis")
        
        # Add Evaluation Comparison Image
        if os.path.exists(IMG_EVAL):
            img = mpimg.imread(IMG_EVAL)
            ax_img = fig.add_axes([0.1, 0.55, 0.8, 0.35])
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title("Novelty Rate & Diversity Across Architectures", fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        # Add Training Loss Image
        if os.path.exists(IMG_LOSS):
            img = mpimg.imread(IMG_LOSS)
            ax_img = fig.add_axes([0.1, 0.12, 0.8, 0.35])
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title("Convergence Analysis (Training Loss Curves)", fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

        pdf.savefig(fig)
        plt.close(fig)

    print("✓ Successfully generated premium report.pdf")

if __name__ == "__main__":
    main()
