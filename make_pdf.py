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

# ── Data (Extracted from report.md) ──────────────────────────────────
REPORT_FILE = "report.md"

def get_report_content():
    if not os.path.exists(REPORT_FILE):
        return "Error: report.md not found"
    with open(REPORT_FILE, "r") as f:
        return f.read()

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


# ── PDF Generation ────────────────────────────────────────────────────
def main():
    content = get_report_content()
    lines = content.splitlines()

    output = "report.pdf"
    with PdfPages(output) as pdf:
        fig, ax = new_page(pdf)
        header_rect(ax, "#1a3a5c")
        t(ax, 0.03, 0.965, "NLP Assignment 2 — B22CS026", fontsize=14,
          fontweight="bold", color="white")
        t(ax, 0.97, 0.965, "Jateen", fontsize=11, color="#aac8e8", ha="right")

        y = 0.90
        line_height = 0.018
        
        for line in lines:
            if not line.strip():
                y -= line_height
                continue
            
            # Simple header handling
            fontsize = 10
            fontweight = 'normal'
            color = "#222222"
            
            if line.startswith("# "):
                line = line[2:]
                fontsize = 14
                fontweight = 'bold'
                color = "#1a3a5c"
                y -= 0.01
            elif line.startswith("## "):
                line = line[3:]
                fontsize = 12
                fontweight = 'bold'
                color = "#1a3a5c"
                y -= 0.008
                hline(ax, y + 0.012)
            elif line.startswith("### "):
                line = line[4:]
                fontsize = 11
                fontweight = 'bold'
                color = "#1a3a5c"
                section_bg(ax, y - 0.005, height=0.025)
            
            # Use wrap_text for very long lines just in case
            wrapped = wrap_text(line, width=95)
            for w_line in wrapped:
                if y < 0.05:
                    add_page(pdf, fig)
                    fig, ax = new_page(pdf)
                    header_rect(ax, "#1a3a5c")
                    y = 0.90
                
                t(ax, 0.03, y, w_line, fontsize=fontsize, fontweight=fontweight, color=color,
                  fontfamily='monospace' if "," in w_line and len(w_line) > 50 else 'sans-serif')
                y -= line_height

        add_page(pdf, fig)

    size_kb = os.path.getsize(output) / 1024
    print(f"✓ {output} generated ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
