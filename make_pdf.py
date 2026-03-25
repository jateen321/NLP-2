"""
make_pdf.py - Generate report.pdf using Matplotlib
===================================================
A simple script to render the report.md content into a 2-page PDF
using Matplotlib's PDF backend, as headlless environments
often lack specialized PDF libraries.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def main():
    report_md = "report.md"
    if not os.path.exists(report_md):
        print(f"Error: {report_md} not found!")
        return

    with open(report_md, "r") as f:
        content = f.read()

    # Split into lines and handle basic formatting
    lines = content.split("\n")
    
    # Create the PDF
    output_pdf = "report.pdf"
    with PdfPages(output_pdf) as pdf:
        # Page configuration
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Text rendering parameters
        curr_y = 0.95
        line_height = 0.02
        chars_per_line = 85
        
        # Title
        ax.text(0.05, curr_y, "NLP Assignment 2 Report", fontsize=16, fontweight='bold')
        curr_y -= 0.05
        
        for line in lines:
            if not line.strip():
                curr_y -= line_height
                continue
            
            # Simple bold/header handling
            fontsize = 10
            fontweight = 'normal'
            if line.startswith("# "):
                line = line[2:]
                fontsize = 14
                fontweight = 'bold'
                curr_y -= 0.02
            elif line.startswith("## "):
                line = line[3:]
                fontsize = 12
                fontweight = 'bold'
                curr_y -= 0.015
            elif line.startswith("### "):
                line = line[4:]
                fontsize = 11
                fontweight = 'bold'
                curr_y -= 0.01

            # Word wrap
            for i in range(0, len(line), chars_per_line):
                # If page is full, save and start new one
                if curr_y < 0.05:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    curr_y = 0.95

                chunk = line[i:i+chars_per_line]
                ax.text(0.05, curr_y, chunk, fontsize=fontsize, fontweight=fontweight, family='monospace')
                curr_y -= line_height

        pdf.savefig(fig)
        plt.close(fig)

    print(f"✓ Generated {output_pdf}")

if __name__ == "__main__":
    main()
