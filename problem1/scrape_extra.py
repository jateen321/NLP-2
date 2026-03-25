"""
scrape_extra.py - Scrape additional IIT Jodhpur sources
========================================================
Scrapes:
1. Annual Reports page (finds all PDF links and downloads them)
2. Vision & Mission page
3. History page
4. 4 specific PDFs (Brochure, Newsletter, Newsletter2, Senate Constitution)

Extracts text from PDFs using PyPDF2, saves raw text files.

Usage:
    python problem1/scrape_extra.py
"""

import os, re, time, requests
from bs4 import BeautifulSoup
import PyPDF2
import io

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def fetch_html(url, timeout=20):
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  ✗ HTML fetch error for {url!r}: {e}")
        return ""


def extract_html_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "noscript", "form", "button", "meta"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def fetch_pdf_bytes(url, timeout=40):
    try:
        r = SESSION.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  ✗ PDF fetch error for {url!r}: {e}")
        return None


def extract_pdf_text(pdf_bytes):
    """Extract all text from a PDF bytes object."""
    text_parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    text_parts.append(t.strip())
            except Exception:
                pass
    except Exception as e:
        print(f"    PDF parse error: {e}")
    return "\n".join(text_parts)


def save_raw(filename, text):
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)
    size = len(text)
    print(f"  ✓ Saved {filename!r} ({size:,} chars)")
    return size


def is_english_line(text):
    """Quick heuristic: >60% ASCII = English."""
    if not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.6


def clean_text(text):
    """Basic cleaning for raw scraped text."""
    lines = text.splitlines()
    clean = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        if not is_english_line(line):
            continue
        # Remove lines that are purely numbers/symbols
        if re.fullmatch(r"[\d\s\W]+", line):
            continue
        clean.append(line)
    return "\n".join(clean)


# ──────────────────────────────────────────────
# Source 1: Annual Reports page
# ──────────────────────────────────────────────
def scrape_annual_reports():
    print("\n[1] Annual Reports Page")
    url = "https://www.iitj.ac.in/Main/en/Annual-Reports-of-the-Institute"
    html = fetch_html(url)
    if not html:
        return

    # Save page text
    page_text = clean_text(extract_html_text(html))
    save_raw("annual_reports_index.txt", page_text)

    # Find all PDF links on the page
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            if href.startswith("http"):
                pdf_links.append(href)
            elif href.startswith("/"):
                pdf_links.append("https://www.iitj.ac.in" + href)

    print(f"  Found {len(pdf_links)} PDF links")
    for i, pdf_url in enumerate(pdf_links[:10]):  # limit to 10 reports
        print(f"  Downloading PDF {i+1}/{min(len(pdf_links), 10)}: {pdf_url}")
        pdf_bytes = fetch_pdf_bytes(pdf_url)
        if not pdf_bytes:
            continue
        text = clean_text(extract_pdf_text(pdf_bytes))
        if len(text) > 100:
            fname = f"annual_report_{i+1:02d}.txt"
            save_raw(fname, text)
        time.sleep(1.5)


# ──────────────────────────────────────────────
# Source 2: HTML pages
# ──────────────────────────────────────────────
def scrape_html_pages():
    pages = {
        "vision_mission": "https://www.iitj.ac.in/main/en/vision-and-mission",
        "history":        "https://www.iitj.ac.in/main/en/history",
    }
    print("\n[2] HTML Pages")
    for fname, url in pages.items():
        print(f"  Scraping {url}")
        html = fetch_html(url)
        text = clean_text(extract_html_text(html))
        if text:
            save_raw(f"{fname}.txt", text)
        time.sleep(1.0)


# ──────────────────────────────────────────────
# Source 3: Specific PDFs
# ──────────────────────────────────────────────
def scrape_specific_pdfs():
    pdfs = {
        "iitj_brochure": "https://www.iitj.ac.in/PageImages/Gallery/03-2025/Brochure-1-638785941111940671.pdf",
        "iitj_newsletter_mar25": "https://www.iitj.ac.in/PageImages/Pages/03-2025/b414c72a-2b15-4374-b860-445eb84d4804.pdf",
        "iitj_newsletter_may25": "https://www.iitj.ac.in/PageImages/Pages/05-2025/a46666d4-3e4f-4a0b-929d-7d7e4aedae15.pdf",
        "senate_constitution": "https://www.iitj.ac.in/PageImages/Gallery/01-2026/Constitution-of-the-SenateIIT-JodhpurUpdated05012026-639032308491423512.pdf",
    }
    print("\n[3] Specific PDFs")
    for fname, url in pdfs.items():
        print(f"  Downloading {fname}: {url}")
        pdf_bytes = fetch_pdf_bytes(url)
        if not pdf_bytes:
            continue
        text = clean_text(extract_pdf_text(pdf_bytes))
        if text:
            save_raw(f"{fname}.txt", text)
        else:
            print(f"    ✗ No text extracted (possibly scanned/image PDF)")
        time.sleep(1.5)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SCRAPING ADDITIONAL IIT JODHPUR SOURCES")
    print("=" * 60)

    scrape_html_pages()
    scrape_specific_pdfs()
    scrape_annual_reports()

    print("\n✓ Scraping complete. Run preprocess.py to rebuild corpus.")


if __name__ == "__main__":
    main()
