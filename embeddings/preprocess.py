"""
preprocess.py - Corpus Cleaning & Preprocessing
=================================================
Processes raw scraped text into a clean corpus for Word2Vec training.

Steps:
1. Load all raw text files
2. Remove non-English text
3. Remove boilerplate, URLs, emails, formatting artifacts
4. Tokenize and lowercase
5. Remove excessive punctuation, numbers-only tokens
6. Generate dataset statistics and word cloud

Usage:
    python embeddings/preprocess.py

Output:
    embeddings/corpus.txt           — cleaned corpus (one sentence per line)
    embeddings/figures/wordcloud.png — word cloud of most frequent words
"""

import os
import re
import json
import string
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "corpus.txt")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Stopwords to remove from word cloud (but keep in corpus for context)
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# Additional IIT-specific stopwords for cleaner word cloud
EXTRA_STOPWORDS = {
    "iit", "jodhpur", "iitj", "ac", "en", "www", "http", "https",
    "page", "click", "view", "home", "menu", "close", "skip",
    "read", "main", "also", "one", "two", "may", "shall", "will",
    "nbsp", "amp", "quot", "gt", "lt",
}


def load_raw_texts():
    """Load all raw text files from the raw directory."""
    documents = []
    
    if not os.path.exists(RAW_DIR):
        print(f"  ✗ Raw directory not found: {RAW_DIR}")
        print("  Please run scraper.py first!")
        return documents
    
    for filename in sorted(os.listdir(RAW_DIR)):
        if filename.startswith("_") or not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        if len(text.strip()) > 0:
            documents.append({
                "source": filename.replace(".txt", ""),
                "text": text
            })
    
    return documents


def is_english_line(line):
    """
    Heuristic: a line is English if ≥60% of its characters are ASCII letters/spaces.
    Filters out Hindi/Devanagari and other non-English text.
    """
    if not line.strip():
        return False
    ascii_chars = sum(1 for c in line if ord(c) < 128)
    return (ascii_chars / len(line)) >= 0.6


def clean_text(text):
    """
    Clean a single document's text:
    - Remove non-English lines
    - Remove URLs, email addresses
    - Remove formatting artifacts
    - Remove excessive whitespace/newlines
    """
    lines = text.split("\n")
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip non-English lines
        if not is_english_line(line):
            continue
        
        # Remove URLs
        line = re.sub(r"https?://\S+", " ", line)
        line = re.sub(r"www\.\S+", " ", line)
        
        # Remove email addresses
        line = re.sub(r"\S+@\S+\.\S+", " ", line)
        
        # Remove phone numbers
        line = re.sub(r"[\+]?[\d\s\-\(\)]{7,15}", " ", line)
        
        # Remove HTML entities that survived
        line = re.sub(r"&\w+;", " ", line)
        line = re.sub(r"&#\d+;", " ", line)
        
        # Remove special Unicode characters
        line = re.sub(r"[^\x00-\x7F]+", " ", line)
        
        # Remove excessive punctuation (3+ repeated)
        line = re.sub(r"[.\-_=*#]{3,}", " ", line)
        
        # Remove standalone numbers/dates that aren't informative
        # (but keep numbers embedded in text like "4 semesters")
        
        # Normalize whitespace
        line = re.sub(r"\s+", " ", line).strip()
        
        # Skip very short lines (likely artifacts)
        if len(line) < 10:
            continue
        
        clean_lines.append(line)
    
    return " ".join(clean_lines)


def tokenize_to_sentences(text):
    """Split cleaned text into sentences, then tokenize each sentence."""
    sentences = sent_tokenize(text)
    tokenized_sentences = []
    
    for sent in sentences:
        # Tokenize
        tokens = word_tokenize(sent.lower())
        
        # Filter tokens
        filtered = []
        for token in tokens:
            # Skip pure punctuation
            if all(c in string.punctuation for c in token):
                continue
            # Skip pure numbers
            if token.isdigit():
                continue
            # Skip very short tokens (single chars except 'a', 'i')
            if len(token) == 1 and token not in ("a", "i"):
                continue
            # Skip tokens that are mostly numbers with some letters
            if re.match(r"^\d+[a-z]{0,2}$", token):
                continue
            
            filtered.append(token)
        
        # Only keep sentences with at least 3 meaningful tokens
        if len(filtered) >= 3:
            tokenized_sentences.append(filtered)
    
    return tokenized_sentences


def compute_statistics(tokenized_sentences, documents):
    """Compute and print dataset statistics."""
    all_tokens = [token for sent in tokenized_sentences for token in sent]
    vocab = set(all_tokens)
    token_freq = Counter(all_tokens)
    
    stats = {
        "total_documents": len(documents),
        "total_sentences": len(tokenized_sentences),
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(vocab),
        "avg_sentence_length": round(len(all_tokens) / max(len(tokenized_sentences), 1), 2),
        "top_20_words": token_freq.most_common(20),
    }
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total documents (scraped pages):  {stats['total_documents']}")
    print(f"  Total sentences:                  {stats['total_sentences']:,}")
    print(f"  Total tokens:                     {stats['total_tokens']:,}")
    print(f"  Vocabulary size (unique tokens):  {stats['vocabulary_size']:,}")
    print(f"  Average sentence length:          {stats['avg_sentence_length']} tokens")
    print(f"\n  Top 20 most frequent words:")
    for word, count in stats["top_20_words"]:
        print(f"    {word:20s} → {count:,}")
    
    return stats, token_freq


def generate_wordcloud(token_freq, output_path):
    """Generate and save a word cloud from token frequencies."""
    # Filter out stopwords for the word cloud
    filtered_freq = {
        word: count for word, count in token_freq.items()
        if word not in STOPWORDS and word not in EXTRA_STOPWORDS and len(word) > 2
    }
    
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=150,
        min_font_size=8,
        max_font_size=120,
        relative_scaling=0.5,
        contour_width=2,
        contour_color="steelblue",
    )
    wc.generate_from_frequencies(filtered_freq)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — IIT Jodhpur Corpus (Most Frequent Words)",
                 fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Word cloud saved to: {output_path}")


def main():
    print("=" * 60)
    print("CORPUS PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw texts
    print("\n[1/5] Loading raw text files...")
    documents = load_raw_texts()
    print(f"  Loaded {len(documents)} documents")
    
    if not documents:
        print("\n  ERROR: No documents found. Run scraper.py first!")
        return
    
    # Step 2: Clean texts
    print("\n[2/5] Cleaning texts...")
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_text(doc["text"])
        if cleaned:
            cleaned_docs.append(cleaned)
            print(f"  ✓ {doc['source']:40s} → {len(cleaned):,} chars")
        else:
            print(f"  ✗ {doc['source']:40s} → empty after cleaning")
    
    # Step 3: Tokenize into sentences
    print("\n[3/5] Tokenizing into sentences...")
    all_sentences = []
    for text in cleaned_docs:
        sents = tokenize_to_sentences(text)
        all_sentences.extend(sents)
    print(f"  Generated {len(all_sentences):,} sentences")
    
    # Step 4: Save corpus
    print("\n[4/5] Saving corpus...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(" ".join(sent) + "\n")
    print(f"  ✓ Corpus saved to: {CORPUS_FILE}")
    print(f"  File size: {os.path.getsize(CORPUS_FILE):,} bytes")
    
    # Step 5: Statistics and word cloud
    print("\n[5/5] Computing statistics and generating word cloud...")
    stats, token_freq = compute_statistics(all_sentences, documents)
    
    wordcloud_path = os.path.join(FIGURES_DIR, "wordcloud.png")
    generate_wordcloud(token_freq, wordcloud_path)
    
    # Save stats to JSON
    stats_path = os.path.join(os.path.dirname(__file__), "dataset_stats.json")
    stats_serializable = {k: v for k, v in stats.items() if k != "top_20_words"}
    stats_serializable["top_20_words"] = [{"word": w, "count": c} for w, c in stats["top_20_words"]]
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"  ✓ Statistics saved to: {stats_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
