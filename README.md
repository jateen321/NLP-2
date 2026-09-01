# Word Embeddings & Character-Level Text Generation

Two NLP systems built from first principles in PyTorch: **Word2Vec** (CBOW and
Skip-gram with negative sampling) trained on a corpus I scraped and cleaned
myself, and a **character-level name generator** comparing three recurrent
architectures whose cells are implemented by hand — no `nn.RNN`, no `nn.LSTM`,
no `gensim` in the from-scratch path.

The goal was not to beat a benchmark. It was to implement the algorithms at the
level where the gradients live, and then to measure honestly what the
implementations actually do.

---

## Headline findings

**A 61K-parameter vanilla RNN beat a 442K-parameter Bi-LSTM at generating
realistic names.** Not marginally — the Bi-LSTM produced zero-vowel strings like
`Gokgs` and `Hgoy` in 25 of 500 samples, while the vanilla RNN produced none and
generated pronounceable names like `Chaitanya` and `Vasuki`. Three causes,
diagnosed in [`RESULTS.md`](RESULTS.md): capacity–data mismatch on a 1,000-sample
set, exposure bias amplified by teacher forcing, and the fact that
**bidirectionality is structurally incompatible with autoregressive generation**
— backward context exists during training and cannot exist at inference.

**Novelty and diversity are misleading metrics in isolation.** The Bi-LSTM scored
100% novelty and 97.6% diversity; its output was nonsense. The vanilla RNN scored
24.6% novelty and produced the only usable names. A generator that never repeats
its training set is not thereby good — it may simply be failing in new ways each
time.

**A from-scratch implementation validated against a mature library.** My PyTorch
CBOW recovers nearly the same nearest-neighbour ordering for `engineering` as
Gensim (`computer`, `metallurgical`, `electrical`) at lower absolute similarity —
the expected signature of a correct implementation without Gensim's subsampling
and optimised C training loop.

---

## Part 1 — Word2Vec from scratch, on a corpus I built

**The data.** No pre-packaged dataset. `scraper.py` and `scrape_extra.py` pull 50
documents from a university web domain — annual reports, course brochures,
academic regulations, a senate constitution, faculty pages, newsletters.
`preprocess.py` cleans and sentence-tokenizes them into **5,228 sentences /
153,804 tokens / 15,822-word vocabulary** (0.94 MB).

**The implementation.** `word2vec_scratch.py` builds the whole stack:

| Component | What it does |
|---|---|
| `Vocabulary` | word↔index mapping; unigram noise distribution `P(w) ∝ f(w)^0.75` for negative sampling |
| `CBOWDataset` / `SkipGramDataset` | emit `(context, target, negatives)` tuples — 14,564 and 212,518 pairs respectively |
| `CBOWModel` / `SkipGramModel` | separate input/output embedding tables; forward pass computes negative-sampling BCE loss directly |
| training loop | Adam, 50 epochs, gradient clipping |

Negative sampling replaces the full softmax over 15,822 words, cutting per-step
cost from `O(V)` to `O(k)`.

**The baseline.** `train_word2vec.py` runs a **36-configuration Gensim grid
search** over embedding dimension (50/100/200), window (3/5/7), and negative
samples (5/10), scored by mean cosine similarity of top-5 neighbours across fixed
query words. Best CBOW: `dim=200, window=5, neg=5` (0.855). Best Skip-gram:
`dim=50, window=3, neg=5` (0.735).

CBOW wins that metric — but the metric rewards tight neighbourhoods, which is
CBOW's bias, not evidence of better embeddings. Skip-gram resolved the analogy
tasks (`UG:B.Tech :: PG:M.Tech`) that CBOW did not.

**Analysis.** `analysis.py` and `visualize.py` produce nearest-neighbour tables,
analogy resolution, and PCA / t-SNE projections comparing all four models
(scratch × library, CBOW × Skip-gram).

## Part 2 — Three recurrent architectures, hand-built

`models.py` implements every cell manually:

- **Vanilla RNN** — `h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)`, Xavier-initialised, stacked 2 deep. 61,393 params.
- **Bi-LSTM** — all four gates written out (input, forget, candidate, output), forward and backward passes concatenated to 256-wide. 442,193 params.
- **RNN + Luong attention** — `score(h_t, h_j) = h_tᵀ W_attn h_j` over all previous positions, softmax-weighted context vector concatenated with the hidden state. 271,185 params.

`train.py` trains all three with teacher forcing and StepLR decay; `evaluate.py`
samples 500 names per model with temperature-controlled softmax and scores
novelty, diversity, and five categorized failure modes (too short, too long,
repetitive, nonsense, no vowels).

### Results

| Model | Params | Novelty | Diversity | Failures / 500 | Realistic? |
|---|---|---|---|---|---|
| **Vanilla RNN** | 61,393 | 24.6% | 80.4% | **3** | **Yes** |
| Bi-LSTM | 442,193 | 100.0% | 97.6% | 34 | No |
| RNN + Attention | 271,185 | 100.0% | 100.0% | 101 | No |

The Bi-LSTM drove training loss to 0.003 — it memorised the training
distribution, then fell apart the moment autoregressive sampling introduced
inputs it had never been forced to recover from.

### A note on the name dataset

**The 1,000 training names are programmatically assembled, not scraped or
downloaded.** `generate_names.py` draws from curated lists of real Indian names
spanning North Indian, South Indian, Bengali, Gujarati, and Marathi pools, plus
names constructed from common Indian phoneme prefixes and suffixes, with a fixed
seed for reproducibility. This is stated plainly because it bounds what the
Part 2 results mean: the models learn the phonotactics of that constructed
distribution, and the small, clean dataset is precisely what makes the
capacity–data mismatch finding visible.

---

## Repository layout

```
embeddings/
  scraper.py  scrape_extra.py   # corpus collection (50 documents)
  preprocess.py                 # cleaning, tokenization, corpus stats
  train_word2vec.py             # Gensim baseline + 36-config grid search
  word2vec_scratch.py           # CBOW & Skip-gram from scratch (PyTorch)
  analysis.py  visualize.py     # neighbours, analogies, PCA / t-SNE
  raw/  corpus.txt  models/  figures/

name_generation/
  generate_names.py             # builds the 1,000-name dataset (see note above)
  models.py                     # VanillaRNN, BiLSTM, AttentionRNN — manual cells
  train.py  evaluate.py         # training pipeline; novelty/diversity/failure metrics
  checkpoints/  figures/

tests/                          # shape, gradient, and sampler correctness checks
RESULTS.md                      # full technical writeup: theory, tables, discussion
```

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Part 1
python embeddings/scraper.py          # or skip — raw/ and corpus.txt are committed
python embeddings/preprocess.py
python embeddings/train_word2vec.py
python embeddings/word2vec_scratch.py
python embeddings/analysis.py
python embeddings/visualize.py

# Part 2
python name_generation/generate_names.py
python name_generation/train.py
python name_generation/evaluate.py

# Tests
pytest tests/ -v
```

## What I'd do differently

- **153K tokens is a small corpus for Word2Vec**, and the embeddings show it —
  Gensim CBOW similarities cluster around 0.97 for almost any query, a classic
  anisotropy symptom. Nearest neighbours for `research` include tokens like
  `'s` and `26.3`. More aggressive frequency filtering and a corpus an order of
  magnitude larger would both help.
- **Scheduled sampling** instead of pure teacher forcing would directly target
  the exposure bias that sank the Bi-LSTM.
- **A learned realism metric** — the failure-mode heuristics catch vowel-less
  strings, but a character n-gram likelihood under a held-out language model
  would score realism continuously rather than by hand-written rules.
