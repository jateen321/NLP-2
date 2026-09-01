# Technical Report

Full write-up of method and results for both parts of the project. See
[`README.md`](README.md) for the summary.

---

# Part 1 — Word Embeddings

## 1.1 Corpus construction

The corpus was built rather than downloaded. `scraper.py` and `scrape_extra.py`
collect 50 documents from a university web domain, deliberately spanning
registers: annual reports, a course brochure, academic regulations, the senate
constitution, faculty and department pages, news, events, and a newsletter.

An early version of the corpus used only the main site pages (~0.17 MB). Adding
long-form documents expanded it roughly 6× and materially improved both
vocabulary coverage and neighbourhood quality — a reminder that corpus design is
not a preliminary to modelling but part of it.

`preprocess.py` lowercases, strips boilerplate and navigation text, sentence-
tokenizes, and writes one sentence per line.

| Statistic | Value |
|---|---|
| Documents | 50 |
| Sentences | 5,228 |
| Tokens | 153,804 |
| Vocabulary | 15,822 |
| Avg. sentence length | 29.4 tokens |
| Corpus size | 0.94 MB |

## 1.2 Theory

**CBOW** predicts a target word from its surrounding context. Given context
`{w_{t-k}, …, w_{t-1}, w_{t+1}, …, w_{t+k}}`, the model averages their embeddings
and predicts `w_t`, maximising `J = log P(w_t | context)`. It trains faster and
handles frequent words well, but averaging smooths over individual context words.

**Skip-gram** inverts this: given `w_t`, predict each context word, maximising
`J = Σ_j log P(w_{t+j} | w_t)`. It is slower — every (target, context) pair is a
separate training example — but represents rare words better for exactly that
reason.

**Negative sampling.** A full softmax over 15,822 words at every step is
wasteful. Negative sampling replaces it with binary classification: separate the
true context word from `k` noise words drawn from `P(w) ∝ f(w)^{3/4}`. Per-step
cost drops from `O(V)` to `O(k)`. The `3/4` exponent flattens the unigram
distribution so that frequent words are sampled as negatives less often than
their raw frequency would dictate.

## 1.3 Hyperparameter search (Gensim baseline)

Grid over embedding dimension (50, 100, 200) × window (3, 5, 7) × negative
samples (5, 10), for both architectures — 36 configurations, 10 epochs each,
`min_count=2`. Scored by mean cosine similarity of top-5 neighbours over a fixed
query set.

| Model | Embed dim | Window | Negatives | Avg. similarity |
|---|---|---|---|---|
| **CBOW (best)** | 200 | 5 | 5 | **0.8550** |
| CBOW | 100 | 5 | 5 | 0.8513 |
| CBOW | 50 | 7 | 5 | 0.8482 |
| Skip-gram | 50 | 7 | 10 | 0.7400 |
| **Skip-gram (best in-family)** | 50 | 3 | 5 | 0.7347 |
| Skip-gram | 200 | 3 | 5 | 0.6627 |

CBOW dominates this metric — but the metric measures neighbourhood compactness,
which is CBOW's structural bias. It is not evidence that CBOW learned better
representations. Skip-gram's advantage appears on relational tasks instead
(§1.5), which this score does not capture. Choosing a scoring function that
favours one architecture and then declaring that architecture the winner is a
trap worth naming explicitly.

## 1.4 From-scratch implementation

`word2vec_scratch.py` implements Word2Vec end-to-end in PyTorch with no NLP
libraries:

- **`Vocabulary`** — word↔index mapping, frequency counts, and the `f(w)^{3/4}`
  noise distribution used to draw negatives.
- **`CBOWDataset` / `SkipGramDataset`** — `torch.utils.data.Dataset` subclasses
  emitting `(context, target, negatives)`. CBOW yields 14,564 training pairs;
  Skip-gram yields 212,518, since each target–context pair is independent.
- **`CBOWModel` / `SkipGramModel`** — two embedding tables (input and output).
  The forward pass computes the negative-sampling binary cross-entropy directly:
  positives labelled 1, negatives labelled 0.
- **Training** — Adam, 50 epochs, gradient clipping.

Keeping input and output embeddings separate matters: the same word plays two
roles (as a centre and as a context), and tying the tables forces a word to be
its own nearest neighbour. Input embeddings are used at inference.

## 1.5 Results

**Nearest neighbours (Skip-gram, dim=200)**

| Query | Neighbours |
|---|---|
| `research` | activities, proposal, interdisciplinary, development, areas |
| `engineering` | metallurgical, bioengineering, materials, computer, bioscience |
| `student` | candidate, assistantship, advise, until, visiting |
| `examination` | comprehensive, viva-voce, make-up, examiners, examinations |
| `phd` | mtech, m.tech./m.tech.-ph.d., dual, governing |

**Analogies resolved**

- `UG : B.Tech :: PG : M.Tech` ✓
- `semester : examination :: thesis : defense` ✓ (viva-voce / defense as terminal assessment)
- `student : learning :: faculty : teaching` ✓

**Scratch vs. Gensim** — nearest neighbours for `engineering`:

| Implementation | Top neighbours |
|---|---|
| Scratch CBOW | computer (0.938), metallurgical (0.920), electrical (0.913), materials (0.902) |
| Gensim CBOW | science (0.967), computer (0.958), metallurgical (0.949), electrical (0.947) |
| Scratch Skip-gram | metallurgical (0.607), bridge (0.593), bioengineering (0.570) |
| Gensim Skip-gram | metallurgical (0.960), bioengineering (0.959), materials (0.954) |

The scratch CBOW recovers almost exactly Gensim's ordering at systematically
lower absolute similarity. That pattern — right ranking, compressed scale — is
what a correct implementation without frequency subsampling and an optimised
training loop should look like, and is the main evidence that the from-scratch
code is sound.

On the analogy `thesis : examination :: thesis : ?`, scratch CBOW returns
`submit, defend, synopsis`; Gensim returns `src, reported, dugc`.

## 1.6 Honest limitations

- **The corpus is small.** Gensim CBOW similarities sit around 0.97 for nearly
  any query — the embedding space is anisotropic, everything is close to
  everything. Nearest neighbours for `research` include `'s` and `26.3`.
- **Domain vocabulary breaks standard evaluation.** Tokens like `viva-voce`,
  `dugc`, and `cr` are meaningful in this corpus and absent from every standard
  analogy benchmark, so analogy triplets had to be hand-selected from
  in-vocabulary words — which weakens them as an objective measure.
- **The scoring function is biased**, as discussed in §1.3.

---

# Part 2 — Character-Level Name Generation

## 2.1 Task and dataset

Learn `P(c_t | c_1, …, c_{t-1})` over characters and generate names
autoregressively: sample from a temperature-controlled softmax, feed the sampled
character back as the next input, stop at `<EOS>`.

**Dataset provenance — stated plainly.** The 1,000 names are *programmatically
assembled*, not scraped or downloaded. `generate_names.py` draws from curated
lists of real Indian names across North Indian, South Indian, Bengali, Gujarati,
Marathi, and pan-Indian pools, supplemented by names constructed from common
Indian phoneme prefixes and suffixes, under a fixed seed (42) for
reproducibility. Average length 6.1 characters, range 2–16, 49-character
vocabulary.

This bounds the results: the models learn the phonotactics of a clean,
constructed distribution. It also enables the central finding — a small, tidy
dataset is exactly the regime where over-capacity shows up clearly.

## 2.2 Architectures

All three are implemented with manual cell computations. `nn.RNN`, `nn.LSTM`,
and `nn.RNNCell` are not used anywhere.

**Vanilla RNN.** `h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b_h)`, Xavier-initialised,
2 stacked layers.
Embedding (49×32) → 2× RNN cell (hidden 128) → Linear (128→49). 61,393 params.
Vanishing gradients are the known weakness, but at an average length of 6
characters there is little depth for gradients to vanish through.

**Bi-LSTM.** Gating lets the cell state carry gradients through
element-wise multiplication:

```
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)     input gate    — what to write
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)     forget gate   — what to erase
g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)  candidate values
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)     output gate   — what to expose
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

A backward LSTM runs right-to-left; forward and backward states concatenate to
256 dimensions. 442,193 params. **This is architecturally mismatched to the
task** — see §2.5.

**RNN + Luong attention.** At step `t`, over all `j < t`:

```
score(h_t, h_j) = h_tᵀ · W_attn · h_j
α               = softmax(scores)
context_t       = Σ_j α_j · h_j
output_t        = tanh(W_c · [context_t ; h_t])
```

271,185 params. Intended to let the model attend to recurring suffixes (`-esh`,
`-raj`, `-iya`) and vowel placement.

## 2.3 Training configuration

| Hyperparameter | Vanilla RNN | Bi-LSTM | RNN + Attention |
|---|---|---|---|
| Embedding dim | 32 | 32 | 32 |
| Hidden size | 128 | 128 | 128 |
| Layers | 2 | 2 | 2 |
| Dropout | 0.20 | 0.40 | 0.30 |
| Learning rate | 0.003 | 0.003 | 0.003 |
| Epochs | 100 | 60 | 60 |
| LR scheduler | StepLR (γ=0.5, step=30) | StepLR (γ=0.5, step=20) | StepLR (γ=0.5, step=20) |
| Final training loss | 1.137 | **0.003** | 1.510 |
| Parameters | 61,393 | 442,193 | 271,185 |
| Checkpoint size | 0.24 MB | 1.69 MB | 1.04 MB |

The Bi-LSTM's training loss of 0.003 is not a success. It is the diagnosis.

## 2.4 Evaluation

500 names sampled per model.
*Novelty* = share not present in the training set. *Diversity* = unique / total.

| Model | Novelty | Diversity | Params |
|---|---|---|---|
| Vanilla RNN | 24.6% | 80.4% | 61,393 |
| Bi-LSTM | 100.0% | 97.6% | 442,193 |
| RNN + Attention | 100.0% | 100.0% | 271,185 |

**Failure modes (count per 500 samples)**

| Failure mode | Vanilla RNN | Bi-LSTM | RNN + Attention |
|---|---|---|---|
| Too short (< 2 chars) | 0 | 5 | 2 |
| Repetitive patterns | 3 | 4 | 15 |
| No vowels | **0** | 25 | 84 |
| **Total** | **3** | 34 | 101 |

**Samples**

- *Vanilla RNN:* Chaitanya, Namit, Vasuki, Sahan, Shubh, Devi, Bhina, Pashay, Sakar — pronounceable, plausibly Indian.
- *Bi-LSTM:* Ojw, Yegi, Tookst, Yotas, Gokgs, Hgoy — novel, but not names.
- *RNN + Attention:* highly diverse, structurally inconsistent, no clear phonotactic pattern.

The "no vowels" column is the most diagnostic number in the project. Every Indian
name contains a vowel; a model that emits 84 vowel-less strings out of 500 has
not learned the most basic constraint of the distribution it was trained on.

## 2.5 Discussion — why the simplest model won

**Capacity–data mismatch.** 442K parameters against 1,000 short sequences. The
Bi-LSTM had more than enough capacity to memorise the training set outright, and
its loss of 0.003 says it did. Memorisation is fine while ground truth is fed in;
it collapses the moment the model must condition on its own output.

**Exposure bias.** Teacher forcing trains on ground-truth prefixes only. At
generation time the model conditions on its own samples, which drift. A
memorising model has never seen a slightly-wrong prefix and has no learned
behaviour for recovering from one, so small deviations cascade. The
lower-capacity RNN, unable to memorise, was forced to learn smoother statistical
regularities that degrade gracefully.

**Bidirectionality is incompatible with autoregressive generation.** This is
structural, not a tuning problem. The backward LSTM consumes future characters —
information that exists during teacher-forced training and *cannot* exist at
inference, when future characters are precisely what is being generated. The
model is trained with strictly more information than it will ever have at test
time. No amount of further training fixes this; the architecture is wrong for
the task, and including it was instructive precisely because it fails in a way
that is explainable from first principles.

**Attention needs sequence length to pay for itself.** With an average of 6
characters, there are at most 5 previous positions to attend over. Luong
attention adds 210K parameters to exploit long-range structure that short names
do not contain — cost without corresponding benefit, and more capacity to
overfit with.

**Novelty is not quality.** 100% novelty means only that the model never
reproduces a training name exactly. If the novel outputs are structurally
invalid, the metric is actively misleading. Realism requires learning
phonotactics, which only the vanilla RNN did. This is the practical lesson:
automatic metrics for generative models must be paired with qualitative
inspection, or they will confidently rank nonsense first.

## 2.6 Conclusions

**Part 1**
- Corpus design is modelling. Expanding source diversity ~6× improved embedding
  quality more than any hyperparameter change.
- CBOW (dim=200, win=5) wins average-similarity (0.855); Skip-gram wins
  analogies. The gap is a property of the metric, not a ranking of the models.
- The from-scratch implementation reproduces Gensim's qualitative neighbourhood
  structure, validating correctness.
- Domain-specific vocabulary makes standard analogy evaluation inapplicable.

**Part 2**
- Match model capacity to dataset size; on small data, simpler models generalise
  better.
- Teacher forcing creates a train/inference mismatch that hurts high-capacity
  models disproportionately.
- Bidirectionality cannot be used for autoregressive generation.
- Attention pays off on long-range dependencies, not 6-character sequences.
- Quantitative metrics and qualitative realism can point in opposite directions.

## 2.7 Future work

- **Scheduled sampling** — anneal from teacher forcing to self-conditioning
  during training to attack exposure bias directly.
- **A larger, real corpus** for Part 1 to address the anisotropy in the embedding
  space.
- **A learned realism metric** — held-out character n-gram likelihood, replacing
  hand-written failure heuristics with a continuous score.
- **Unidirectional LSTM** as the proper capacity comparison against the vanilla
  RNN, isolating capacity from the bidirectionality confound.
