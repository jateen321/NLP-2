# NLP Assignment 2 Report
**Name:** Jateen  
**Roll Number:** B22CS026  
**Problem 1: Learning Word Embeddings from IIT Jodhpur Data**  
**Problem 2: Character-Level Name Generation using RNN Variants**

---

## PROBLEM 1: Word Embeddings from IIT Jodhpur Data

### P1: Corpus Statistics
- **Size of the corpus file (corpus.txt):** 0.937 MB
- **Total number of documents:** 50
- **Total sentences:** 5,228
- **Total tokens:** 153,804
- **Vocabulary size:** 9,970

### P1: Curation and Preprocessing Steps
1. **Step-1: Data Collection**: Scraped 50 documents from IIT Jodhpur website 
   (Departments, Academics, Research, Faculty profiles, Announcements, 
   Annual Reports, Brochures, and Newsletters) using BeautifulSoup.
2. **Step-2: Boilerplate Removal**: Removed website navigation, headers, 
   footers, and non-English lines using an ASCII character proportion 
   heuristic (>70% ASCII = English).
3. **Step-3: Text Cleaning**: Removed URLs, email addresses, LaTeX fragments, 
   excessive punctuation, and non-textual artifacts using regular 
   expressions.
4. **Step-4: Case Normalization**: Converted all text to lower-case to ensure 
   consistency in the vocabulary.
5. **Step-5: Tokenization**: Tokenized the cleaned text into sentences and 
   then into words using the nltk library.
6. **Step-6: Frequency Filtering**: Filtered out short words and low-frequency 
   terms (frequency < 2) to reduce noise in the embeddings.

### P1: Word Embedding (300-dimensional vector for "student")
**Word:** student  
**Vector:**  
-0.4817, 0.3603, 0.0493, 0.4250, -0.3770, -0.0673, 0.0594, 0.3550, 0.4441, 0.0108, 
0.4734, 0.2026, -0.0270, 0.5809, -0.2244, -0.5872, 0.0789, -0.1344, -0.1896, -0.5505, 
0.3547, 0.5694, -0.2025, 0.3512, 0.2020, -0.0790, -0.4439, -0.6036, 0.4092, 0.0152, 
-0.8031, -0.1061, 0.2068, -0.0710, 0.1819, -0.1978, 0.1547, 0.2107, -0.0920, -0.2565, 
0.3691, -0.9547, -0.2578, -0.0465, 0.6128, -0.1441, 0.2610, -0.0176, -0.5609, 0.1823, 
-0.1211, -0.2968, -0.2824, -0.4080, 0.1438, -0.1003, 0.1738, -0.1842, 0.6324, 0.2406, 
0.1792, 0.3348, 0.2112, -0.3231, -0.0153, 0.3718, -0.3385, 0.1989, 0.0410, 1.1182, 
0.2469, 0.2355, -0.7243, -0.1846, -0.7783, -0.2690, -0.2737, 0.1185, -0.1831, 0.0040, 
-0.3267, 0.3263, 0.1461, -0.3609, -0.2161, -0.4403, 0.5465, -0.3295, -0.0014, -0.1480, 
-0.3047, -0.0644, -0.4163, -0.2091, -0.4770, -0.1277, -0.3629, 0.3530, 0.2578, 0.4835, 
0.2564, 0.3275, 0.7638, -0.5899, -0.1397, -0.1678, 0.3567, -0.2041, -0.4595, -0.1959, 
-0.0501, 0.2504, -0.0106, -0.1677, -0.4346, 0.0239, 0.1260, 0.1062, -0.7511, 0.0171, 
0.2365, 0.0898, -0.6174, -0.2059, -0.2231, 0.0275, -0.5134, 0.5352, 0.0127, 0.0232, 
-0.1253, -0.1755, 0.2097, -0.4404, -0.4178, -0.0805, -0.3315, -0.0120, -0.2659, -0.5955, 
0.0000, 0.1635, -0.2164, -0.0874, -0.1708, -0.0451, 0.7730, -0.0028, 0.0360, -0.3441, 
0.0187, -0.3389, 0.2444, 0.3140, -0.2221, -0.4685, -0.4509, 0.3447, 0.2380, -0.0195, 
-0.1323, -0.0985, -0.0388, 0.0161, -0.5808, 0.0477, 0.2680, -0.2245, 0.5469, 0.2117, 
-0.5573, 0.4815, 0.3396, -0.0683, 0.0381, -0.5191, 0.0503, -0.1327, -0.0704, 0.1641, 
-0.1939, -0.2861, -0.6366, 0.1738, -0.6682, -0.1642, -0.0663, -0.2858, -0.1228, -0.1536, 
-0.0461, -0.6755, -0.1673, -0.1956, -0.3070, -0.3313, -0.1935, 0.1770, 0.0497, -0.4135, 
0.5211, -0.6660, -0.3345, -0.0580, -0.5002, -0.3094, -0.1044, -0.2882, 0.0873, 0.3455, 
-0.2908, 0.1523, 0.1844, 0.3205, 0.2177, 0.3667, -0.1245, -0.2883, 0.5046, 0.6984, 
-0.4036, 0.1387, 0.5568, 0.0524, 0.2946, 0.1458, 0.5299, -0.0657, -0.7453, 0.0522, 
0.3259, 0.0074, 0.2894, 0.0353, -0.5590, 0.0846, 0.2497, -0.2185, 0.2897, -0.1043, 
-0.5683, 0.3164, -0.0470, 0.0958, -0.2009, -0.5172, -0.0821, -0.0654, 0.6564, -0.8608, 
0.1632, 0.1346, -0.0572, 0.0744, -0.2215, -0.0736, 0.1789, -0.0172, 0.0034, 0.3277, 
0.4270, -0.2283, -0.4266, 0.3323, 0.3555, 0.3139, -0.1971, -0.0703, -0.5721, 0.5277, 
0.2169, 0.2295, 0.0806, -0.0821, -0.0810, -0.0473, 0.0258, 0.2180, -1.0748, -0.3741, 
-0.4136, -0.2667, 0.0858, -0.1651, 0.4323, -0.6172, 0.3687, 0.5480, 0.1269, 0.4493, 
-0.1455, -0.6407, 0.5454, -0.4102, -0.3061, 0.0469, 0.3951, 0.2586, -0.2073, -0.1485

*(Note: Vector is from the trained Word2Vec Skip-gram model).*

### P1: Top-10 Words Frequency-wise
the, 6154, of, 4758, and, 4423, in, 2181, to, 2174, for, 1805, a, 1677, ug, 1014, 
iit, 951, students, 951

### P1: Interesting Analogy
**semester : examination :: thesis : defense**
*Observation: Just as a semester culminates in an examination, a thesis 
culminates in a defense.*

---

## PROBLEM 2: Character-Level Name Generation

### P2: Quantitative Performance Comparison
| Model | Novelty Rate | Diversity | Parameters |
| :--- | :--- | :--- | :--- |
| **Vanilla RNN** | 24.6% | 80.4% | 61,393 |
| **Bi-LSTM** | 100.0% | 97.6% | 442,193 |
| **Attention + RNN** | 100.0% | 100.0% | 271,185 |

### P2: Evaluation Results
**Which model works better?**
For this specific task of character-level Indian name generation on a small 
dataset (1000 names), **Vanilla RNN** worked significantly better in terms 
of qualitative realism.
- **Vanilla RNN** produces realistic, pronounceable Indian names like 
  *Chaitanya*, *Namit*, and *Vasuki*.
- **Bi-LSTM** overfitted rapidly (training loss near zero) and generated novel 
  but somewhat random strings (e.g., *Ojw*, *Objs*).
- **Attention + RNN** captured complex structures but produced stylized 
  strings on such a small training set.

**Conclusion:** Vanilla RNN is the most effective for this limited dataset as 
it captures local character transitions robustly without over-parameterization 
issues.

### P2: Vanilla RNN Details
- **Number of Parameters:** 61,393
- **Model Size:** 0.24 MB on disk

---

## Deliverables
- **GitHub Repository:** [Link provided in submission]
- **corpus.txt:** Preprocessed text for Problem 1
- **report.pdf:** This document

---
*(End of Report)*
