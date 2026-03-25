# NLP Assignment 2: Word Embeddings & Name Generation
**Course:** NLP (IIT Jodhpur)  
**Student:** Jateen (B22CS026)

## Project Structure
- `problem1/`: Word2Vec implementation (Gensim vs. Scratch)
    - `scraper.py`: Scrapes IIT Jodhpur website data.
    - `preprocess.py`: Cleans and tokenizes the corpus.
    - `train_word2vec.py`: Hyperparameter tuning with Gensim.
    - `word2vec_scratch.py`: PyTorch implementation of CBOW and Skip-gram.
    - `analysis.py`: Semantic analysis (Nearest neighbors, analogies).
    - `visualize.py`: PCA and t-SNE visualizations.
    - `corpus.txt`: The final 172KB cleaned corpus.
- `problem2/`: Character-Level Name Generation
    - `generate_names.py`: Generates 1000 Indian names for training.
    - `models.py`: From-scratch implementations of RNN, BLSTM, and Attention RNN.
    - `train.py`: Training pipeline for all three models.
    - `evaluate.py`: Generates names and computes Novelty/Diversity metrics.
- `report.md`: Detailed assignment report.
- `requirements.txt`: Python dependencies.

## How to Run
1. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Problem 1**:
   ```bash
   # Run scraping and preprocessing
   python problem1/scraper.py
   python problem1/preprocess.py
   
   # Train and analyze (Gensim)
   python problem1/train_word2vec.py
   python problem1/analysis.py
   python problem1/visualize.py
   
   # Run from-scratch implementation and comparison
   python problem1/word2vec_scratch.py
   ```

3. **Problem 2**:
   ```bash
   # Generate training data
   python problem2/generate_names.py
   
   # Train all models
   python problem2/train.py
   
   # Evaluate performance
   python problem2/evaluate.py
   ```

## Key Results
- **Problem 1**: Skip-gram captured more meaningful semantic clusters in the IITJ domain compared to CBOW.
- **Problem 2**: Vanilla RNN proved most effective for realistic name generation on the limited Indian names dataset, outperforming the more complex Bi-LSTM and Attention variants which tended to overfit.
