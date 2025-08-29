# 📊 Ambiguity in Earnings Calls and Market Reactions  

## 🔎 Overview  
This project investigates whether **linguistic ambiguity** in corporate earnings calls can predict **future stock price volatility** and **post-announcement returns**.  

The intuition:  
- Confident management → clear, direct language → more stable market reactions
- Uncertain management → vague, hedging language (“may”, “might”, “uncertain”) → higher volatility and stronger surprises

Inspired by prior research on linguistic complexity and investor behavior:  
- *Alduais et al. (2022)* — showing that higher disclosure complexity is associated with lower stock returns
- *Winchel (2013)* — exploring how investors react to ambiguous language and mixed positive/negative arguments in analyst reports

---

## ⚙️ Project Pipeline 

1. **📥 Data Collection**  
   - Earnings call transcripts (SeekingAlpha / Kaggle)  
   - Historical stock prices (Yahoo Finance)  

2. **🧹 Preprocessing**  
   - Clean transcripts (remove speaker labels, punctuation, stopwords)  
   - Align transcripts with earnings announcement dates  

3. **📝 Feature Engineering**  
   - **Hedging words count**: frequency of terms like *“may”, “might”, “possibly”*  
   - **Readability scores**: Flesch-Kincaid, Gunning Fog (via `textstat`)  
   - **Semantic ambiguity**: embedding variance (via `sentence-transformers`)  

4. **📈 Signal Construction**  
   - Compute an ambiguity score per call  
   - Align with post-call realized volatility and abnormal returns  

5. **💹 Backtesting**  
   - Long/short portfolio: long low-ambiguity calls vs short high-ambiguity calls  
   - Performance metrics: Sharpe ratio, alpha, drawdown  

6. **📊 Visualization & Results**  
   - Distribution of ambiguity scores  
   - Relationship between ambiguity and volatility  
   - Portfolio equity curve  

---

## 🛠️ Tech Stack  
- **NLP**: spaCy, textstat, sentence-transformers  
- **Data**: pandas, numpy, yfinance  
- **Backtesting & Stats**: statsmodels, scipy  
- **Visualization**: matplotlib, seaborn  

---

## 📂 Repository Structure  

```
nlp-earnings-ambiguity/
│
├── data/
│ ├── transcripts.csv
│ ├── prices.csv
│
├── notebooks/ # Jupyter notebooks for prototyping
│ ├── 01_data_cleaning.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_backtest.ipynb
│
├── src/ # modular scripts
│ ├── text_cleaner.py   # TranscriptCleaner class
│ ├── data_loader.py    # Load cleaned transcripts into DataFrame
│ ├── feature_extractor.py # Extract linguistic features
│ ├── backtester.py     # Extract NLP features and ambiguity scores
│
├── results/ # figures, tables, metrics
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Results (to be added)  
- 📊 Correlation between ambiguity and post-call volatility  
- 📈 Long/short portfolio performance  
- 🔍 Interpretability of text-based features  

---

## 🧭 Next Steps  
- Extend features with modern embeddings (FinBERT, LLMs).  
- Test robustness across sectors and firm sizes.  
- Compare textual ambiguity with implied volatility from options.  

---
