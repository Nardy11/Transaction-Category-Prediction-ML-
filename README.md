```markdown
✅ README – Transaction Category Prediction (ML Final Project)
By Nardy Attaalla – ReDI School Machine Learning Course  
Project Files Reference: FINAL PROJECT Presentation 08e16aac-4046-4d83-a9a7-9554c00…

## 📌 1. Project Overview
This project automatically classifies customer transactions into product categories using a hybrid approach combining:

- NLP (TF-IDF)
- Unsupervised Learning (KMeans Clustering)
- Supervised Learning (Logistic Regression)

The dataset contains retail transactions but no category labels, so the solution generates its own categories through clustering, then trains a classifier to predict them for new unseen transactions.  
This reflects real applications in fintech, banking, and expense-tracking apps such as Revolut, Monzo, and N26.

## 📌 2. Problem Statement
Retail transaction data rarely comes with clean product categories.  
The goal of this project is to:

- Discover meaningful product categories automatically (unsupervised clustering)
- Use these discovered categories to train a classifier
- Predict categories for new transactions in real time

This helps financial applications summarize spending, detect unusual purchases, and assist users in money management.

## 📌 3. Dataset Information
**Source:** Kaggle – Transaction Data  
https://www.kaggle.com/datasets/vipin20/transaction-data

**Rows:** ~540K

**Columns:**
- UserId
- ItemDescription
- NumberOfItemsPurchased
- CostPerItem
- Country

**Challenges:**
- No category labels
- Duplicate entries
- Outliers
- Very skewed country distribution (90% UK)
- Text is messy, requires NLP preprocessing

## 📌 4. Installation & Requirements
**Required Libraries**
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
```

**Install Libraries**
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- scipy  

## 📌 5. How to Run the Project
**Step 1** — Open the Notebook
```bash
jupyter notebook Transaction_Category_Prediction_ML.ipynb
```

**Step 2** — Follow the Sections in Order
1. Load & clean the data
2. Run EDA visualizations
3. Run TF-IDF + SVD
4. Generate clusters (K=56)
5. Assign category names to clusters
6. Train the classification model
7. Evaluate and visualize results

## 📌 6. Data Cleaning & Preprocessing
Steps performed:
- Removed duplicates
- Removed nulls
- Fixed invalid values (UserId -1 → 1)
- Cleaned text (lowercase, remove symbols/spaces)
- Feature engineering:
  - TotalCost = NumberOfItemsPurchased * CostPerItem
  - Label-encoded Country
  - Removed outliers using IQR
  - Normalized & standardized numerical values

All cleaning is saved directly to the updated `df_new` dataframe.

## 📌 7. Exploratory Data Analysis (EDA)
EDA included:
- Distribution plots for numeric columns
- Boxplots of numeric features by country
- Outlier visualization
- Observation that no strong correlation exists between numeric features
- Text length distribution removed because not useful

EDA confirmed that `ItemDescription` is the main signal in the dataset.

## 📌 8. Modeling Approach
**Step 1** — Text Vectorization  
Used TF-IDF with bigrams (1–2 grams) and `max_features=5000`.

**Step 2** — Dimensionality Reduction  
Applied TruncatedSVD (150 components) for speeding up clustering.

**Step 3** — Unsupervised Clustering
- Used KMeans with 56 clusters
- Determined using:
  - Silhouette Score curve
  - Elbow Method curve
- Extracted 8 representative samples per cluster
- Assigned semantic category names manually

**Step 4** — Supervised Classification
- Model: Logistic Regression (balanced, max_iter=2000)
- Input: Combined TF-IDF+SVD (text) + scaled numeric features
- Achieved:
  - 99% accuracy
  - Excellent macro F1 score
  - Proper performance on large dataset (~540K rows)

## 📌 9. Results
- Model achieved ~99% accuracy when text features included.
- Without `ItemDescription` → accuracy dropped to ~28%, proving text is essential.
- Confusion matrix shows strong separation between categories.

This demonstrates the effectiveness of the hybrid pipeline:  
**NLP + Clustering + Logistic Regression**

## 📌 10. Ethical Considerations
- The dataset is extremely biased toward UK transactions (90%)
- This may cause:
  - Overconfidence for UK-based patterns
  - Poor generalization to other countries
- Manual category creation may introduce human bias
- Missing columns (merchant, channel, timestamp) limit fairness and diversity

**Bias Mitigation Suggestions**
- Add more countries
- Add merchant/store metadata
- Use balanced sampling
- Use multilingual NLP
- Use model explainability to detect unwanted bias

## 📌 11. Limitations
- Heavy reliance on `ItemDescription` → fragile if text is missing
- Outliers remain even after IQR filtering
- Manual labeling of clusters required
- No time-series information
- No user behavioral features

## 📌 12. Future Work
- Use Deep Learning (BERT, DistilBERT) for better text understanding
- Add more features (merchant, timestamp, product brand)
- Use time-series forecasting to detect seasonal purchases
- Automate cluster naming using LLM or topic modeling
- Build a simple API to classify new transactions

## 📌 13. Project Files Included
- Jupyter Notebook: `Transaction_Category_Prediction_ML.ipynb`
- Cleaned Dataset / Preprocessed version
- Final Presentation Slides (PPTX) — FINAL PROJECT 08e16aac-4046-4d83-a9a7-9554c00…
- This `README.md`
- Any generated visualizations

## 📌 14. Acknowledgements
- Dataset by vipin20 on Kaggle
- ReDI School Machine Learning Course
- Scikit-Learn Documentation
- ChatGPT used for brainstorming & debugging assistance (as required by academic honesty rules)
```
