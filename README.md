# 🛡️ CyberShield – Phishing Website Detection & Analytics Platform
**IBM Data Analytics Academic Internship Project**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Academic-IBM%20Internship-brightgreen.svg)]()

---

## 1. Project Title & Overview
**CyberShield** is an AI-powered cybersecurity analytics web platform designed to detect, analyze, and quantify the risk of malicious phishing websites using Machine Learning.

Rather than only classifying a link behind the scenes, CyberShield provides complete threat transparency:
- Evaluates **18 genuine lexical URL features** for live prediction without visiting untrusted web servers.
- Displays an intuitive **0–100 CyberShield Risk Score** categorized into Low, Medium, and High risk.
- Explains detection reasons through clear **Security Indicators** (e.g., protocol security, IP usage, excessive special characters, subdomain depth).
- Delivers an exploratory **Security Analytics Dashboard** and full **Model Performance Benchmarks** on the official 235,795-sample UCI dataset.

---

## 2. Problem Statement
Phishing is the #1 vector for credential harvesting, financial fraud, and malware distribution globally. Cybercriminals frequently spin up thousands of deceptive URLs mimicking legitimate banks, universities, and tech companies. Because these fraudulent domains often live for only a few hours, traditional static blacklists cannot keep up. 

**CyberShield** leverages supervised Machine Learning to identify the subtle structural, mathematical, and lexical fingerprints left behind by attackers.

---

## 3. Objectives
- Clean and explore the official **UCI PhiUSIIL Phishing URL Dataset** (235,795 records).
- Develop an honest, dual-track machine learning design:
  - **Track A (Full Dataset Model):** Demonstrates complete academic research benchmarking on 50 numerical and webpage structural attributes.
  - **Track B (URL-Only Model):** Extracts 18 genuine lexical features directly from pasted URL strings for safe, live prediction without active network requests.
- Calculate and visualize critical cybersecurity evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Build a clean, responsive Streamlit dashboard suitable for academic presentations and viva defense.

---

## 4. Dataset
- **Name:** PhiUSIIL Phishing URL (Website) Dataset
- **Source:** UCI Machine Learning Repository (Dataset ID: 967)
- **URL:** [https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset)
- **Total Records:** 235,795
- **Legitimate Websites (Class 1):** 134,850 (57.2%)
- **Phishing Websites (Class 0):** 100,945 (42.8%)
- **Attributes:** 56 columns (URL lexical features, webpage source features, and target label)

---

## 5. Technologies Used
- **Python 3.10+**: Core programming environment.
- **Pandas**: Tabular data loading, manipulation, and statistical aggregation.
- **NumPy**: Matrix computation and numerical arrays.
- **Scikit-learn**: Machine learning modeling, stratified train/test split, and evaluation metrics.
- **Matplotlib & Seaborn**: Statistical charting, correlation heatmaps, and distribution plots.
- **Joblib**: Efficient serialization of trained models, feature schemas, and metric dictionaries.
- **Streamlit**: Multi-page web dashboard and interactive UI.

---

## 6. Methodology & Project Workflow
```
Data Collection (UCI PhiUSIIL)
       ↓
Data Cleaning & Quality Audit (0 nulls, 0 duplicates, remove non-predictive metadata)
       ↓
Exploratory Data Analysis (EDA - Class distributions, length density, protocol adoption)
       ↓
Feature Engineering & Selection (Separating URL-extractable vs webpage features)
       ↓
Stratified Train/Test Partitioning (80% Train: 188,636 / 20% Test: 47,159)
       ↓
Random Forest Classifier Training (Ensemble bagging across decision trees)
       ↓
Model Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
       ↓
CyberShield Interactive Web Dashboard (Streamlit)
       ↓
Live URL Risk Assessment & Security Indicators
```

---

## 7. Machine Learning Model & Results
We chose **Random Forest Classifier** because it combines hundreds of decision trees using bagging and random feature subspaces, preventing overfitting and providing transparent Gini feature importance.

### Actual Evaluated Results on Test Set (47,159 Samples):

| Metric | Track A: Full Dataset Model (50 Features) | Track B: URL-Only Model (18 Features) |
|---|---|---|
| **Accuracy** | **100.00%** | **99.72%** |
| **Precision** | **100.00%** | **99.59%** |
| **Recall** | **100.00%** | **99.93%** |
| **F1-Score** | **100.00%** | **99.76%** |
| **True Negatives (Phishing)** | 20,189 | 20,077 |
| **False Positives (Phishing flagged Legit)** | 0 | 112 |
| **False Negatives (Legit flagged Phishing)** | 0 | 19 |
| **True Positives (Legitimate)** | 26,970 | 26,951 |

*Note: All numbers represent actual evaluation outputs computed from the test set.*

---

## 8. Application Features
1. **🏠 Home Page:** High-level metrics, phishing fundamentals, and architectural workflow.
2. **🔎 Live URL Analyzer:**
   - Quick pre-set test buttons (`Google`, `Example.com`, `IP Address URL`, `Phishing Spoof`).
   - URL validation and genuine feature extraction (no scraping or visiting malicious links).
   - Probability-based **CyberShield Risk Score (0–100)**:
     - `0 – 30`: 🟢 LOW RISK (Legitimate)
     - `31 – 60`: 🟡 MEDIUM RISK (Suspicious)
     - `61 – 100`: 🔴 HIGH RISK (Phishing)
   - Color-coded security indicator badges with plain-English rationales.
3. **📊 Security Analytics:** Class distribution bar charts, HTTPS adoption rates, URL/domain length curves, and feature correlation heatmaps.
4. **🤖 Model Performance:** Interactive comparison between Full and URL-Only models, confusion matrices, classification reports, and top 10 feature importance charts.
5. **💡 Cybersecurity Insights:** Threat analytics on lexical obfuscation, HTTPS false sense of security, and SOC triage workflows.
6. **ℹ️ About Project:** Capstone metadata, technology matrix, dataset citations, and viva defense guide.

---

## 9. Project Directory Structure
```
CyberShield/
│
├── data/
│   └── phishing_dataset.csv       # Official UCI PhiUSIIL dataset (235,795 rows)
│
├── model/
│   ├── phishing_model.pkl          # Serialized Full 50-feature Random Forest model
│   ├── feature_columns.pkl         # Feature column names for Full model
│   ├── model_metrics.pkl           # Saved metrics dictionary (Full model)
│   ├── feature_importance.csv      # Sorted Gini importance table (Full model)
│   ├── url_model.pkl               # Serialized URL-Only 18-feature Random Forest model
│   ├── url_feature_columns.pkl     # Feature column names for URL-only model
│   ├── url_model_metrics.pkl       # Saved metrics dictionary (URL-only model)
│   └── url_feature_importance.csv  # Sorted Gini importance table (URL-only model)
│
├── notebooks/
│   └── CyberShield_Analysis.ipynb  # Comprehensive 17-section analysis notebook
│
├── app/
│   └── CyberShield.py              # Complete Streamlit multi-page platform
│
├── reports/
│   └── project_summary.md          # Academic project summary report
│
├── requirements.txt                # Python package dependencies
├── train_models.py                 # Reproducible model training script
└── README.md                       # Complete project documentation
```

---

## 10. How to Install & Run

### Step 1: Open Terminal in Project Folder
Navigate to the project root:
```bash
cd CyberShield
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app/CyberShield.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 11. Screenshots Section Placeholder
*(Take screenshots of the live running application for your academic submission)*
- **Home Dashboard:** `reports/screenshots/home.png`
- **Live URL Analyzer:** `reports/screenshots/url_analyzer.png`
- **Security Analytics:** `reports/screenshots/analytics.png`
- **Confusion Matrix & Metrics:** `reports/screenshots/metrics.png`

---

## 12. Viva Defense Guide (Questions & Answers)
- **Q1: Why did you choose Random Forest over a Single Decision Tree?**  
  *Answer:* A single decision tree is prone to high variance and overfitting. Random Forest builds an ensemble of multiple uncorrelated trees using bootstrap sampling and random feature subsets, providing significantly higher generalization and native feature importance.
- **Q2: Why did you train two models (Track A and Track B)?**  
  *Answer:* The original dataset contains both URL lexical features and webpage HTML features (like lines of code and iframe tags). To make a genuinely operational live URL analyzer without dangerously fetching or executing malicious websites, Track B uses only the 18 features extractable from the URL string alone, achieving an outstanding 99.72% accuracy.
- **Q3: Why is Recall critical in cybersecurity?**  
  *Answer:* A False Negative in cybersecurity means a phishing attack slips past the filter undetected, exposing the user to identity theft and ransomware. High recall ensures maximum threat interception.

---

## 13. Future Enhancements
1. Integrating real-time DNS record verification and WHOIS domain age lookups.
2. Expanding detection against IDN (Internationalized Domain Name) homoglyph spoofing.
3. Adding sandbox headless browser rendering for deep dynamic analysis of medium-risk URLs.

---

## 14. Dataset Citation
```bibtex
@misc{phiusiil2024,
  author       = {Prasad, Arvind and Chandra, Shalini},
  title        = {PhiUSIIL Phishing URL (Website) Dataset},
  year         = {2024},
  howpublished = {UCI Machine Learning Repository},
  note         = {Dataset ID: 967}
}
```
