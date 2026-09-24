# 🛡️ Project Summary: CyberShield – Phishing Website Detection & Analytics Platform
**IBM Data Analytics Academic Internship Capstone Project**

---

## 1. Executive Summary & Project Title
- **Project Title:** CyberShield – Phishing Website Detection & Analytics Platform
- **Domain:** Cybersecurity Analytics, Machine Learning & Threat Intelligence
- **Target Audience:** Cybersecurity Analysts, SOC Teams, Web Security Administrators, End-Users

CyberShield is an enterprise-grade academic machine learning platform built to detect, analyze, and quantify the risk of malicious phishing websites. It implements an end-to-end data analytics workflow:
$$\text{Data Fundamentals} \longrightarrow \text{Data Cleaning} \longrightarrow \text{EDA} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Machine Learning} \longrightarrow \text{Interactive Web Application}$$

---

## 2. Problem Statement
Phishing attacks represent over 80% of reported social engineering incidents worldwide. Cybercriminals systematically engineer convincing replicas of legitimate banking portals, payment processors, and corporate single sign-on (SSO) pages.
Traditional defenses rely heavily on reactive domain blacklists (such as Google Safe Browsing and DNS sinkholes), which fail to catch newly registered zero-day domains that remain active for only hours before retiring.

To combat this, machine learning provides automated, proactive classification based on structural, lexical, and behavioral anomalies present in the URL and webpage metadata.

---

## 3. Project Objectives
1. **Curate and Clean Benchmark Data:** Acquire and validate the official UCI PhiUSIIL Phishing URL Dataset (235,795 records).
2. **Transparent Architectural Design:** Separate lexical features that can genuinely be extracted from a pasted URL string from webpage-dependent HTML features.
3. **Train High-Performance Classifiers:** Implement Random Forest ensembles with stratified partitioning to prevent data leakage and evaluate real-world generalization.
4. **Deliver Actionable Security Signals:** Translate model probability distributions into an intuitive **0–100 CyberShield Risk Score** with clear, understandable security indicators.
5. **Interactive Streamlit Platform:** Build a responsive multi-page dashboard displaying exploratory distributions, model performance metrics, and a live URL analysis engine.

---

## 4. Dataset Description
- **Dataset Name:** PhiUSIIL Phishing URL (Website) Dataset
- **Repository:** UCI Machine Learning Repository (Dataset ID: 967)
- **Total Records:** 235,795 unique instances
- **Legitimate Websites (`label = 1`):** 134,850 (57.2%)
- **Phishing Websites (`label = 0`):** 100,945 (42.8%)
- **Total Raw Columns:** 56 columns
- **Class Balance:** Approximately 1.34:1, representing a realistic, balanced classification problem without synthetic sampling.

The dataset includes both:
- **Lexical/URL Characteristics:** Length, domain length, protocol, subdomains, token continuation rates, character probabilities, digit/special character ratios, and obfuscation markers.
- **Webpage Content Characteristics:** Total lines of code, internal/external hyperlink references, embedded iframes, forms, popups, and scripts.

---

## 5. Data Cleaning & Preprocessing
A rigorous data audit was conducted to preserve integrity and avoid data leakage:
1. **Missing Value Audit:** Confirmed zero missing or null values across all 235,795 rows and 56 features.
2. **Duplicate Row Audit:** Confirmed zero duplicate instances.
3. **Exclusion of Non-Predictive Metadata:** 
   - `FILENAME`: Unique file identifier having zero generalizable security signal.
   - `URL`, `Domain`, `TLD`, `Title`: Raw text strings excluded from the numerical feature matrix to avoid memorization artifacts.
4. **Target Variable Isolation:** `label` was strictly separated into target vector $y$ before feature transformation.
5. **Stratification:** Splitting was stratified to ensure the 57.2% / 42.8% class balance was maintained in both training ($N=188,636$) and test ($N=47,159$) sets.

---

## 6. Exploratory Data Analysis (EDA)
EDA established vital cybersecurity signals across the dataset:
- **Class Balance:** A healthy 57.2% legitimate vs 42.8% phishing distribution ensures unskewed threshold modeling.
- **Protocol Adoption:** 100% of legitimate sites in the dataset employed HTTPS, whereas over 50% of phishing links used unencrypted HTTP. However, 49% of phishing websites also used HTTPS, highlighting that SSL certificates alone do not guarantee authenticity.
- **Lexical Dispersion:** Phishing URLs showed higher variance in URL length, domain length, and special character density, driven by deceptive subdomain stacking and redirection tokens.
- **Webpage Richness:** Legitimate sites displayed thousands of lines of code and extensive external references (`NoOfExternalRef`), while phishing kit landing pages were typically isolated and compact.

---

## 7. Machine Learning Methodology
To ensure transparency and academic rigor, CyberShield implemented a dual-track strategy:

### Track A: Full Dataset Model (50 Features)
- **Features:** All 50 numerical and structural features (URL lexical attributes + webpage content attributes).
- **Algorithm:** `RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)`
- **Role:** Demonstrates research-level benchmark classification across the complete multi-modal feature set.

### Track B: URL-Only Model (18 Features)
- **Features:** 18 features genuinely extractable from a raw URL string without active network requests:
  `URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`.
- **Algorithm:** `RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)`
- **Role:** Powers the real-world live URL analyzer, ensuring 100% genuine feature computation without fabricating unavailable webpage content.

---

## 8. Actual Model Performance & Results

All metrics were computed on the held-out test split of 47,159 samples ($20\%$ test split):

| Evaluation Metric | Track A (Full 50-Feature Model) | Track B (URL-Only 18-Feature Model) |
|---|---|---|
| **Accuracy** | **100.00%** | **99.72%** |
| **Precision** | **100.00%** | **99.59%** |
| **Recall** | **100.00%** | **99.93%** |
| **F1-Score** | **100.00%** | **99.76%** |
| **True Negatives (Phishing)** | 20,189 | 20,077 |
| **False Positives (Phishing as Legit)** | 0 | 112 |
| **False Negatives (Legit as Phishing)** | 0 | 19 |
| **True Positives (Legit)** | 26,970 | 26,951 |

### Cybersecurity Significance of Metrics:
- **Precision (99.59%):** Guarantees that when an alert is fired, it is genuinely malicious, minimizing alert fatigue in security operations.
- **Recall (99.93%):** Ensures that out of 26,970 legitimate samples, 26,951 are safeguarded, catching nearly all malicious variants.

---

## 9. Top Feature Importance

### URL-Only Model (Top Drivers):
1. **`IsHTTPS` (39.3%):** Strong initial indicator separating basic HTTP phishing sites.
2. **`NoOfOtherSpecialCharsInURL` (13.5%):** Excessive punctuation indicates token concealment.
3. **`NoOfDegitsInURL` (9.3%):** Automated random hash generation in phishing URLs.
4. **`LetterRatioInURL` (7.7%):** Low alphabetic ratio signals obfuscation.
5. **`DegitRatioInURL` (7.3%):** Numerical density within the path and parameters.
6. **`URLLength` (5.9%):** Deep subfolder nesting and session tracking tokens.

---

## 10. Application Features
The Streamlit application (`app/CyberShield.py`) provides:
1. **🏠 Home:** Executive overview, metric cards, problem definition, and workflow diagrams.
2. **🔎 URL Analyzer:** Input field, quick example presets, instant risk scoring (0–100 scale), categorical risk tiers (Low/Medium/High), and granular security indicator badges.
3. **📊 Security Analytics:** Interactive tabs showing class distributions, protocol usage, URL length boxplots, domain density curves, and correlation matrices.
4. **🤖 Model Performance:** Interactive comparison of Track A and Track B, confusion matrix heatmaps, classification report tables, and feature importance charts.
5. **💡 Cybersecurity Insights:** Actionable intelligence on lexical obfuscation, HTTPS deceptive trust, and defense-in-depth SOC strategies.
6. **ℹ️ About Project:** Architecture overview, technology matrix, dataset citations, and viva defense guide.

---

## 11. Limitations & Future Scope
- **Current Limitations:**
  - Lexical URL models cannot detect compromised legitimate websites hosting newly injected phishing subpages until retraining occurs.
  - Active URL cloaking (delivering different content based on user agent or IP geofencing) requires dynamic execution environments.
- **Future Scope:**
  - Integrate real-time WHOIS domain age queries and DNS record history.
  - Add sandboxed headless browser DOM snapshotting for medium-risk URLs.
  - Extend the model to detect homoglyph and IDN (Internationalized Domain Name) punycode attacks.

---

## 12. Conclusion
CyberShield demonstrates how structured data analytics, rigorous feature engineering, and ensemble machine learning can provide robust, real-time cyber defense. By strictly distinguishing lexical URL attributes from webpage content attributes, CyberShield delivers high accuracy while maintaining operational safety and transparency.
