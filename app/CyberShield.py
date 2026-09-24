"""
🛡️ CyberShield – Phishing Website Detection & Analytics Platform
IBM Data Analytics Academic Internship Project

Technologies: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit
"""

import os
import re
import urllib.parse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="CyberShield – Phishing Website Detection & Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1e3d59;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-low {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .risk-med {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .indicator-box {
        background: #fdfdfe;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 4px solid #ddd;
    }
    .indicator-pass {
        border-left-color: #28a745;
    }
    .indicator-warn {
        border-left-color: #ffc107;
    }
    .indicator-fail {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. PATH RESOLUTION & DATA LOADERS
# =============================================================================
def get_base_dir():
    # If app is inside CyberShield/app, base is CyberShield
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Check candidates
    for candidate in [current_dir, parent_dir, os.path.join(parent_dir, 'CyberShield'), os.getcwd(), os.path.join(os.getcwd(), 'CyberShield')]:
        if os.path.exists(os.path.join(candidate, 'model', 'phishing_model.pkl')) or os.path.exists(os.path.join(candidate, 'data', 'phishing_dataset.csv')):
            return candidate
    return parent_dir

BASE_DIR = get_base_dir()

@st.cache_resource
def load_models_and_metrics():
    model_dir = os.path.join(BASE_DIR, 'model')
    
    # Full Model
    full_model_path = os.path.join(model_dir, 'phishing_model.pkl')
    full_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    full_metrics_path = os.path.join(model_dir, 'model_metrics.pkl')
    full_fi_path = os.path.join(model_dir, 'feature_importance.csv')
    
    # URL-Only Model
    url_model_path = os.path.join(model_dir, 'url_model.pkl')
    url_cols_path = os.path.join(model_dir, 'url_feature_columns.pkl')
    url_metrics_path = os.path.join(model_dir, 'url_model_metrics.pkl')
    url_fi_path = os.path.join(model_dir, 'url_feature_importance.csv')
    
    data = {}
    if os.path.exists(full_model_path):
        data['full_model'] = joblib.load(full_model_path)
    if os.path.exists(full_cols_path):
        data['full_cols'] = joblib.load(full_cols_path)
    if os.path.exists(full_metrics_path):
        data['full_metrics'] = joblib.load(full_metrics_path)
    if os.path.exists(full_fi_path):
        data['full_fi'] = pd.read_csv(full_fi_path)
        
    if os.path.exists(url_model_path):
        data['url_model'] = joblib.load(url_model_path)
    if os.path.exists(url_cols_path):
        data['url_cols'] = joblib.load(url_cols_path)
    if os.path.exists(url_metrics_path):
        data['url_metrics'] = joblib.load(url_metrics_path)
    if os.path.exists(url_fi_path):
        data['url_fi'] = pd.read_csv(url_fi_path)
        
    return data

@st.cache_data
def load_dataset_sample():
    data_path = os.path.join(BASE_DIR, 'data', 'phishing_dataset.csv')
    if os.path.exists(data_path):
        # Read a manageable sample for visualization and statistics
        return pd.read_csv(data_path)
    return None

models_data = load_models_and_metrics()
df_data = load_dataset_sample()

# =============================================================================
# 3. GENUINE URL FEATURE EXTRACTION ENGINE
# =============================================================================
def extract_live_url_features(url_str):
    """
    Extracts lexical and structural characteristics from raw URL string.
    Matches the exact training definitions of the PhiUSIIL dataset.
    Does NOT require visiting or fetching the website.
    """
    url_str = url_str.strip()
    if not url_str.startswith(('http://', 'https://')):
        url_str = 'https://' + url_str

    parsed = urllib.parse.urlparse(url_str)
    domain = parsed.netloc.split(':')[0]
    
    # In the PhiUSIIL dataset, legitimate apex domains (e.g. google.com) were archived with standard 'www.'
    # Standardize canonical representation for feature extraction
    canon_url = url_str
    ip_match = 1 if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', domain) else 0
    
    if not domain.startswith('www.') and not ip_match:
        parts = domain.split('.')
        if len(parts) == 2:
            canon_url = url_str.replace(domain, 'www.' + domain, 1)
            parsed = urllib.parse.urlparse(canon_url)
            domain = parsed.netloc.split(':')[0]

    url_len = max(len(canon_url) - 1, 1)
    domain_len = len(domain)
    
    parts = domain.split('.')
    tld = parts[-1] if len(parts) > 1 else ''
    tld_len = len(tld)
    
    no_of_subdomains = max(domain.count('.') - 1, 0)
    clean_no_www = re.sub(r'^https?://(www\.)?', '', canon_url)
    
    obf_chars = len(re.findall(r'%[0-9a-fA-F]{2}', canon_url))
    has_obf = 1 if obf_chars > 0 else 0
    obf_ratio = obf_chars / url_len
    
    letters = max(len(re.findall(r'[a-zA-Z]', clean_no_www)) - 1, 0)
    letter_ratio = round(letters / url_len, 3)
    
    digits = len(re.findall(r'[0-9]', clean_no_www))
    digit_ratio = round(digits / url_len, 3)
    
    equals = canon_url.count('=')
    qmark = canon_url.count('?')
    amp = canon_url.count('&')
    
    other_spec = len(re.findall(r'[^a-zA-Z0-9=?&]', clean_no_www))
    spec_ratio = round(other_spec / url_len, 3)
    
    is_https = 1 if canon_url.lower().startswith('https://') else 0

    features = {
        'URLLength': url_len,
        'DomainLength': domain_len,
        'IsDomainIP': ip_match,
        'TLDLength': tld_len,
        'NoOfSubDomain': no_of_subdomains,
        'HasObfuscation': has_obf,
        'NoOfObfuscatedChar': obf_chars,
        'ObfuscationRatio': obf_ratio,
        'NoOfLettersInURL': letters,
        'LetterRatioInURL': letter_ratio,
        'NoOfDegitsInURL': digits,
        'DegitRatioInURL': digit_ratio,
        'NoOfEqualsInURL': equals,
        'NoOfQMarkInURL': qmark,
        'NoOfAmpersandInURL': amp,
        'NoOfOtherSpecialCharsInURL': other_spec,
        'SpacialCharRatioInURL': spec_ratio,
        'IsHTTPS': is_https
    }
    
    # Additional Security Heuristic Indicators
    suspicious_keywords = [
        'login', 'signin', 'verify', 'verification', 'update', 'banking', 
        'secure', 'security', 'account', 'auth', 'password', 'credential',
        'paypal', 'ebay', 'amazon', 'apple', 'support', 'wallet', 'crypto'
    ]
    path_and_domain = (domain + parsed.path).lower()
    detected_keywords = [k for k in suspicious_keywords if k in path_and_domain]
    
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do']
    is_shortened = 1 if any(s in domain.lower() for s in shorteners) else 0
    has_at_symbol = 1 if '@' in url_str else 0
    has_hyphen_in_domain = 1 if '-' in domain else 0

    heuristics = {
        'Domain': domain,
        'RawURL': url_str,
        'DetectedKeywords': detected_keywords,
        'IsShortened': is_shortened,
        'HasAtSymbol': has_at_symbol,
        'HasHyphenInDomain': has_hyphen_in_domain,
        'RawLength': len(url_str)
    }

    return features, heuristics

# =============================================================================
# 4. SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown("## 🛡️ CyberShield")
st.sidebar.markdown("**Phishing Detection & Analytics**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "🔎 URL Analyzer",
        "📊 Security Analytics",
        "🤖 Model Performance",
        "💡 Cybersecurity Insights",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("IBM Data Analytics Academic Internship")
st.sidebar.caption("Dataset: UCI PhiUSIIL (ID: 967)")
st.sidebar.caption("Algorithm: Random Forest")

# =============================================================================
# 5. PAGE: 🏠 HOME
# =============================================================================
if menu == "🏠 Home":
    st.markdown('<p class="main-title">🛡️ CyberShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Phishing Website Detection & Cybersecurity Analytics Platform</p>', unsafe_allow_html=True)
    
    st.info("💡 **Welcome to CyberShield**: An end-to-end data analytics and machine learning solution designed to identify deceptive phishing websites using lexical URL patterns and security indicators.")

    # High-level metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = models_data.get('full_metrics', {}).get('total_records', 235795)
    phishing_count = models_data.get('full_metrics', {}).get('phishing_count', 100945)
    legitimate_count = models_data.get('full_metrics', {}).get('legitimate_count', 134850)
    url_f1 = models_data.get('url_metrics', {}).get('f1_score', 0.9976)
    
    with col1:
        st.metric(label="Total Dataset Records", value=f"{total_records:,}")
    with col2:
        st.metric(label="Phishing Websites (0)", value=f"{phishing_count:,}", delta="42.8%", delta_color="inverse")
    with col3:
        st.metric(label="Legitimate Websites (1)", value=f"{legitimate_count:,}", delta="57.2%")
    with col4:
        st.metric(label="URL Model F1-Score", value=f"{url_f1*100:.2f}%")

    st.markdown("---")
    
    # Core explanation
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### 🎯 What is Phishing?")
        st.write("""
        **Phishing** is a deceptive cyber attack where attackers engineer fraudulent websites mimicking authentic organizations (such as online banking portals, social media platforms, cloud services, and email providers).
        
        The primary objective is social engineering: manipulating victims into disclosing confidential information including:
        - Login credentials (usernames and passwords)
        - Credit card details and banking OTPs
        - Personal identification numbers (SSN, national IDs)
        
        Traditional blacklist approaches struggle against modern cybercrime because attackers constantly spawn short-lived disposable domains. **Machine Learning** solves this by uncovering generalized lexical, structural, and behavioral markers.
        """)
        
        st.markdown("### 🔬 How CyberShield Works")
        st.write("""
        CyberShield uses a dual-engine architecture:
        1. **Live URL Lexical Engine:** Inspects 18 URL-extractable attributes (protocol, subdomains, token lengths, character ratios, obfuscation) directly from the URL string without visiting untrusted servers.
        2. **Research Benchmark Engine:** Evaluates full 50-feature webpage structure data from the **UCI PhiUSIIL benchmark**, providing deep institutional insights into web layout anomalies.
        """)

    with c2:
        st.markdown("### 📋 Academic Workflow")
        st.markdown("""
        ```mermaid
        flowchart TD
            A[Data Collection] --> B[Data Cleaning & Prep]
            B --> C[Exploratory Data Analysis]
            C --> D[Feature Engineering]
            D --> E[Random Forest Classifier]
            E --> F[Performance Evaluation]
            F --> G[Interactive Dashboard]
            G --> H[Live Risk Analyzer]
        ```
        """)
        st.caption("IBM Data Analytics Methodology: Data Fundamentals → Cleaning → EDA → Machine Learning → Business Application")

# =============================================================================
# 6. PAGE: 🔎 URL ANALYZER
# =============================================================================
elif menu == "🔎 URL Analyzer":
    st.markdown('<p class="main-title">🔎 Live URL Security Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter any website URL to extract security indicators, compute risk score, and classify via Machine Learning.</p>', unsafe_allow_html=True)

    st.warning("⚠️ **Safety Notice:** CyberShield parses URLs purely as strings. It **never** initiates network connections, requests, or executes remote code on suspicious targets.")

    # Pre-filled example buttons
    st.markdown("**Quick Test Examples:**")
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    
    default_url = "https://www.google.com"
    if 'input_url' not in st.session_state:
        st.session_state.input_url = default_url

    if exp_col1.button("🌐 Google (Legitimate)"):
        st.session_state.input_url = "https://www.google.com"
    if exp_col2.button("🏢 Example.com (Legitimate)"):
        st.session_state.input_url = "https://example.com"
    if exp_col3.button("⚠️ IP Address URL (Suspicious)"):
        st.session_state.input_url = "http://192.168.1.1/admin/login.php?update=1"
    if exp_col4.button("🚨 Phishing Spoof (Malicious)"):
        st.session_state.input_url = "http://secure-paypal-verification-alert.servehttp.com/login.php"

    user_url = st.text_input("Enter Website URL:", value=st.session_state.input_url, placeholder="e.g., https://secure-bank.example.com/login")
    analyze_btn = st.button("🔍 Analyze URL", type="primary", use_container_width=True)

    if analyze_btn or user_url:
        if not user_url.strip():
            st.error("Please enter a valid URL to analyze.")
        else:
            # Feature extraction
            features, heuristics = extract_live_url_features(user_url)
            
            # Predict using URL model
            url_model = models_data.get('url_model')
            url_cols = models_data.get('url_feature_columns') or models_data.get('url_cols')
            
            if url_model is None or url_cols is None:
                st.error("URL-Only Model artifacts not found. Please train models first.")
            else:
                input_df = pd.DataFrame([features])[url_cols]
                prediction = url_model.predict(input_df)[0]
                probabilities = url_model.predict_proba(input_df)[0]
                
                # Class 0 = Phishing, Class 1 = Legitimate
                phishing_prob = probabilities[0]
                legit_prob = probabilities[1]
                
                # Risk Score (0 to 100): Phishing probability multiplied by 100
                risk_score = int(round(phishing_prob * 100))
                
                st.markdown("---")
                st.markdown("### 📊 CyberShield Assessment Results")
                
                # Risk level card and metrics
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown("#### 🎯 Classification & Probability")
                    if prediction == 1 and risk_score <= 30:
                        st.markdown(f'<div class="risk-low">🟢 LOW RISK: LEGITIMATE WEBSITE<br><span style="font-size:0.9rem; font-weight:normal;">CyberShield Risk Score: {risk_score}/100</span></div>', unsafe_allow_html=True)
                    elif risk_score <= 60:
                        st.markdown(f'<div class="risk-med">🟡 MEDIUM RISK: SUSPICIOUS WEBSITE<br><span style="font-size:0.9rem; font-weight:normal;">CyberShield Risk Score: {risk_score}/100</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-high">🔴 HIGH RISK: PHISHING WEBSITE<br><span style="font-size:0.9rem; font-weight:normal;">CyberShield Risk Score: {risk_score}/100</span></div>', unsafe_allow_html=True)
                    
                    st.write("")
                    st.progress(risk_score / 100.0)
                    st.caption("Probability Scale: 0 (Definitely Legitimate) ──→ 100 (High-Confidence Phishing)")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Phishing Probability", f"{phishing_prob*100:.1f}%")
                    m2.metric("Legitimate Probability", f"{legit_prob*100:.1f}%")

                with res_col2:
                    st.markdown("#### 🛡️ CyberShield Risk Scale")
                    st.markdown("""
                    - **0 – 30 (🟢 LOW RISK):** Normal lexical structure, valid HTTPS, standard domain hierarchy, low character entropy.
                    - **31 – 60 (🟡 MEDIUM RISK):** Anomalous characteristics present (unusual length, multiple subdomains, or unencrypted HTTP). Manual verification advised.
                    - **61 – 100 (🔴 HIGH RISK):** Severe indicators detected (direct IP usage, deceptive keywords, extreme special characters, or known phishing signatures).
                    """)
                    st.caption("*Note: The CyberShield Risk Score is a probabilistic machine learning index trained on the UCI PhiUSIIL dataset, not an official regulatory certification.*")

                st.markdown("---")
                
                # Security Indicators Breakdown
                st.markdown("### 🔍 Security Indicators Breakdown")
                
                ind_col1, ind_col2 = st.columns(2)
                
                with ind_col1:
                    # HTTPS Indicator
                    if features['IsHTTPS'] == 1:
                        st.markdown('<div class="indicator-box indicator-pass">✅ <b>HTTPS Protocol:</b> Enabled (Encrypted connection present)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="indicator-box indicator-fail">⚠️ <b>HTTPS Protocol:</b> Insecure HTTP detected (Associated with higher phishing risk)</div>', unsafe_allow_html=True)
                        
                    # IP Address Domain Indicator
                    if features['IsDomainIP'] == 0:
                        st.markdown('<div class="indicator-box indicator-pass">✅ <b>Domain Format:</b> Standard domain name used (No raw IP host)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="indicator-box indicator-fail">⚠️ <b>IP Address Host:</b> Direct IP address used instead of domain name (High phishing correlation)</div>', unsafe_allow_html=True)

                    # URL Length Indicator
                    if heuristics['RawLength'] < 60:
                        st.markdown(f'<div class="indicator-box indicator-pass">✅ <b>URL Length:</b> Normal ({heuristics["RawLength"]} characters)</div>', unsafe_allow_html=True)
                    elif heuristics['RawLength'] < 100:
                        st.markdown(f'<div class="indicator-box indicator-warn">🟡 <b>URL Length:</b> Elevated length ({heuristics["RawLength"]} characters)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="indicator-box indicator-fail">⚠️ <b>URL Length:</b> Unusually long ({heuristics["RawLength"]} characters; often conceals redirect tokens)</div>', unsafe_allow_html=True)

                    # Subdomain Indicator
                    if features['NoOfSubDomain'] <= 1:
                        st.markdown(f'<div class="indicator-box indicator-pass">✅ <b>Subdomains:</b> Standard hierarchy ({features["NoOfSubDomain"]} subdomain detected)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="indicator-box indicator-warn">⚠️ <b>Subdomains:</b> Multiple subdomains stacked ({features["NoOfSubDomain"]} subdomains detected)</div>', unsafe_allow_html=True)

                with ind_col2:
                    # Suspicious Keywords
                    if not heuristics['DetectedKeywords']:
                        st.markdown('<div class="indicator-box indicator-pass">✅ <b>Target Keywords:</b> No deceptive security/financial keywords found in path</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="indicator-box indicator-fail">⚠️ <b>Deceptive Keywords:</b> Detected keywords: <code>{", ".join(heuristics["DetectedKeywords"])}</code></div>', unsafe_allow_html=True)

                    # Shortening Service
                    if heuristics['IsShortened'] == 0:
                        st.markdown('<div class="indicator-box indicator-pass">✅ <b>URL Shortener:</b> Direct host (No link-masking shortener detected)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="indicator-box indicator-fail">⚠️ <b>URL Shortener:</b> Shortening service in use (Masks true destination)</div>', unsafe_allow_html=True)

                    # Special Characters & Obfuscation
                    if features['HasObfuscation'] == 0 and features['SpacialCharRatioInURL'] < 0.15:
                        st.markdown(f'<div class="indicator-box indicator-pass">✅ <b>Lexical Characters:</b> Normal symbol density (Special char ratio: {features["SpacialCharRatioInURL"]:.3f})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="indicator-box indicator-warn">⚠️ <b>Lexical Complexity:</b> High special character ratio ({features["SpacialCharRatioInURL"]:.3f}) or hex obfuscation detected</div>', unsafe_allow_html=True)

                    # Digit Density
                    if features['DegitRatioInURL'] < 0.10:
                        st.markdown(f'<div class="indicator-box indicator-pass">✅ <b>Digit Density:</b> Low digit count ({features["NoOfDegitsInURL"]} digits, ratio: {features["DegitRatioInURL"]:.2f})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="indicator-box indicator-warn">⚠️ <b>Digit Density:</b> High numerical ratio ({features["NoOfDegitsInURL"]} digits, ratio: {features["DegitRatioInURL"]:.2f})</div>', unsafe_allow_html=True)

                st.markdown("---")
                with st.expander("🛠️ View All 18 URL-Extractable Model Features (Input Vector)"):
                    feat_display = pd.DataFrame([features]).T
                    feat_display.columns = ["Extracted Value"]
                    st.dataframe(feat_display, use_container_width=True)

# =============================================================================
# 7. PAGE: 📊 SECURITY ANALYTICS
# =============================================================================
elif menu == "📊 Security Analytics":
    st.markdown('<p class="main-title">📊 Dataset Security Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Exploratory Data Analysis across 235,795 websites from the UCI PhiUSIIL repository.</p>', unsafe_allow_html=True)

    if df_data is None:
        st.error("Dataset not found in `data/phishing_dataset.csv`. Please ensure the dataset file exists.")
    else:
        # Overview Cards
        n_total = len(df_data)
        n_legit = int((df_data['label'] == 1).sum())
        n_phish = int((df_data['label'] == 0).sum())
        phish_pct = (n_phish / n_total) * 100
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed URLs", f"{n_total:,}")
        c2.metric("Legitimate Websites", f"{n_legit:,}")
        c3.metric("Phishing Websites", f"{n_phish:,}")
        c4.metric("Phishing Percentage", f"{phish_pct:.1f}%")

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Class & Protocol Distributions", "Lexical Structure Distributions", "Correlation & Relationships"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Target Class Distribution")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                colors = ['#e74c3c', '#2ecc71']
                counts = [n_phish, n_legit]
                labels = ['Phishing (0)', 'Legitimate (1)']
                bars = ax.bar(labels, counts, color=colors, width=0.5)
                ax.set_ylabel("Number of Websites", fontsize=11)
                ax.set_title("Phishing vs Legitimate Class Distribution", fontsize=12, fontweight='bold')
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f'{yval:,}\n({yval/n_total*100:.1f}%)', 
                            ha='center', va='center', color='white', fontweight='bold')
                st.pyplot(fig)
                st.caption("The dataset contains 134,850 legitimate websites (57.2%) and 100,945 phishing instances (42.8%), providing a robust class balance.")

            with col_b:
                st.markdown("#### HTTPS Protocol Usage by Class")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                https_ct = pd.crosstab(df_data['IsHTTPS'], df_data['label'], normalize='columns') * 100
                https_ct.plot(kind='bar', color=['#e74c3c', '#2ecc71'], ax=ax, width=0.5)
                ax.set_xticklabels(['HTTP (Insecure)', 'HTTPS (Encrypted)'], rotation=0)
                ax.set_ylabel("Percentage within Class (%)", fontsize=11)
                ax.set_xlabel("Protocol Security", fontsize=11)
                ax.set_title("Proportion of HTTPS Usage Across Classes", fontsize=12, fontweight='bold')
                ax.legend(['Phishing (0)', 'Legitimate (1)'])
                st.pyplot(fig)
                st.caption("100% of legitimate sites in this dataset utilize HTTPS, while a notable portion of phishing sites operate on unencrypted HTTP.")

        with tab2:
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.markdown("#### URL Length Distribution (Characters)")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.boxplot(x='label', y='URLLength', data=df_data, palette=['#e74c3c', '#2ecc71'], ax=ax, showfliers=False)
                ax.set_xticklabels(['Phishing (0)', 'Legitimate (1)'])
                ax.set_xlabel("Classification", fontsize=11)
                ax.set_ylabel("URL Length", fontsize=11)
                ax.set_title("URL Length Comparison by Website Class", fontsize=12, fontweight='bold')
                st.pyplot(fig)
                st.caption("Phishing URLs tend to have larger length dispersion due to keyword stuffing, token redirects, and nested subdirectories.")

            with col_d:
                st.markdown("#### Domain Length Density Distribution")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.kdeplot(data=df_data[df_data['label']==0]['DomainLength'], label='Phishing (0)', color='#e74c3c', fill=True, alpha=0.3, ax=ax)
                sns.kdeplot(data=df_data[df_data['label']==1]['DomainLength'], label='Legitimate (1)', color='#2ecc71', fill=True, alpha=0.3, ax=ax)
                ax.set_xlim(0, 50)
                ax.set_xlabel("Domain Length", fontsize=11)
                ax.set_ylabel("Density", fontsize=11)
                ax.set_title("Domain Length Density Curve", fontsize=12, fontweight='bold')
                ax.legend()
                st.pyplot(fig)
                st.caption("Legitimate corporate domains peak sharply between 12-22 characters, whereas phishing domains show significant variability.")

        with tab3:
            st.markdown("#### Correlation Heatmap of Key Numerical Features")
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_cols = [
                'URLLength', 'DomainLength', 'NoOfSubDomain', 'NoOfLettersInURL',
                'NoOfDegitsInURL', 'SpacialCharRatioInURL', 'IsHTTPS', 'LineOfCode',
                'NoOfExternalRef', 'label'
            ]
            corr_mat = df_data[corr_cols].corr()
            sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap with Target Variable (label)", fontsize=13, fontweight='bold')
            st.pyplot(fig)
            st.caption("Notice the strong positive correlation of `IsHTTPS`, `LineOfCode`, and `NoOfExternalRef` with legitimate sites (`label=1`), while special char and digit ratios negatively correlate.")

# =============================================================================
# 8. PAGE: 🤖 MODEL PERFORMANCE
# =============================================================================
elif menu == "🤖 Model Performance":
    st.markdown('<p class="main-title">🤖 Model Performance & Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Detailed evaluation metrics loaded directly from actual serialized model training.</p>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select Model Evaluation Track:",
        ["Track A: Full Dataset Model (50 Features: URL + Webpage)", "Track B: URL-Only Model (18 Features: Live URL Analyzer)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if "Track A" in model_choice:
        metrics = models_data.get('full_metrics')
        fi_df = models_data.get('full_fi')
        st.markdown("### 📊 Track A: Full Dataset Model (50 Numerical & Structural Features)")
        st.write("Trained across all available numerical features in the PhiUSIIL dataset, including HTML structural signals (LineOfCode, ExternalRef, Favicon, etc.).")
    else:
        metrics = models_data.get('url_metrics')
        fi_df = models_data.get('url_fi')
        st.markdown("### 📊 Track B: URL-Only Model (18 Lexical & Protocol Features)")
        st.write("Trained strictly on features genuinely extractable from a pasted URL string without active web scraping or visiting the destination.")

    if not metrics:
        st.error("Saved model metrics could not be loaded. Please ensure `model/model_metrics.pkl` and `model/url_model_metrics.pkl` are generated.")
    else:
        # Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")

        st.markdown("---")
        
        # Confusion Matrix and Feature Importance Columns
        col_cm, col_fi = st.columns([1, 1])
        
        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix (Actual Test Split)")
            cm = np.array(metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Phishing (0)', 'Legitimate (1)'],
                yticklabels=['Phishing (0)', 'Legitimate (1)']
            )
            ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
            ax.set_ylabel("True Class", fontsize=11, fontweight='bold')
            ax.set_title("Confusion Matrix (47,159 Test Samples)", fontsize=12, fontweight='bold')
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation:**
            - **True Negatives (Top-Left):** Phishing websites accurately identified as Phishing.
            - **False Positives (Top-Right):** Phishing websites misclassified as Legitimate.
            - **False Negatives (Bottom-Left):** Legitimate websites misclassified as Phishing.
            - **True Positives (Bottom-Right):** Legitimate websites accurately identified as Legitimate.
            """)

        with col_fi:
            st.markdown("#### 🏆 Top 10 Feature Importance")
            if fi_df is not None:
                top_10 = fi_df.head(10)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.barplot(data=top_10, x='Importance', y='Feature', palette='viridis', ax=ax)
                ax.set_xlabel("Gini Feature Importance", fontsize=11)
                ax.set_ylabel("Feature Name", fontsize=11)
                ax.set_title("Top 10 Most Influential Features", fontsize=12, fontweight='bold')
                st.pyplot(fig)
                st.caption("Gini importance measures how effectively each feature splits the data across all decision trees.")

        st.markdown("---")
        
        # Classification Report Table
        st.markdown("#### 📑 Full Classification Report")
        if 'classification_report_str' in metrics:
            st.code(metrics['classification_report_str'], language='text')

        # Cybersecurity Context Box
        st.markdown("### 🛡️ Why Precision & Recall Matter in Cybersecurity")
        st.info("""
        - **Why Accuracy is Not Enough:** In real-world security operations, missing a zero-day phishing campaign (False Negative) can result in a devastating corporate data breach, ransomware installation, or financial wire fraud.
        - **Recall Focus:** Measures the percentage of actual phishing attacks intercepted. High recall ensures threats do not slip through corporate firewalls.
        - **Precision Focus:** Measures the percentage of flagged alerts that are genuine threats. High precision prevents 'alert fatigue' among SOC analysts and ensures benign user traffic is not blocked unnecessarily.
        - **F1-Score Balance:** Represents the harmonic mean between Precision and Recall, guaranteeing balanced performance across both critical dimensions.
        """)

# =============================================================================
# 9. PAGE: 💡 CYBERSECURITY INSIGHTS
# =============================================================================
elif menu == "💡 Cybersecurity Insights":
    st.markdown('<p class="main-title">💡 Cybersecurity & Threat Analytics Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Actionable security findings derived from machine learning analysis of the PhiUSIIL dataset.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 1. The Deception of Lexical Obfuscation")
        st.write("""
        Attackers frequently manipulate URL lexical structure to bypass heuristic gateway filters:
        - **Subdomain Stacking:** Crafting deep subdomains such as `login.paypal.com.account-verification.servehttp.com` where the legitimate brand name appears at the beginning of the string to visually deceive mobile users.
        - **Character Substitution & Obfuscation:** Hex encoding (`%20`, `%2E`), excessive hyphens, and high digit-to-letter ratios are statistically correlated with malicious intent.
        """)

        st.markdown("### 2. HTTPS Does Not Equal Safe")
        st.write("""
        - While unencrypted `http://` sites in the dataset had a near 100% phishing rate, over **49% of all phishing attacks now also utilize HTTPS**.
        - Free automated certificate authorities (such as Let's Encrypt) allow threat actors to acquire valid SSL/TLS certificates instantly.
        - **Key Takeaway:** End-users must be educated that the browser padlock icon indicates transport encryption, **not** organizational trustworthiness.
        """)

    with c2:
        st.markdown("### 3. Webpage Content vs URL Signals (Defense-in-Depth)")
        st.write("""
        - Analysis of the full 50-feature model revealed that **webpage structural signals** (such as `LineOfCode`, `NoOfExternalRef`, and `NoOfJS`) provide extraordinary discriminative power.
        - Legitimate enterprise portals typically have thousands of lines of code and diverse external resource dependencies (CDNs, analytics, stylesheets).
        - Phishing kits, in contrast, are often lightweight single-page clones with zero or minimal external references.
        """)

        st.markdown("### 4. Operational SOC Recommendations")
        st.write("""
        1. **Pre-Filter Stage:** Employ fast, safe URL-based Random Forest screening to score inbound emails and chat links without initiating network requests.
        2. **Deep Inspection Stage:** For medium-risk URLs (Risk Score 31-60), dispatch automated headless browser sandboxes to inspect webpage DOM, external references, and certificate age.
        3. **Human-in-the-Loop:** High-confidence malicious URLs (Risk Score > 60) should be instantly quarantined and submitted to centralized threat intelligence feeds.
        """)

# =============================================================================
# 10. PAGE: ℹ️ ABOUT PROJECT
# =============================================================================
elif menu == "ℹ️ About Project":
    st.markdown('<p class="main-title">ℹ️ About CyberShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">IBM Data Analytics Academic Internship Capstone Project</p>', unsafe_allow_html=True)

    col_meta1, col_meta2 = st.columns([2, 1])
    
    with col_meta1:
        st.markdown("### 📌 Project Synopsis")
        st.write("""
        **CyberShield** is an academic cybersecurity analytics platform developed as part of the **IBM Data Analytics Academic Internship**.
        The project tackles the escalating threat of credential harvesting and identity theft by applying supervised machine learning to classify websites as **Legitimate** or **Phishing**.
        
        Rather than treating the URL as an opaque string or fabricating unextractable webpage features, CyberShield implements a transparent, dual-track data science methodology:
        - A **Full 50-Feature Model** benchmarks research performance on the complete dataset.
        - A **URL-Only 18-Feature Model** powers a genuine live screening tool for real-world URL triage.
        """)

        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        | Layer | Technology | Purpose |
        |---|---|---|
        | **Language** | Python 3.14 | Core programming environment |
        | **Data Manipulation** | Pandas & NumPy | High-performance tabular data wrangling |
        | **Machine Learning** | Scikit-learn | RandomForestClassifier, train/test split, metrics |
        | **Visualization** | Matplotlib & Seaborn | Statistical distributions, heatmaps, bar charts |
        | **Model Serialization** | Joblib | High-efficiency pickle serialization of models |
        | **User Interface** | Streamlit | Reactive web dashboard and analytics application |
        """)

    with col_meta2:
        st.markdown("### 📚 Dataset Attribution")
        st.info("""
        **PhiUSIIL Phishing URL Dataset**
        - **Source:** UCI Machine Learning Repository
        - **Dataset ID:** 967
        - **Instances:** 235,795
        - **Features:** 54+ Attributes
        - **Citation:** Prasad, A., & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website) Dataset*. UCI Machine Learning Repository.
        """)

    st.markdown("---")
    st.markdown("### 🎓 Academic Viva Defense Guide")
    st.markdown("""
    When presenting this project in a viva or academic review, highlight the following foundational concepts:
    1. **Data Fundamentals:** Why the dataset contains 235,795 rows and 54 features, and why no synthetic or fake data was used.
    2. **Data Cleaning Decisions:** Why non-predictive metadata (`FILENAME`, raw strings) were excluded to avoid data leakage and artificial memorization.
    3. **Stratified Splitting:** Why stratification was used to maintain the exact 57.2% / 42.8% class ratio between training and testing sets.
    4. **Algorithm Choice:** Why Random Forest was chosen over a single Decision Tree (ensemble bagging reduces variance, resists overfitting, provides native feature importance).
    5. **Evaluation Metrics:** Why Accuracy alone is deceptive in security and why Precision, Recall, and F1-Score are the primary standards for risk evaluation.
    """)
