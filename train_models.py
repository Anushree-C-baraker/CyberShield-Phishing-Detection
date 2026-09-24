"""
CyberShield - Model Training & Evaluation Script
Trains both the Full Dataset Model (for comprehensive research analytics)
and the URL-Only Model (for transparent, genuine live URL analysis).
All metrics are saved directly from actual model evaluation.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def main():
    # Resolve directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'phishing_dataset.csv')
    model_dir = os.path.join(base_dir, 'model')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Dataset overview
    total_records = len(df)
    legitimate_count = int((df['label'] == 1).sum())
    phishing_count = int((df['label'] == 0).sum())
    print(f"Total: {total_records} | Legitimate: {legitimate_count} | Phishing: {phishing_count}")

    # Metadata columns to exclude from training
    non_feature_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']

    # =========================================================================
    # 1. FULL DATASET MODEL (50 numerical features)
    # =========================================================================
    full_feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"\nTraining Full Model with {len(full_feature_cols)} features...")
    
    X_full = df[full_feature_cols]
    y = df['label']

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Full Train size: {len(X_train_full)}, Test size: {len(X_test_full)}")

    full_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    full_rf.fit(X_train_full, y_train)
    full_preds = full_rf.predict(X_test_full)

    full_acc = float(accuracy_score(y_test, full_preds))
    full_prec = float(precision_score(y_test, full_preds))
    full_rec = float(recall_score(y_test, full_preds))
    full_f1 = float(f1_score(y_test, full_preds))
    full_cm = confusion_matrix(y_test, full_preds).tolist()
    full_report = classification_report(y_test, full_preds, target_names=['Phishing (0)', 'Legitimate (1)'], digits=4)
    full_report_dict = classification_report(y_test, full_preds, target_names=['Phishing (0)', 'Legitimate (1)'], output_dict=True)

    print("\n--- FULL MODEL PERFORMANCE ---")
    print(f"Accuracy:  {full_acc:.4f}")
    print(f"Precision: {full_prec:.4f}")
    print(f"Recall:    {full_rec:.4f}")
    print(f"F1-Score:  {full_f1:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(full_cm))
    print("\nClassification Report:\n", full_report)

    # Save Full Model Feature Importance
    importances_full = full_rf.feature_importances_
    fi_full_df = pd.DataFrame({
        'Feature': full_feature_cols,
        'Importance': importances_full
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_full_path = os.path.join(model_dir, 'feature_importance.csv')
    fi_full_df.to_csv(fi_full_path, index=False)
    print(f"Saved feature importance to {fi_full_path}")

    # Save Full Model Metrics
    full_metrics = {
        'model_name': 'RandomForestClassifier (Full Dataset 50 Features)',
        'accuracy': full_acc,
        'precision': full_prec,
        'recall': full_rec,
        'f1_score': full_f1,
        'confusion_matrix': full_cm,
        'classification_report_str': full_report,
        'classification_report_dict': full_report_dict,
        'train_samples': len(X_train_full),
        'test_samples': len(X_test_full),
        'total_records': total_records,
        'legitimate_count': legitimate_count,
        'phishing_count': phishing_count,
        'feature_count': len(full_feature_cols)
    }
    metrics_path = os.path.join(model_dir, 'model_metrics.pkl')
    joblib.dump(full_metrics, metrics_path)
    print(f"Saved model metrics to {metrics_path}")

    # Save Full Model & Columns
    full_model_path = os.path.join(model_dir, 'phishing_model.pkl')
    joblib.dump(full_rf, full_model_path)
    print(f"Saved phishing model to {full_model_path}")

    feat_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    joblib.dump(full_feature_cols, feat_cols_path)
    print(f"Saved feature columns to {feat_cols_path}")

    # =========================================================================
    # 2. URL-ONLY MODEL (Features extractable from raw URL string)
    # =========================================================================
    url_cols = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain',
        'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
        'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
        'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
        'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
    ]
    print(f"\nTraining URL-Only Model with {len(url_cols)} URL features...")
    
    X_url = df[url_cols]
    X_train_url, X_test_url, _, _ = train_test_split(
        X_url, y, test_size=0.20, random_state=42, stratify=y
    )

    url_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    url_rf.fit(X_train_url, y_train)
    url_preds = url_rf.predict(X_test_url)

    url_acc = float(accuracy_score(y_test, url_preds))
    url_prec = float(precision_score(y_test, url_preds))
    url_rec = float(recall_score(y_test, url_preds))
    url_f1 = float(f1_score(y_test, url_preds))
    url_cm = confusion_matrix(y_test, url_preds).tolist()
    url_report = classification_report(y_test, url_preds, target_names=['Phishing (0)', 'Legitimate (1)'], digits=4)
    url_report_dict = classification_report(y_test, url_preds, target_names=['Phishing (0)', 'Legitimate (1)'], output_dict=True)

    print("\n--- URL-ONLY MODEL PERFORMANCE ---")
    print(f"Accuracy:  {url_acc:.4f}")
    print(f"Precision: {url_prec:.4f}")
    print(f"Recall:    {url_rec:.4f}")
    print(f"F1-Score:  {url_f1:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(url_cm))
    print("\nClassification Report:\n", url_report)

    # Save URL Model Feature Importance
    importances_url = url_rf.feature_importances_
    fi_url_df = pd.DataFrame({
        'Feature': url_cols,
        'Importance': importances_url
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_url_path = os.path.join(model_dir, 'url_feature_importance.csv')
    fi_url_df.to_csv(fi_url_path, index=False)
    print(f"Saved URL feature importance to {fi_url_path}")

    # Save URL Model Metrics
    url_metrics = {
        'model_name': 'RandomForestClassifier (URL-Only 18 Features)',
        'accuracy': url_acc,
        'precision': url_prec,
        'recall': url_rec,
        'f1_score': url_f1,
        'confusion_matrix': url_cm,
        'classification_report_str': url_report,
        'classification_report_dict': url_report_dict,
        'train_samples': len(X_train_url),
        'test_samples': len(X_test_url),
        'feature_count': len(url_cols)
    }
    url_metrics_path = os.path.join(model_dir, 'url_model_metrics.pkl')
    joblib.dump(url_metrics, url_metrics_path)
    print(f"Saved URL model metrics to {url_metrics_path}")

    # Save URL Model & Columns
    url_model_path = os.path.join(model_dir, 'url_model.pkl')
    joblib.dump(url_rf, url_model_path)
    print(f"Saved url model to {url_model_path}")

    url_cols_path = os.path.join(model_dir, 'url_feature_columns.pkl')
    joblib.dump(url_cols, url_cols_path)
    print(f"Saved url feature columns to {url_cols_path}")

    print("\n=== Model training and serialization completed successfully! ===")

if __name__ == '__main__':
    main()
