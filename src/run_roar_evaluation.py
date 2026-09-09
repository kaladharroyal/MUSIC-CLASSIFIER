"""
Runner Script for ROAR XAI Faithfulness Evaluation
Loads GTZAN features, trains XGBoost, calculates SHAP values,
computes ROAR Deletion and Insertion curves via faithfulness_eval.py,
and saves the publication-quality plot figure to 'research paper/roar_faithfulness_curves.png'.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add research paper dir to path
from faithfulness_eval import evaluate_shap_faithfulness

def main():
    csv_path = os.path.join("Data", "csv", "features_30_sec.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("Data", "features_30_sec.csv")
        
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    drop_cols = ['filename', 'length', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y_raw = df['label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train_scaled, y_train)
    
    acc = xgb_model.score(X_test_scaled, y_test)
    print(f"XGBoost Test Accuracy: {acc*100:.2f}%")
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    print("Running ROAR Faithfulness Evaluation Engine...")
    results = evaluate_shap_faithfulness(xgb_model, X_test_scaled, y_test, shap_values, step_percentile=10)
    
    steps = results['steps']
    del_accs = results['deletion_accs']
    ins_accs = results['insertion_accs']
    
    print("\n--- ROAR Faithfulness Metrics ---")
    for s, d, i in zip(steps, del_accs, ins_accs):
        print(f"Modified {s}% features -> Deletion Acc: {d*100:.2f}% | Insertion Acc: {i*100:.2f}%")
        
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(steps, [d*100 for d in del_accs], 'r-o', linewidth=2.5, label='Deletion Curve (Removing Top SHAP Features)')
    plt.plot(steps, [i*100 for i in ins_accs], 'g-s', linewidth=2.5, label='Insertion Curve (Adding Top SHAP Features)')
    plt.xlabel('Percentage of Features Modified (%)', fontsize=12)
    plt.ylabel('Model Classification Accuracy (%)', fontsize=12)
    plt.title('SHAP Quantitative Faithfulness Evaluation via ROAR (IEEE Access Benchmark)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_img = os.path.join("docs", "figures", "roar_faithfulness_curves.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img)
    plt.close()
    print(f"\nROAR Faithfulness Curves saved to '{out_img}' successfully!")

if __name__ == "__main__":
    main()
