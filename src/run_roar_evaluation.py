"""
Runner Script for ROAR XAI Faithfulness Evaluation
Loads GTZAN features, trains XGBoost, calculates SHAP values,
computes true ROAR Deletion and Insertion curves (retraining at each step)
via faithfulness_eval.py, and saves the publication-quality plot figure to 'docs/figures/roar_faithfulness_curves.png'.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add src dir to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faithfulness_eval import evaluate_shap_faithfulness, plot_faithfulness_curves


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

    # Standardized 70% train, 15% val, 15% test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training XGBoost Classifier on device: {xgb_device}...")
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric='mlogloss', tree_method='hist',
        device=xgb_device
    )
    xgb_model.fit(X_train_scaled, y_train)

    acc = xgb_model.score(X_test_scaled, y_test)
    print(f"XGBoost Test Accuracy: {acc*100:.2f}%")

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_scaled)

    print("Running Genuine ROAR (Remove and Retrain) Faithfulness Engine...")
    results = evaluate_shap_faithfulness(
        xgb_model, X_test_scaled, y_test, shap_values,
        step_percentile=10,
        X_train=X_train_scaled, y_train=y_train,
        mode="retrain",
        device=xgb_device
    )

    steps = results['steps']
    del_accs = results['deletion_accs']
    ins_accs = results['insertion_accs']

    print(f"\n--- {results.get('method', 'ROAR')} Metrics ---")
    for s, d, i in zip(steps, del_accs, ins_accs):
        print(f"Modified {s:3d}% features -> Deletion Acc: {d*100:5.2f}% | Insertion Acc: {i*100:5.2f}%")

    out_img = os.path.join("docs", "figures", "roar_faithfulness_curves.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plot_faithfulness_curves(
        steps, del_accs, ins_accs,
        title="SHAP Quantitative Faithfulness Evaluation via ROAR (Remove and Retrain)",
        save_path=out_img
    )
    print(f"\nROAR Faithfulness Curves saved to '{out_img}' successfully!")


if __name__ == "__main__":
    main()
