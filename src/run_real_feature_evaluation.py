"""
Evaluation Script for Real Extracted Features on MTG-Jamendo & MagnaTagATune
Loads 'Data/processed/jamendo_real_features.csv' and 'Data/processed/mtat_real_features.csv',
merges with ground truth annotations, trains authentic XGBoost models,
calculates real ROC-AUC scores and test accuracies, and generates SHAP explanations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import shap


def evaluate_jamendo_real():
    csv_path = os.path.join("Data", "processed", "jamendo_real_features.csv")
    meta_path = os.path.join("Data", "research_external", "MTG-Jamendo", "jamendo_gtzan_single_label.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: '{csv_path}'. Run extraction first.")
        return None

    print(f"\n=======================================================")
    print(f"--- Evaluating Authentic MTG-Jamendo Real Features ---")
    print(f"=======================================================")
    df_feat = pd.read_csv(csv_path)
    print(f"Loaded Jamendo extracted features: {df_feat.shape}")

    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path)
        df_feat['track_id'] = df_feat['filename'].apply(
            lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else -1
        )
        df = pd.merge(df_feat, df_meta[['track_id', 'label']], on='track_id', how='inner')
        print(f"Merged with GTZAN-subset metadata: {len(df)} matching tracks.")
    else:
        df = df_feat.copy()
        if 'label' not in df.columns:
            print("Error: No label metadata found for Jamendo.")
            return None

    drop_cols = ['filename', 'filepath', 'track_id', 'label']
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # Standardized 70% train, 15% val, 15% test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric='mlogloss', tree_method='hist',
        device=device
    )
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs, multi_class='ovr')
    except Exception as e:
        auc = float('nan')

    print(f"MTG-Jamendo Real Feature XGBoost Test Accuracy: {acc*100:.2f}%")
    print(f"MTG-Jamendo Real Feature XGBoost ROC-AUC Score: {auc:.3f} (Paper Reference: 0.729)")

    # SHAP Feature Importance Plot
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_s)
        
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(
            shap_vals, pd.DataFrame(X_test_s, columns=feat_cols),
            class_names=le.classes_, plot_type="bar", show=False
        )
        plt.title("Authentic MTG-Jamendo SHAP Feature Importance (Signal + Mid-Level)", fontsize=12)
        plt.tight_layout()
        out_img = os.path.join("docs", "figures", "jamendo_real_shap.png")
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        plt.savefig(out_img)
        plt.close()
        print(f"Saved Jamendo SHAP importance figure to '{out_img}'")
    except Exception as e:
        print(f"SHAP plot warning: {e}")

    return {"accuracy": acc, "roc_auc": auc, "model": model, "feature_names": feat_cols}


def evaluate_mtat_real():
    csv_path = os.path.join("Data", "processed", "mtat_real_features.csv")
    meta_path = os.path.join("Data", "research_external", "MagnaTagATune", "annotations_final.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: '{csv_path}'. Run extraction first.")
        return None

    print(f"\n=========================================================")
    print(f"--- Evaluating Authentic MagnaTagATune Real Features ---")
    print(f"=========================================================")
    df_feat = pd.read_csv(csv_path)
    print(f"Loaded MagnaTagATune extracted features: {df_feat.shape}")

    if not os.path.exists(meta_path):
        print(f"Error: MagnaTagATune annotations not found at '{meta_path}'.")
        return None

    df_meta = pd.read_csv(meta_path, sep="\t" if "\t" in open(meta_path).readline() else ",")
    df_meta['filename'] = df_meta['mp3_path'].apply(os.path.basename)

    target_tags = ['classical', 'opera', 'metal', 'guitar', 'ambient', 'electronic', 'vocal', 'violin', 'piano', 'drums']
    available_tags = [t for t in target_tags if t in df_meta.columns]

    df = pd.merge(df_feat, df_meta[['filename'] + available_tags], on='filename', how='inner')
    print(f"Merged with annotations: {len(df)} matching tracks.")

    drop_cols = ['filename', 'filepath'] + available_tags
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].fillna(0).values
    Y = df[available_tags].values

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag_aucs = []

    for tag_idx, tag in enumerate(available_tags):
        y_train_tag = Y_train[:, tag_idx]
        y_test_tag = Y_test[:, tag_idx]

        if len(np.unique(y_train_tag)) < 2 or len(np.unique(y_test_tag)) < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric='logloss', tree_method='hist',
            device=device
        )
        model.fit(X_train_s, y_train_tag)
        probs = model.predict_proba(X_test_s)[:, 1]

        try:
            auc = roc_auc_score(y_test_tag, probs)
            tag_aucs.append(auc)
            print(f"  Tag '{tag:<12}': Real ROC-AUC = {auc:.3f}")
        except Exception:
            pass

    mean_auc = np.mean(tag_aucs) if len(tag_aucs) > 0 else float('nan')
    print(f"\nMagnaTagATune Overall Multi-Label Real Feature ROC-AUC: {mean_auc:.3f} (Paper Reference: 0.840)")

    # SHAP Contrastive Plot for sample tags (Opera vs Metal)
    for sample_tag in ['classical', 'guitar']:
        if sample_tag in available_tags:
            s_idx = available_tags.index(sample_tag)
            y_tr_s = Y_train[:, s_idx]
            if len(np.unique(y_tr_s)) >= 2:
                model_s = xgb.XGBClassifier(
                    n_estimators=80, max_depth=4, learning_rate=0.05,
                    random_state=42, eval_metric='logloss', tree_method='hist',
                    device=device
                )
                model_s.fit(X_train_s, y_tr_s)
                try:
                    explainer_s = shap.TreeExplainer(model_s)
                    shap_s = explainer_s.shap_values(X_test_s)
                    plt.figure(figsize=(9, 4), dpi=300)
                    shap.summary_plot(shap_s, pd.DataFrame(X_test_s, columns=feat_cols), plot_type="bar", show=False)
                    plt.title(f"MagnaTagATune SHAP Feature Importance for '{sample_tag.upper()}'")
                    plt.tight_layout()
                    out_tag_img = os.path.join("docs", "figures", f"mtat_{sample_tag}_shap.png")
                    os.makedirs(os.path.dirname(out_tag_img), exist_ok=True)
                    plt.savefig(out_tag_img)
                    plt.close()
                except Exception as e:
                    print(f"SHAP tag warning for {sample_tag}: {e}")

    return {"mean_roc_auc": mean_auc, "tag_aucs": tag_aucs, "tags": available_tags}


def main():
    evaluate_jamendo_real()
    evaluate_mtat_real()


if __name__ == "__main__":
    main()
