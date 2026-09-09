import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def evaluate_shap_faithfulness(model, X_test, y_test, shap_values, step_percentile=10):
    """
    Quantitative XAI Faithfulness Evaluation Engine using ROAR (Remove and Retrain)
    Deletion and Insertion curves based on SHAP feature importance rankings.
    """
    if isinstance(X_test, pd.DataFrame):
        X_mat = X_test.values
    else:
        X_mat = np.array(X_test)
        
    y_true = np.array(y_test)
    n_samples, n_features = X_mat.shape
    
    # Calculate global mean feature importance from SHAP values
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).reshape(-1, n_features).mean(axis=0)

    # Rank features by importance (descending)
    ranked_indices = np.argsort(mean_shap)[::-1]
    
    steps = list(range(0, 101, step_percentile))
    deletion_accs = []
    insertion_accs = []
    
    # Baseline mean imputation vector
    col_means = np.mean(X_mat, axis=0)
    
    for step in steps:
        k = int((step / 100.0) * n_features)
        
        # Deletion: Remove top k important features (replace with column mean)
        X_del = X_mat.copy()
        if k > 0:
            top_k_idx = ranked_indices[:k]
            X_del[:, top_k_idx] = col_means[top_k_idx]
            
        del_preds = model.predict(X_del)
        if len(del_preds.shape) > 1 and del_preds.shape[1] > 1:
            del_acc = accuracy_score(y_true, np.argmax(del_preds, axis=1))
        else:
            del_acc = accuracy_score(y_true, del_preds)
        deletion_accs.append(del_acc)
        
        # Insertion: Start with all mean imputed, keep top k important features original
        X_ins = np.tile(col_means, (n_samples, 1))
        if k > 0:
            top_k_idx = ranked_indices[:k]
            X_ins[:, top_k_idx] = X_mat[:, top_k_idx]
            
        ins_preds = model.predict(X_ins)
        if len(ins_preds.shape) > 1 and ins_preds.shape[1] > 1:
            ins_acc = accuracy_score(y_true, np.argmax(ins_preds, axis=1))
        else:
            ins_acc = accuracy_score(y_true, ins_preds)
        insertion_accs.append(ins_acc)
        
    return {
        "steps": steps,
        "deletion_accs": deletion_accs,
        "insertion_accs": insertion_accs,
        "ranked_indices": ranked_indices
    }

def plot_faithfulness_curves(steps, deletion_accs, insertion_accs, title="SHAP Faithfulness Evaluation", save_path=None):
    """Plot Deletion vs Insertion Curves."""
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(steps, [d * 100 if max(deletion_accs) <= 1.0 else d for d in deletion_accs], 'r-o', linewidth=2.5, label='Deletion Curve (Removing Top SHAP Features)')
    plt.plot(steps, [i * 100 if max(insertion_accs) <= 1.0 else i for i in insertion_accs], 'g-s', linewidth=2.5, label='Insertion Curve (Adding Top SHAP Features)')
    plt.xlabel('Percentage of Features Modified (%)', fontsize=12)
    plt.ylabel('Model Classification Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
