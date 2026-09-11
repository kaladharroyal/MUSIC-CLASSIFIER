"""
Quantitative XAI Faithfulness Evaluation Engine
Supports both:
1. Genuine ROAR (Remove and Retrain): Retrains model at each percentile deletion/insertion step.
2. SHAP Feature Masking / Perturbation: Fast mean-imputation evaluation without retraining.
Handles arbitrary SHAP multi-class output formats (lists, 3D ndarrays, Explanation objects).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
import copy
import io
import json


def patch_shap_for_xgboost():
    """
    Patches shap.explainers._tree.XGBTreeModelLoader to support multi-class XGBoost >= 2.0/3.0
    where learner_model_param['base_score'] is serialized as a JSON vector string (e.g. '[1.4e-3, ...]').
    """
    try:
        import shap.explainers._tree as st
        import scipy.special

        orig_init = st.XGBTreeModelLoader.__init__

        def patched_xgb_loader_init(self, xgb_model):
            import xgboost as xgb
            st._check_xgboost_version(xgb.__version__)
            model = xgb_model

            raw = xgb_model.save_raw(raw_format="ubj")
            with io.BytesIO(raw) as fd:
                jmodel = st.decode_ubjson_buffer(fd)

            learner = jmodel["learner"]
            learner_model_param = learner["learner_model_param"]
            objective = learner["objective"]

            booster = learner["gradient_booster"]
            n_classes = max(int(learner_model_param["num_class"]), 1)
            n_targets = max(int(learner_model_param["num_target"]), 1)
            n_targets = max(n_targets, n_classes)

            if "gbtree" in booster and "model" not in booster:
                booster = booster["gbtree"]
            if booster["model"].get("iteration_indptr", None) is not None:
                iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
                diff = np.diff(iteration_indptr)
            else:
                n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
                diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())
            if np.any(diff != diff[0]):
                raise ValueError("vector-leaf is not yet supported.:", diff)

            self.n_trees_per_iter = int(diff[0])
            self.n_targets = n_targets

            # Parse base_score safely
            bs_val = learner_model_param["base_score"]
            if isinstance(bs_val, str) and (bs_val.startswith("[") or "," in bs_val):
                try:
                    bs_parsed = json.loads(bs_val)
                    base_score = float(bs_parsed[0])
                except Exception:
                    bs_parsed = [float(x) for x in bs_val.strip("[]").split(",") if x.strip()]
                    base_score = float(bs_parsed[0])
            else:
                base_score = float(bs_val)

            self.base_score = base_score
            assert self.n_trees_per_iter > 0

            self.name_obj = objective["name"]
            self.name_gbm = booster["name"]
            if self.name_obj in ("binary:logistic", "reg:logistic"):
                self.base_score = scipy.special.logit(base_score)
            elif self.name_obj in (
                "reg:gamma",
                "reg:tweedie",
                "count:poisson",
                "survival:cox",
                "survival:aft",
            ):
                self.base_score = np.log(self.base_score)
            else:
                self.base_score = base_score

            self.num_feature = int(learner_model_param["num_feature"])
            self.num_class = int(learner_model_param["num_class"])

            trees = booster["model"]["trees"]
            self.num_trees = len(trees)

            self.node_parents = []
            self.node_cleft = []
            self.node_cright = []
            self.node_sindex = []
            self.children_default = []
            self.sum_hess = []

            self.values = []
            self.thresholds = []
            self.threshold_types = []
            self.features = []

            self.split_types = []
            self.categories = []

            feature_types = model.feature_types
            if feature_types is not None:
                cat_feature_indices = np.where(np.asarray(feature_types) == "c")[0]
                if len(cat_feature_indices) == 0:
                    self.cat_feature_indices = None
                else:
                    self.cat_feature_indices = cat_feature_indices
            else:
                self.cat_feature_indices = None

            def to_integers(data):
                assert isinstance(data, list)
                return np.asanyarray(data, dtype=np.uint8)

            for i in range(self.num_trees):
                tree = trees[i]
                parents = np.asarray(tree["parents"])
                self.node_parents.append(parents)
                self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
                self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
                self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

                base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
                if base_weight.size != self.node_cleft[-1].size:
                    raise ValueError("vector-leaf is not yet supported.")

                default_left = to_integers(tree["default_left"])
                default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
                self.children_default.append(default_child)
                self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

                is_leaf = self.node_cleft[-1] == -1

                split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
                leaf_weight = np.where(is_leaf, split_cond, 0.0)
                thresholds = np.where(is_leaf, 0.0, split_cond)

                thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
                threshold_types = np.zeros_like(thresholds, dtype=np.int32)

                self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
                self.thresholds.append(thresholds)
                self.threshold_types.append(threshold_types)

                split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
                self.features.append(split_idx)

                split_types = to_integers(tree["split_type"])
                self.split_types.append(split_types)
                cat_segments = tree["categories_segments"]
                cat_sizes = tree["categories_sizes"]
                cat_nodes = tree["categories_nodes"]
                cats = tree["categories"]

                tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
                self.categories.append(tree_categories)

        st.XGBTreeModelLoader.__init__ = patched_xgb_loader_init
    except Exception:
        pass


# Apply patch upon module import
patch_shap_for_xgboost()



def extract_mean_shap_importance(shap_values, n_features):
    """
    Extracts 1D global mean absolute feature importance across diverse SHAP output formats:
    - list of 2D arrays (one per class)
    - 3D numpy array (samples, features, classes) or (samples, classes, features)
    - 2D numpy array (samples, features)
    - SHAP Explanation object
    """
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    if isinstance(vals, list):
        # List of arrays per class (e.g. multi-class TreeExplainer in older/newer versions)
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in vals], axis=0)
    elif isinstance(vals, np.ndarray):
        if vals.ndim == 3:
            # Check shape orientation
            if vals.shape[1] == n_features:
                mean_shap = np.abs(vals).mean(axis=(0, 2))
            elif vals.shape[2] == n_features:
                mean_shap = np.abs(vals).mean(axis=(0, 1))
            else:
                mean_shap = np.abs(vals).reshape(-1, n_features).mean(axis=0)
        elif vals.ndim == 2:
            mean_shap = np.abs(vals).mean(axis=0)
        else:
            mean_shap = np.abs(vals).reshape(-1, n_features).mean(axis=0)
    else:
        mean_shap = np.abs(np.array(vals)).reshape(-1, n_features).mean(axis=0)

    # Ensure shape matches n_features
    if len(mean_shap) != n_features:
        mean_shap = np.resize(mean_shap, n_features)

    return mean_shap


def evaluate_shap_masking_faithfulness(model, X_test, y_test, shap_values, step_percentile=10):
    """
    SHAP Feature Masking / Perturbation Faithfulness (Mean Imputation without retraining).
    """
    if isinstance(X_test, pd.DataFrame):
        X_mat = X_test.values.astype(np.float32)
    else:
        X_mat = np.array(X_test, dtype=np.float32)

    y_true = np.array(y_test)
    n_samples, n_features = X_mat.shape

    mean_shap = extract_mean_shap_importance(shap_values, n_features)
    ranked_indices = np.argsort(mean_shap)[::-1]

    steps = list(range(0, 101, step_percentile))
    deletion_accs = []
    insertion_accs = []

    col_means = np.mean(X_mat, axis=0)

    for step in steps:
        k = int(round((step / 100.0) * n_features))

        # Deletion: Mask top-k most important features with column mean
        X_del = X_mat.copy()
        if k > 0:
            top_k_idx = ranked_indices[:k]
            X_del[:, top_k_idx] = col_means[top_k_idx]

        del_preds = model.predict(X_del)
        if len(getattr(del_preds, "shape", [])) > 1 and del_preds.shape[1] > 1:
            del_acc = accuracy_score(y_true, np.argmax(del_preds, axis=1))
        else:
            del_acc = accuracy_score(y_true, del_preds)
        deletion_accs.append(del_acc)

        # Insertion: Start from all column means, insert top-k features
        X_ins = np.tile(col_means, (n_samples, 1))
        if k > 0:
            top_k_idx = ranked_indices[:k]
            X_ins[:, top_k_idx] = X_mat[:, top_k_idx]

        ins_preds = model.predict(X_ins)
        if len(getattr(ins_preds, "shape", [])) > 1 and ins_preds.shape[1] > 1:
            ins_acc = accuracy_score(y_true, np.argmax(ins_preds, axis=1))
        else:
            ins_acc = accuracy_score(y_true, ins_preds)
        insertion_accs.append(ins_acc)

    return {
        "steps": steps,
        "deletion_accs": deletion_accs,
        "insertion_accs": insertion_accs,
        "ranked_indices": ranked_indices,
        "method": "SHAP Masking (Mean Imputation)"
    }


def evaluate_roar_retraining(
    model_or_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    shap_values,
    step_percentile=10,
    device="cpu"
):
    """
    True ROAR (Remove and Retrain) Faithfulness Engine.
    For each percentile k, removes the top k% features from train and test,
    retrains a fresh model on remaining features, and evaluates accuracy.
    """
    X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train, dtype=np.float32)
    X_te = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test, dtype=np.float32)
    y_tr = np.array(y_train)
    y_te = np.array(y_test)

    n_features = X_tr.shape[1]
    mean_shap = extract_mean_shap_importance(shap_values, n_features)
    ranked_indices = np.argsort(mean_shap)[::-1]

    steps = list(range(0, 101, step_percentile))
    deletion_accs = []
    insertion_accs = []

    for step in steps:
        k = int(round((step / 100.0) * n_features))

        # --- DELETION: Train on features EXCLUDING top k ---
        keep_del_indices = ranked_indices[k:] if k < n_features else []
        if len(keep_del_indices) == 0:
            # No features remaining -> Majority class baseline
            majority_class = np.bincount(y_tr).argmax()
            del_acc = accuracy_score(y_te, np.full_like(y_te, majority_class))
        else:
            X_tr_del = X_tr[:, keep_del_indices]
            X_te_del = X_te[:, keep_del_indices]
            
            if callable(model_or_fn):
                clf = model_or_fn()
            else:
                clf = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    random_state=42, eval_metric='mlogloss', tree_method='hist',
                    device=device
                )
            clf.fit(X_tr_del, y_tr)
            preds = clf.predict(X_te_del)
            del_acc = accuracy_score(y_te, preds)
        deletion_accs.append(del_acc)

        # --- INSERTION: Train ONLY on top k features ---
        keep_ins_indices = ranked_indices[:k] if k > 0 else []
        if len(keep_ins_indices) == 0:
            # No features added yet -> Majority class baseline
            majority_class = np.bincount(y_tr).argmax()
            ins_acc = accuracy_score(y_te, np.full_like(y_te, majority_class))
        else:
            X_tr_ins = X_tr[:, keep_ins_indices]
            X_te_ins = X_te[:, keep_ins_indices]
            
            if callable(model_or_fn):
                clf = model_or_fn()
            else:
                clf = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    random_state=42, eval_metric='mlogloss', tree_method='hist',
                    device=device
                )
            clf.fit(X_tr_ins, y_tr)
            preds = clf.predict(X_te_ins)
            ins_acc = accuracy_score(y_te, preds)
        insertion_accs.append(ins_acc)

    return {
        "steps": steps,
        "deletion_accs": deletion_accs,
        "insertion_accs": insertion_accs,
        "ranked_indices": ranked_indices,
        "method": "True ROAR (Remove and Retrain)"
    }


def evaluate_shap_faithfulness(
    model,
    X_test,
    y_test,
    shap_values,
    step_percentile=10,
    X_train=None,
    y_train=None,
    mode="retrain",
    device="cpu"
):
    """
    Unified entry point for SHAP Faithfulness Evaluation.
    If mode='retrain' and (X_train, y_train) are provided, runs True ROAR.
    Otherwise runs SHAP feature masking (mean imputation).
    """
    if mode == "retrain" and X_train is not None and y_train is not None:
        return evaluate_roar_retraining(
            model, X_train, y_train, X_test, y_test, shap_values,
            step_percentile=step_percentile, device=device
        )
    return evaluate_shap_masking_faithfulness(
        model, X_test, y_test, shap_values, step_percentile=step_percentile
    )


def plot_faithfulness_curves(
    steps,
    deletion_accs,
    insertion_accs,
    title="SHAP Faithfulness Evaluation (ROAR Curves)",
    save_path=None
):
    """Plot publication-quality Deletion vs Insertion Curves."""
    plt.figure(figsize=(9, 5), dpi=300)
    d_vals = [d * 100 if max(deletion_accs) <= 1.0 else d for d in deletion_accs]
    i_vals = [i * 100 if max(insertion_accs) <= 1.0 else i for i in insertion_accs]

    plt.plot(steps, d_vals, 'r-o', linewidth=2.5, label='Deletion Curve (Removing Top SHAP Features)')
    plt.plot(steps, i_vals, 'g-s', linewidth=2.5, label='Insertion Curve (Adding Top SHAP Features)')
    plt.xlabel('Percentage of Features Modified (%)', fontsize=12)
    plt.ylabel('Model Classification Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

