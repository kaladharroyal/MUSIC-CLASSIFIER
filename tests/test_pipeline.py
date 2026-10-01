import os
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestMusicClassificationPipeline(unittest.TestCase):

    def test_split_indices_integrity(self):
        """Verify split_indices JSON exists and maintains 0 train-val-test overlap."""
        split_paths = [
            REPO_ROOT / "Data" / "metadata" / "split_indices.json",
            REPO_ROOT / "Data" / "metadata" / "split_indices_v2.json"
        ]
        
        checked = 0
        for split_path in split_paths:
            if not split_path.exists():
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            train_idx = set(data.get("train", []))
            val_idx = set(data.get("val", []))
            test_idx = set(data.get("test", []))
            
            # Verify no intersection between partitions (Strict Leakage-Free)
            self.assertEqual(len(train_idx.intersection(val_idx)), 0, f"Leakage between train and val in {split_path}")
            self.assertEqual(len(train_idx.intersection(test_idx)), 0, f"Leakage between train and test in {split_path}")
            self.assertEqual(len(val_idx.intersection(test_idx)), 0, f"Leakage between val and test in {split_path}")
            self.assertGreater(len(train_idx), 0, "Train partition is empty")
            checked += 1
            
        self.assertGreater(checked, 0, "At least one split index file should be checked")

    def test_tabular_features_schema(self):
        """Ensure standard GTZAN feature CSVs contain necessary audio descriptors."""
        csv_paths = [
            REPO_ROOT / "Data" / "csv" / "features_30_sec.csv",
            REPO_ROOT / "Data" / "csv" / "features_3_sec.csv"
        ]
        
        required_cols = ["filename", "length", "label"]
        
        for p in csv_paths:
            if p.exists():
                df = pd.read_csv(p)
                self.assertFalse(df.empty, f"{p.name} is empty")
                for col in required_cols:
                    self.assertIn(col, df.columns, f"Missing required column '{col}' in {p.name}")

    def test_multi_dataset_features_schema(self):
        """Verify MTG-Jamendo and MTAT real feature CSVs if available."""
        processed_dir = REPO_ROOT / "Data" / "processed"
        if not processed_dir.exists():
            return
            
        jamendo_csv = processed_dir / "jamendo_real_features.csv"
        mtat_csv = processed_dir / "mtat_real_features.csv"
        
        if jamendo_csv.exists():
            df_j = pd.read_csv(jamendo_csv)
            self.assertFalse(df_j.empty, "Jamendo features CSV is empty")
            self.assertGreater(len(df_j.columns), 5, "Jamendo features missing descriptor columns")
            
        if mtat_csv.exists():
            df_m = pd.read_csv(mtat_csv)
            self.assertFalse(df_m.empty, "MTAT features CSV is empty")
            self.assertGreater(len(df_m.columns), 5, "MTAT features missing descriptor columns")

    def test_faithfulness_engine_import(self):
        """Ensure faithfulness evaluation engine and shap patchers load smoothly."""
        from src.faithfulness_eval import patch_shap_for_xgboost
        # Test patch execution does not raise unexpected fatal errors
        patch_shap_for_xgboost()


if __name__ == "__main__":
    unittest.main()
