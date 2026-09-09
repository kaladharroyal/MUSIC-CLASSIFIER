"""
Script to safely append research paper modules, scripts, and manuscript draft cells
to the end of music_classification_fixed.ipynb using nbformat.
"""

from pathlib import Path
import nbformat as nbf

def append_files_to_notebook(notebook_path):
    print(f"Loading notebook '{notebook_path}'...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    print(f"Current notebook cell count: {len(nb.cells)}")

    # 1. Main Header Markdown Cell
    header_md = """# Part 10: Academic Standalone Modules & Complete IEEE Access Manuscript

This section contains all standalone Python research modules, feature extractors, quantitative XAI evaluation engines (ROAR curves), and the complete pre-submission manuscript draft for IEEE Access / ICASSP publication.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))

    # List of files to append: (title, filepath, cell_type)
    files_to_append = [
        (
            "## 10.1 Essentia & Librosa Signal Processing Extractor (`essentia_features.py`)",
            "Extracts 23 signal processing descriptors (*Danceability, Spectral Centroid, Onset Rate, BPM*, etc.) with cross-platform fallback for Windows/Linux/Colab.",
            "src/essentia_features.py",
            "code"
        ),
        (
            "## 10.2 PyTorch Mid-Level Perceptual Feature DNN (`midlevel_vggish.py`)",
            "VGG-ish deep neural network architecture predicting 7 mid-level perceptual features (*Melodiousness, Articulation, Rhythmic Stability, Rhythmic Complexity, Dissonance, Tonal Stability, Minorness*).",
            "src/midlevel_vggish.py",
            "code"
        ),
        (
            "## 10.3 Quantitative XAI Faithfulness Engine (`faithfulness_eval.py`)",
            "Implements Remove and Retrain (ROAR) Deletion & Insertion accuracy curves to mathematically measure SHAP feature fidelity.",
            "src/faithfulness_eval.py",
            "code"
        ),
        (
            "## 10.4 ROAR Faithfulness Curve Generator & Evaluator (`run_roar_evaluation.py`)",
            "Trains XGBoost model on GTZAN, calculates SHAP values, and outputs publication-quality figure `roar_faithfulness_curves.png`.",
            "src/run_roar_evaluation.py",
            "code"
        ),
        (
            "## 10.5 Multi-Dataset Batch Feature Extractor (`extract_all_real_features.py`)",
            "Iterates through local audio folders for MTG-Jamendo and MagnaTagATune to extract authentic 23 signal + 7 mid-level feature vectors.",
            "src/extract_all_real_features.py",
            "code"
        ),
        (
            "## 10.6 Authentic Real Feature Model Evaluator (`run_real_feature_evaluation.py`)",
            "Evaluates Multi-Label and Multi-Class XGBoost models on authentic extracted Jamendo and MagnaTagATune features, outputting ROC-AUC & SHAP plots.",
            "src/run_real_feature_evaluation.py",
            "code"
        ),
        (
            "## 10.7 IEEE Access Academic Manuscript Draft (`manuscript_draft.md`)",
            "Complete pre-submission IEEE Access manuscript draft incorporating theoretical architecture, mathematical equations, multi-dataset benchmarks, and ROAR curves.",
            "docs/manuscript_draft.md",
            "markdown"
        )
    ]

    for section_title, section_desc, fpath, cell_type in files_to_append:
        print(f"Appending '{fpath}'...")
        # Add markdown explanation
        md_text = f"{section_title}\n\n{section_desc}"
        nb.cells.append(nbf.v4.new_markdown_cell(md_text))
        
        p = Path(fpath)
        content = p.read_text(encoding='utf-8') if p.exists() else f"# File {fpath} not found."
        if cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(content))

    # Validate notebook structure
    nbf.validate(nb)
    print(f"Validated notebook structure cleanly! New cell count: {len(nb.cells)}")

    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Successfully updated '{notebook_path}'!")

if __name__ == "__main__":
    nb_path = "music_classification_fixed.ipynb"
    append_files_to_notebook(nb_path)
