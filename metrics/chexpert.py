# metrics/chexpert.py
"""
Metric: Clinical Fidelity (CheXpert Labeler)
--------------------------------------------
Wrapper for the Stanford CheXpert Labeler to compute Macro-F1 and Micro-F1.
Requires the 'chexpert-labeler' tool to be installed/accessible.

Paper Reference: Section III-D (Fidelity Objective), Table V
"""

import pandas as pd
import argparse
import subprocess
import tempfile
import shutil
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

CHEXPERT_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

def run_labeler(texts, labeler_dir):
    """Runs the CheXpert labeler on a list of strings."""
    # This is a simplified wrapper. In production, you might import the python module directly
    # if available, or use subprocess as the original script did.
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "output.csv")
        
        # Prepare input format for labeler (usually requires 'Reports' column)
        pd.DataFrame({"Reports": texts}).to_csv(input_csv, index=False)
        
        # Call external labeler (assuming 'label.py' exists in labeler_dir)
        cmd = [
            "python", os.path.join(labeler_dir, "label.py"),
            "--reports_path", input_csv,
            "--output_path", output_csv,
            "--clean" # Enable text cleaning
        ]
        
        print(f"[CheXpert] Running external labeler...")
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Read back results
        res_df = pd.read_csv(output_csv)
        # Keep only the 14 label columns
        return res_df[CHEXPERT_COLS].fillna(0).replace({-1: 1}) # Treat uncertain as positive? Check paper.
        # Paper setting: "Uncertainty policy set to zero" (Section IV-G)
        # So replace -1 with 0:
        # return res_df[CHEXPERT_COLS].fillna(0).replace({-1: 0})

def main(args):
    df = pd.read_csv(args.input_csv)
    
    # 1. Label Generated Reports
    print("[Metric] Labeling Hypotheses...")
    hyp_labels = run_labeler(df['generated_report'].fillna("").tolist(), args.labeler_dir)
    
    # 2. Label Reference Reports (Ground Truth)
    print("[Metric] Labeling References...")
    ref_labels = run_labeler(df['impression_with_history'].fillna("").tolist(), args.labeler_dir)
    
    # 3. Compute F1
    # Paper Policy: Uncertainty (U) -> 0 (Negative)
    y_pred = hyp_labels.replace(-1, 0).values
    y_true = ref_labels.replace(-1, 0).values
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    print(f"\n[Results] CheXpert Metric:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    
    # Save detailed stats
    results = {
        'Macro_F1': macro_f1,
        'Micro_F1': micro_f1
    }
    pd.DataFrame([results]).to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="chexpert_metrics.csv")
    parser.add_argument("--labeler_dir", required=True, help="Path to stanfordmlgroup/chexpert-labeler folder")
    args = parser.parse_args()
    main(args)