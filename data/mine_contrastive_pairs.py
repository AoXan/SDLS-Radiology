import pandas as pd
import argparse
import pathlib
import re
import string
from tqdm.auto import tqdm

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def extract_impression(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.strip()
    # Lookahead regex to capture Impression until Findings or End
    pattern = r'(?i)(?:impression|conclusion|summary)\s*[:\-]?(.*?)(?=(?:\n\s*(?:findings|clinical history|indication|technique))|$)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def main(args):
    print(f"[Stage I] Mining contrastive pairs...")
    # Load cleaned (no-history) datasets
    df_train = pd.read_csv(args.clean_train_csv)
    df_test = pd.read_csv(args.clean_test_csv)
    df_clean = pd.concat([df_train, df_test])
    
    # Load original full reports (with history)
    import json
    with open(args.original_json, 'r') as f:
        full_data = json.load(f)
    
    # Map dicom_id -> original report
    original_map = {}
    for split in full_data.values():
        for item in split:
            original_map[str(item['id'])] = item['report']

    contrastive_pairs = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
        dicom_id = str(row['dicom_id'])
        r_curr = str(row['report']) # This is the cleaned "Impression"
        
        if dicom_id not in original_map or not r_curr.strip(): continue

        r_hist_full = original_map[dicom_id]
        r_hist = extract_impression(r_hist_full) # Extract Impression from original
        
        if not r_hist: continue

        # Save only if they differ (meaning history was removed)
        if normalize_text(r_curr) != normalize_text(r_hist):
            contrastive_pairs.append({
                'dicom_id': dicom_id,
                'impression_no_history': r_curr,
                'impression_with_history': r_hist
            })

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(contrastive_pairs).to_csv(output_path, index=False)
    print(f"[Success] Saved {len(contrastive_pairs)} pairs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_train_csv", required=True)
    parser.add_argument("--clean_test_csv", required=True)
    parser.add_argument("--original_json", required=True)
    parser.add_argument("--output_path", default="data/contrastive_pairs.csv")
    args = parser.parse_args()
    main(args)