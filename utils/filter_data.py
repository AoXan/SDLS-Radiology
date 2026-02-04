# utils/filter_data.py
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True, help="multimodal_pairs.csv")
    parser.add_argument("--semantics_csv", required=True, help="Output from semantics.py")
    parser.add_argument("--category", required=True, help="e.g., 'UNCHANGED' or 'COMPARISON'")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # Load semantics
    sem_df = pd.read_csv(args.semantics_csv)
    target_ids = sem_df[sem_df['semantic_category'] == args.category]['dicom_id'].unique()
    target_ids = set(str(x) for x in target_ids)
    
    # Filter the main pairs file
    pairs_df = pd.read_csv(args.pairs_csv)
    pairs_df['dicom_id'] = pairs_df['dicom_id'].astype(str)
    
    filtered_df = pairs_df[pairs_df['dicom_id'].isin(target_ids)]
    
    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Filtered {len(filtered_df)} pairs for category '{args.category}'. Saved to {args.output_csv}")

if __name__ == "__main__":
    main()