import json
import pathlib
import argparse
import pandas as pd
from tqdm.auto import tqdm

def main(args):
    print(f"[Stage I] Linking images...")
    with open(args.annotation_json, 'r') as f:
        ann_data = json.load(f)

    # O(1) Lookup: dicom_id -> first image path
    image_lookup = {}
    for split in ann_data.values():
        for record in split:
            if record.get("image_path"):
                image_lookup[str(record["id"])] = record["image_path"][0]

    df = pd.read_csv(args.input_csv, dtype={'dicom_id': str})
    df['image_path'] = df['dicom_id'].map(image_lookup)
    
    # Filter missing
    df_valid = df.dropna(subset=['image_path'])
    
    output_path = pathlib.Path(args.output_path)
    df_valid.to_csv(output_path, index=False)
    print(f"[Success] Linked {len(df_valid)} pairs. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--output_path", default="data/multimodal_pairs.csv")
    args = parser.parse_args()
    main(args)