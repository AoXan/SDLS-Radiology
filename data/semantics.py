# data/semantics.py
"""
Stage II: Semantic Decomposition (LLM-based)
--------------------------------------------
Uses a local LLM (Ollama) to decompose complex radiology impressions into 
atomic, semantically labeled clinical findings.

Paper Reference: Section III-B (Semantic Decomposition)
Input: multimodal_pairs.csv
Output: disentangled_findings.csv (Rows = Atomic Findings)
"""

import pandas as pd
import ollama
import json
import re
import time
import argparse
import pathlib
from tqdm.auto import tqdm

# Categories defined in SDLS methodology for orthogonality checks
CATEGORIES = [
    "UNCHANGED", 
    "WORSENED / INCREASED", 
    "IMPROVED / DECREASED",
    "DEVICE / PROCEDURE RELATED",
    "NO_CHANGE_PRESENT"
]

PROMPT_TEMPLATE = """
You are an expert radiologist assistant. 
Task: Analyze the following radiology impression and decompose it into a JSON list of atomic findings.
For each finding, classify it into one of these categories: 
{categories}

Input Impression: "{impression}"

Output Format (JSON ONLY):
[
  {{
    "original_finding": "Exact substring from input",
    "rewritten_finding": "Stand-alone clinical fact (current status only)",
    "classification": "CATEGORY"
  }}
]
Do not output markdown code blocks or explanations. Just the JSON string.
"""

def clean_json_output(response_text: str) -> str:
    """
    Robustly extracts JSON list from LLM response (handles chatty prefix/suffix).
    """
    # Attempt 1: Direct JSON parsing
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass

    # Attempt 2: Regex extraction of list pattern [...]
    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if match:
        return match.group(0)
    
    return "[]"

def main(args):
    input_path = pathlib.Path(args.input_csv)
    output_path = pathlib.Path(args.output_csv)
    
    print(f"[Stage II] Loading pairs from {input_path}")
    df = pd.read_csv(input_path)
    
    # We need a diverse set of samples per category to build robust vectors
    # We will process reports until we reach N samples per category or exhaust data
    category_counts = {k: 0 for k in CATEGORIES}
    final_results = []
    
    print(f"[Stage II] Starting decomposition with model: {args.model}")
    pbar = tqdm(total=args.samples_per_category * len(CATEGORIES))
    
    for _, row in df.iterrows():
        # Stop if all categories are full
        if all(c >= args.samples_per_category for c in category_counts.values()):
            print("All semantic categories filled.")
            break
            
        impression = row.get('impression_with_history', '')
        if not isinstance(impression, str) or len(impression) < 10:
            continue

        # Prepare Prompt
        prompt = PROMPT_TEMPLATE.format(
            categories=CATEGORIES,
            impression=impression.replace('"', "'")
        )
        
        try:
            # Call Ollama
            response = ollama.chat(model=args.model, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            
            # Parse
            cleaned_json = clean_json_output(content)
            findings = json.loads(cleaned_json)
            
            for f in findings:
                cat = f.get('classification', 'OTHER')
                
                # Normalize category string
                matched_cat = next((c for c in CATEGORIES if c in cat.upper()), None)
                
                if matched_cat and category_counts[matched_cat] < args.samples_per_category:
                    final_results.append({
                        'dicom_id': row['dicom_id'],
                        'image_path': row['image_path'],
                        'original_finding': f.get('original_finding'),
                        'rewritten_finding': f.get('rewritten_finding'),
                        'semantic_category': matched_cat
                    })
                    category_counts[matched_cat] += 1
                    pbar.update(1)
                    
            # Rate limit protection for local inference
            time.sleep(0.1)
            
        except Exception as e:
            # print(f"Error processing row: {e}") # Optional verbose logging
            continue

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_results).to_csv(output_path, index=False)
    
    print(f"\n[Success] Semantic decomposition complete.")
    print(f"Saved {len(final_results)} atomic findings to {output_path}")
    print("Distribution:", category_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to multimodal_pairs.csv")
    parser.add_argument("--output_csv", default="./data_processed/disentangled_findings.csv")
    parser.add_argument("--model", default="llama3", help="Ollama model name (e.g., llama3, mistral)")
    parser.add_argument("--samples_per_category", type=int, default=50, help="Target samples per semantic type")
    args = parser.parse_args()
    main(args)