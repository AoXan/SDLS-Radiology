import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device).eval()
    
    df = pd.read_csv(args.input_csv)
    texts = df['generated_report'].fillna("").tolist()
    probs = []
    
    for i in tqdm(range(0, len(texts), 32)):
        batch = texts[i:i+32]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy() # Class 1 = History
            probs.extend(p)
            
    df['filbert_prob'] = probs
    df.to_csv(args.output_csv, index=False)
    print(f"Avg Prob: {df['filbert_prob'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_path", default="rajpurkarlab/filbert")
    args = parser.parse_args()
    main(args)