import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm
try:
    from bert_score import score as bert_score
except: pass
try:
    import radgraph
except: pass
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main(args):
    df = pd.read_csv(args.input_csv).dropna(subset=['generated_report', 'impression_with_history'])
    hyps = [str(x) for x in df['generated_report']]
    refs = [str(x) for x in df['impression_with_history']]
    
    # 1. BERTScore
    try:
        _, _, F1 = bert_score(hyps, refs, lang="en", verbose=True)
        df['BERTScore'] = F1.numpy()
    except:
        df['BERTScore'] = 0.0

    # 2. RadGraph & RadCliQ
    try:
        rg = radgraph.F1RadGraph(reward_level="all")
        # Batch processing recommended for speed
        rg_scores = rg(hyps=hyps, refs=refs)[0] 
        df['RadGraph_F1'] = rg_scores
        
        # RadCliQ
        cc = SmoothingFunction()
        radcliq = []
        for h, r, s in zip(hyps, refs, rg_scores):
            bleu = sentence_bleu([r.split()], h.split(), smoothing_function=cc.method1)
            radcliq.append((s + bleu) / 2)
        df['RadCliQ'] = radcliq
    except Exception as e:
        print(f"RadGraph failed: {e}")
        df['RadGraph_F1'] = 0.0
        df['RadCliQ'] = 0.0

    df.to_csv(args.output_csv, index=False)
    print("Metrics saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    main(args)