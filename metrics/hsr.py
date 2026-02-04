import pandas as pd
import string

TRIGGERS = {
    'Stability': [
    'stable', 'unchanged', 'no change', 'no significant change', 'no interval change',
    'persistent', 'remains', 'similar', 'similarly', 'has not changed', 'is stable',
    'grossly stable', 'essentially unchanged', 'chronic', 'long-standing'
    ], 
    'Comparison': [
    'prior', 'previous', 'compared with', 'compared to', 'since the previous',
    'again seen', 'on the prior', 'from the prior'
    ],
    "Progression": [
    'worsened', 'increased', 'larger', 'more', 'progression', 'development of',
    'now demonstrates', 'interval development'
    ],
    'Improvement': [
    'improved', 'decreased', 'smaller', 'less', 'resolved', 'resolution',
    'less conspicuous', 'clearing'
    ],

    'Negative_Absence': [
    'no prior', 'no comparison', 'no previous', 'without comparison'
    ] 
}
FLAT = [w for cat in TRIGGERS.values() for w in cat]
FLAT.sort(key=lambda x: len(x.split()), reverse=True)

def calculate_hsr(text: str) -> float:
    if not isinstance(text, str): return 0.0
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    if not tokens: return 0.0
    
    is_hist = [False] * len(tokens)
    for trig in FLAT:
        trig_toks = trig.split()
        for i in range(len(tokens) - len(trig_toks) + 1):
            if tokens[i:i+len(trig_toks)] == trig_toks:
                for j in range(i, i+len(trig_toks)): is_hist[j] = True
                
    return sum(is_hist) / len(tokens)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    df['HSR'] = df['generated_report'].apply(calculate_hsr)
    df.to_csv(args.output_csv, index=False)
    print(f"Avg HSR: {df['HSR'].mean():.4f}")