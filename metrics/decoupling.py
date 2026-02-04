# metrics/decoupling.py
"""
Analysis: Metric Decoupling (Table IV)
--------------------------------------
Performs Linear Regression to validate that hallucination suppression (HSR) 
is independent of general semantic similarity (BERTScore).

Paper Reference: Section IV-D-3, Table IV
Dependent Variable: Probability of being history-free (1 - FilBERT_Prob)
Predictors: BERTScore, HSR (token-level), Report Length
"""

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

def run_decoupling_analysis(df: pd.DataFrame, output_dir: pathlib.Path):
    print("--- Table IV: Metric Decoupling Analysis ---")
    
    # 1. Check required columns
    required_cols = ['generated_report', 'HSR', 'filbert_prob', 'BERTScore']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV is missing column: {col}. Please run hsr.py, classifier.py, and clinical.py first.")

    # 2. Prepare Variables
    # Dependent Variable: Probability of being history-free
    # P(No History) = 1 - P(History)
    df['prob_no_history'] = 1.0 - df['filbert_prob']
    
    # Independent Variables
    # a. BERTScore (Generic Similarity)
    # b. HSR (Task-Specific Hallucination Rate)
    # c. Report Length (Control Variable)
    df['report_length'] = df['generated_report'].fillna("").str.len()
    
    # Drop NaNs
    analysis_df = df[['prob_no_history', 'BERTScore', 'HSR', 'report_length']].dropna()
    
    print(f"Data Points: {len(analysis_df)}")
    
    # 3. Linear Regression (OLS)
    # y = b0 + b1*BERTScore + b2*HSR + b3*Length + epsilon
    X = analysis_df[['BERTScore', 'HSR', 'report_length']]
    X = sm.add_constant(X) # Add Intercept (b0)
    y = analysis_df['prob_no_history']
    
    model = sm.OLS(y, X).fit()
    
    # 4. Output Results matching Table IV
    print("\n=== Table IV Replication: Regression Results ===")
    print(model.summary())
    
    # Extract key stats for simplified view
    summary = pd.DataFrame({
        'Coefficient': model.params,
        'Std Error': model.bse,
        't-statistic': model.tvalues,
        'P>t': model.pvalues
    })
    
    print("\n--- Simplified Table IV ---")
    print(summary)
    
    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "table_iv_regression.csv")
    with open(output_dir / "table_iv_full_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        
    # 6. Optional: Visualization (Scatter Plot)
    # HSR vs BERTScore (to show lack of correlation)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=analysis_df, x='BERTScore', y='HSR', alpha=0.5)
    plt.title("Decoupling: HSR vs BERTScore")
    plt.xlabel("BERTScore (Generic Quality)")
    plt.ylabel("HSR (Hallucination Rate)")
    plt.savefig(output_dir / "decoupling_scatter.png")
    print(f"\nSaved analysis to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, 
                        help="CSV file containing merged results (must have HSR, filbert_prob, BERTScore)")
    parser.add_argument("--output_dir", default="./analysis/table_iv")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    run_decoupling_analysis(df, pathlib.Path(args.output_dir))