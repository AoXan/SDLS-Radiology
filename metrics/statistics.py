# metrics/statistics.py
"""
Statistical Utility Module
--------------------------
Implements non-parametric bootstrapping for Confidence Intervals (CI) 
and paired significance tests with False Discovery Rate (FDR) control.

Paper Reference: Statistical Analysis section (95% CIs, BH correction)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple

def bootstrap_ci(data: List[float], n_bootstraps: int = 10000, ci: int = 95, seed: int = 42) -> Tuple[float, float]:
    """
    Calculates the non-parametric bootstrap confidence interval.
    Returns: (lower, upper)
    """
    data = np.array(data)
    if len(data) < 2: return (data.mean(), data.mean())
    
    rng = np.random.default_rng(seed)
    bootstrapped_means = []
    
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
        
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return lower, upper

def compare_methods_fdr(baseline_scores, target_scores, method='wilcoxon'):
    """
    Performs paired test (Wilcoxon) and returns p-value.
    Note: BH correction should be applied after collecting all p-values.
    """
    if len(baseline_scores) != len(target_scores):
        raise ValueError("Samples are not paired.")
        
    if method == 'wilcoxon':
        try:
            _, p = stats.wilcoxon(baseline_scores, target_scores)
        except ValueError:
            p = 1.0 # If all diffs are zero
    else:
        _, p = stats.ttest_rel(baseline_scores, target_scores)
        
    return p