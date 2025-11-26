import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def calculate_psi(base, current, buckets=10):
    base = pd.Series(base).dropna()
    current = pd.Series(current).dropna()
    
    # create bins on base
    quantiles = np.linspace(0, 1, buckets+1)
    breakpoints = base.quantile(quantiles).values
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    base_counts, _ = np.histogram(base, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)

    base_perc = base_counts / len(base)
    curr_perc = curr_counts / len(current)

    # avoid zero
    base_perc = np.where(base_perc == 0, 1e-6, base_perc)
    curr_perc = np.where(curr_perc == 0, 1e-6, curr_perc)

    psi = np.sum((base_perc - curr_perc) * np.log(base_perc / curr_perc))
    return psi

def calculate_kl(base, current, bins=50):
    base = pd.Series(base).dropna()
    current = pd.Series(current).dropna()

    # same bins for both
    counts_base, bin_edges = np.histogram(base, bins=bins, density=True)
    counts_curr, _ = np.histogram(current, bins=bin_edges, density=True)

    counts_base = np.where(counts_base == 0, 1e-6, counts_base)
    counts_curr = np.where(counts_curr == 0, 1e-6, counts_curr)

    kl = np.sum(counts_base * np.log(counts_base / counts_curr))
    return kl

def calculate_ks(base, current):
    base = pd.Series(base).dropna()
    current = pd.Series(current).dropna()
    stat, p_value = ks_2samp(base, current)
    return stat, p_value

def analyze_drift(df_base, df_curr, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df_base.select_dtypes(include=[np.number]).columns

    results = []
    for col in numeric_cols:
        psi = calculate_psi(df_base[col], df_curr[col])
        kl = calculate_kl(df_base[col], df_curr[col])
        ks_stat, ks_p = calculate_ks(df_base[col], df_curr[col])

        if psi < 0.1:
            severity = "Stable"
        elif psi < 0.25:
            severity = "Moderate"
        else:
            severity = "Severe"

        results.append({
            "feature": col,
            "psi": psi,
            "kl_divergence": kl,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "severity": severity
        })

    return pd.DataFrame(results)
