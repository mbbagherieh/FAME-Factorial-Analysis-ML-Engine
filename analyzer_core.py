# ============================================================
# FAME — Factorial Analysis & Machine Learning Engine
# analyzer_core.py — Shared analysis engine
# Do NOT run this file directly. Import from wrapper scripts.
# 
# License: GPL-3
# Repository: https://github.com/mbbagherieh/FAME-streamlit-ml-analysis
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, shapiro, levene, kruskal, pearsonr, studentized_range, ttest_rel
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.decomposition import PCA
import shap
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import io
import zipfile
import subprocess
import tempfile
import os

# ─────────────────────────────────────────────────────────────
# Shared helper: correlation-style standardization with ddof=1
# Used by PCA, beta coefficients, and any user-facing
# standardization to ensure eigenvalues sum to p and that
# variance estimates are consistent across the pipeline.
# ─────────────────────────────────────────────────────────────
def generate_python_cld(groups, means, p_matrix, alpha=0.05):
    """
    Generate a Compact Letter Display (CLD) using a greedy algorithm.
    Produces valid, non-overlapping letter groupings.
    """
    order = np.argsort(means)[::-1]
    sorted_groups = [groups[i] for i in order]
    
    group_letters = {g: set() for g in groups}
    current_letter_idx = 0
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    for i in range(len(sorted_groups)):
        g_i = sorted_groups[i]
        idx_i = groups.index(g_i)
        
        assigned = False
        for l_idx in range(current_letter_idx + 1):
            l_char = alphabet[l_idx]
            compatible = True
            for g_j in groups:
                if l_char in group_letters[g_j]:
                    idx_j = groups.index(g_j)
                    if p_matrix[idx_i, idx_j] < alpha:
                        compatible = False
                        break
            if compatible:
                group_letters[g_i].add(l_char)
                assigned = True
                
        if not assigned:
            current_letter_idx += 1
            group_letters[g_i].add(alphabet[current_letter_idx])
            
    return {g: "".join(sorted(list(letters_set))) for g, letters_set in group_letters.items()}
        
    return result

def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values in the same order as input p-values.
    NaN values are preserved and excluded from the correction.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return q
    p_valid = p[mask]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    # q_i = min over j>=i of (n/j * p_j)
    adjusted = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    # Restore original order
    q_valid = np.empty(n)
    q_valid[order] = adjusted
    q[mask] = q_valid
    return q

def nadeau_bengio_ttest(a, b, K):
    """
    Nadeau-Bengio corrected t-test for comparison of CV fold scores.
    
    Parameters
    ----------
    a, b : array-like
        Per-fold scores for two models. Must have length K.
    K : int
        Number of cross-validation folds.
    
    Returns
    -------
    t_stat : float
    p_value : float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d <= 0:
        return np.nan, np.nan

    # Nadeau-Bengio: inflate variance by (1/K + n_test/n_train)
    n_test = n
    n_train = max(n * (K - 1), 1)
    correction = 1.0 / n + n_test / n_train
    t_stat = mean_d / np.sqrt(correction * var_d)
    df = n - 1
    p_value = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df))
    return float(t_stat), float(p_value)

def shap_null_and_bootstrap(X_ml, Y_ml, n_responses, rf_estimators=100,
                              n_permutations=50, n_bootstrap=100,
                              n_jobs=-1, random_state=42):
    """
    Compute null distribution and bootstrap CIs for SHAP importance (Item 1f).
    """
    from joblib import Parallel, delayed
    
    n_factors = X_ml.shape[1]
    
    def _shap_importance(X, y, seed):
        try:
            rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=None,
                                        min_samples_leaf=2, random_state=seed, n_jobs=1)
            rf.fit(X, y)
            explainer = shap.TreeExplainer(rf)
            sv = np.abs(explainer.shap_values(X)).mean(axis=0)
            return 100.0 * sv / sv.sum()
        except Exception:
            return np.full(n_factors, np.nan)
    
    # Real fit
    real_shap = np.zeros((n_responses, n_factors))
    for j in range(n_responses):
        real_shap[j, :] = _shap_importance(X_ml, Y_ml[:, j], random_state)
    
    # Null distribution
    def _null_fit(j, perm_idx):
        rng = np.random.default_rng(random_state + perm_idx * 1000 + j)
        y_perm = Y_ml[rng.permutation(len(Y_ml)), j]
        return _shap_importance(X_ml, y_perm, random_state + perm_idx)
    
    null_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_null_fit)(j, p) for j in range(n_responses) for p in range(n_permutations)
    )
    null_arr = np.array(null_results).reshape(n_responses, n_permutations, n_factors)
    null_p95 = np.nanpercentile(null_arr, 95, axis=1)
    
    # Bootstrap
    def _boot_fit(j, boot_idx):
        rng = np.random.default_rng(random_state + boot_idx * 1000 + j)
        idx = rng.integers(0, len(X_ml), size=len(X_ml))
        return _shap_importance(X_ml[idx], Y_ml[idx, j], random_state + boot_idx)
    
    boot_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_boot_fit)(j, b) for j in range(n_responses) for b in range(n_bootstrap)
    )
    boot_arr = np.array(boot_results).reshape(n_responses, n_bootstrap, n_factors)
    boot_lower = np.nanpercentile(boot_arr, 2.5, axis=1)
    boot_upper = np.nanpercentile(boot_arr, 97.5, axis=1)
    
    exceeds_null = real_shap > null_p95
    
    return {
        'null_p95': null_p95,
        'real_mean': real_shap,
        'boot_ci_lower': boot_lower,
        'boot_ci_upper': boot_upper,
        'exceeds_null': exceeds_null,
    }

def run_seed_stability(X_ml, Y_ml, factors, n_responses, rf_estimators,
                        cv_folds, n_seeds=20):
    """
    Run SHAP and CV under multiple random seeds to assess stability.
    
    Parameters
    ----------
    X_ml : array         Feature matrix (n_samples, n_factors)
    Y_ml : array         Response matrix (n_samples, n_responses)
    factors : list       Factor names
    n_responses : int
    rf_estimators : int
    cv_folds : int
    n_seeds : int        Number of random seeds to test
    
    Returns
    -------
    dict with keys:
      'shap_mean'  : (n_responses, n_factors) — mean SHAP % across seeds
      'shap_sd'    : (n_responses, n_factors) — SD of SHAP % across seeds
      'cv_r2_mean' : (n_responses,) — mean CV R² across seeds
      'cv_r2_sd'   : (n_responses,) — SD of CV R² across seeds
      'seeds'      : list of seeds tested
    """
    seeds = list(range(1, n_seeds + 1))
    n_factors = len(factors)
    
    shap_all = np.zeros((n_seeds, n_responses, n_factors))
    cv_r2_all = np.zeros((n_seeds, n_responses))
    
    for i, seed in enumerate(seeds):
        for j in range(n_responses):
            y = Y_ml[:, j]
            try:
                rf = RandomForestRegressor(
                    n_estimators=rf_estimators, max_depth=None,
                    min_samples_leaf=2, random_state=seed, n_jobs=-1)
                rf.fit(X_ml, y)
                explainer = shap.TreeExplainer(rf)
                sv = np.abs(explainer.shap_values(X_ml)).mean(axis=0)
                shap_all[i, j, :] = 100.0 * sv / sv.sum()
                
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                scores = cross_val_score(rf, X_ml, y, cv=cv, scoring='r2')
                cv_r2_all[i, j] = scores.mean()
            except Exception:
                shap_all[i, j, :] = np.nan
                cv_r2_all[i, j] = np.nan
    
    return {
        'shap_mean': np.nanmean(shap_all, axis=0),
        'shap_sd': np.nanstd(shap_all, axis=0),
        'cv_r2_mean': np.nanmean(cv_r2_all, axis=0),
        'cv_r2_sd': np.nanstd(cv_r2_all, axis=0),
        'seeds': seeds,
    }

def compute_effect_sizes(df_term, df_error, ss_term, ss_error, ms_error, n_total):
    """
    Compute classical effect sizes for one ANOVA term (Item 1k).
    
    Parameters
    ----------
    df_term : int     Degrees of freedom for this term
    df_error : int    Degrees of freedom for residuals
    ss_term : float   Sum of squares for this term
    ss_error : float  Sum of squares for residuals
    ms_error : float  Mean square error
    n_total : int     Total sample size
    
    Returns
    -------
    dict with:
      'eta2'     : classical eta-squared (SS_term / SS_total)
      'partial_eta2' : SS_term / (SS_term + SS_error)
      'omega2'   : SS_term - df_term * MS_error, divided by (SS_total + MS_error)
      'epsilon2' : SS_term - df_term * MS_error, divided by SS_total
    """
    ss_total = ss_term + ss_error  # only for this comparison
    # Classical eta-squared
    eta2 = ss_term / ss_total if ss_total > 0 else np.nan
    # Partial eta-squared (as currently reported)
    partial_eta2 = ss_term / (ss_term + ss_error) if (ss_term + ss_error) > 0 else np.nan
    # Omega-squared (less biased than eta-squared)
    numerator = ss_term - df_term * ms_error
    denominator = ss_total + ms_error
    omega2 = numerator / denominator if denominator > 0 else np.nan
    # Epsilon-squared (alternative less-biased estimator)
    epsilon2 = numerator / ss_total if ss_total > 0 else np.nan
    return {
        'eta2': float(eta2),
        'partial_eta2': float(partial_eta2),
        'omega2': float(omega2),
        'epsilon2': float(epsilon2),
    }

def morf_out_of_fold_residuals(X_ml, Y_ml, cv_folds=5, rf_estimators=200,
                                random_state=42):
    """
    Compute MORF residual correlations using out-of-fold predictions (Item 1j).
    
    For each fold:
      1. Fit a multi-output Random Forest on training rows
      2. Predict on held-out rows
      3. Collect residuals (observed - predicted) for held-out rows
    
    Correlate the collected out-of-fold residuals across all responses.
    
    Returns
    -------
    resid_corr_oof : (n_responses, n_responses) ndarray
        Pearson correlation matrix of out-of-fold residuals.
    """
    n_samples, n_responses = Y_ml.shape
    resid_oof = np.zeros_like(Y_ml, dtype=float)
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in kf.split(X_ml):
        rf = RandomForestRegressor(
            n_estimators=rf_estimators, max_depth=None,
            min_samples_leaf=2, random_state=random_state, n_jobs=-1)
        rf.fit(X_ml[train_idx], Y_ml[train_idx])
        predictions = rf.predict(X_ml[test_idx])
        resid_oof[test_idx] = Y_ml[test_idx] - predictions
    
    # Correlate residuals across responses
    with np.errstate(invalid='ignore', divide='ignore'):
        resid_corr = np.corrcoef(resid_oof.T)
    
    # Handle NaN (constant columns)
    resid_corr = np.nan_to_num(resid_corr, nan=0.0)
    np.fill_diagonal(resid_corr, 1.0)
    
    return resid_corr

def simple_effects(df, response, factor_fixed, factor_contrast):
    """
    Test the effect of `factor_fixed` at each level of `factor_contrast`.
    
    For each level of the contrast factor, subset the data and run a
    one-way ANOVA of the fixed factor. Reports F and p for the fixed
    factor at each level, revealing how its effect changes across the
    levels of the other factor.
    
    Parameters
    ----------
    df : DataFrame
    response : str
    factor_fixed : str      Factor whose effect is tested
    factor_contrast : str   Factor whose levels define the subsets
    
    Returns
    -------
    dict: {contrast_level: {'F': float, 'p': float, 'df': int}}
    """
    results = {}
    if factor_fixed not in df.columns or factor_contrast not in df.columns:
        return results
    
    for level in sorted(df[factor_contrast].dropna().unique()):
        subset = df[df[factor_contrast] == level].dropna(subset=[response])
        if len(subset) < 4:
            results[str(level)] = {'F': np.nan, 'p': np.nan, 'df': 0}
            continue
        try:
            formula = f'Q("{response}") ~ C({factor_fixed})'
            model = ols(formula, data=subset).fit()
            aov = sm.stats.anova_lm(model, typ=2)  # one-way; typ irrelevant here
            key = [idx for idx in aov.index if idx != 'Residual'][0]
            results[str(level)] = {
                'F': float(aov.loc[key, 'F']),
                'p': float(aov.loc[key, 'PR(>F)']),
                'df': int(aov.loc[key, 'df']),
            }
        except Exception:
            results[str(level)] = {'F': np.nan, 'p': np.nan, 'df': 0}
    
    return results

def art_anova(df, response, factors, is_cat_col=True):
    """
    Aligned Rank Transform (ART) ANOVA.
    
    For each effect (main + interactions), align the response by removing
    the effect, rank the aligned residuals, and run parametric ANOVA on
    the ranks.
    
    Parameters
    ----------
    df : DataFrame     Must contain `response` and factor columns.
    response : str     Name of the response column.
    factors : list     List of factor column names (categorical, _cat suffix).
    is_cat_col : bool  If True, factor names refer to _cat columns.
    
    Returns
    -------
    dict: {term_name: {'F': float, 'p': float}}
    """
    from itertools import combinations
    from scipy.stats import rankdata
    
    sub = df[[response] + factors].dropna(subset=[response]).copy()
    
    # Build all terms: main + two-way interactions
    terms = list(factors)
    for f1, f2 in combinations(factors, 2):
        terms.append(f'{f1}:{f2}')
    
    results = {}
    
    for term in terms:
        try:
            # Build formula WITHOUT this term
            other_terms = [t for t in terms if t != term]
            if other_terms:
                formula_reduced = f'Q("{response}") ~ ' + ' + '.join(
                    [f'C({t})' if ':' not in t else f'C({t.split(":")[0]}):C({t.split(":")[1]})'
                     for t in other_terms])
                model_reduced = ols(formula_reduced, data=sub).fit()
                # Aligned response = original - effect_not_in_reduced_model
                # (i.e., only this term remains in the alignment)
                aligned = model_reduced.resid.values
            else:
                aligned = sub[response].values
            
            # Rank the aligned values
            ranks = rankdata(aligned)
            sub['_art_rank'] = ranks
            
            # Fit the term on the ranks
            if ':' in term:
                f1, f2 = term.split(':')
                formula_full = f'Q("_art_rank") ~ C({f1}) + C({f2}) + C({f1}):C({f2})'
            else:
                other_mains = [f for f in factors if f != term]
                if other_mains:
                    formula_full = f'Q("_art_rank") ~ ' + ' + '.join(
                        [f'C({f})' for f in [term] + other_mains])
                else:
                    formula_full = f'Q("_art_rank") ~ C({term})'
            
            model_full = ols(formula_full, data=sub).fit()
            aov = sm.stats.anova_lm(model_full, typ=2)
            
            # Extract the F-statistic and p-value for this term
            target_key = None
            for idx in aov.index:
                if idx == 'Residual':
                    continue
                if term in idx or all(f in idx for f in term.split(':')):
                    target_key = idx
                    break
            
            if target_key is not None:
                results[term] = {
                    'F': float(aov.loc[target_key, 'F']),
                    'p': float(aov.loc[target_key, 'PR(>F)']),
                }
            else:
                results[term] = {'F': np.nan, 'p': np.nan}
        except Exception:
            results[term] = {'F': np.nan, 'p': np.nan}
        
        # Clean up
        if '_art_rank' in sub.columns:
            sub.drop(columns=['_art_rank'], inplace=True)
    
    return results

def standardize_ddof1(A):
    """Column-wise Z-score with sample SD (ddof=1)."""
    A = np.asarray(A, dtype=float)
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)   # protect constant columns
    return (A - mu) / sd
    
# Optional: R interface for correct post-hoc tests (via subprocess)
R_AVAILABLE = False
_R_ERROR_MSG = ""
try:
    result = subprocess.run(['R', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        check = subprocess.run(
            ['R', '-e', 'library(agricolae)'],
            capture_output=True, text=True, timeout=10
        )
        if check.returncode == 0:
            R_AVAILABLE = True
        else:
            _R_ERROR_MSG = "agricolae package not found in R. Install with: install.packages('agricolae')"
    else:
        _R_ERROR_MSG = "R not found on system PATH"
except Exception as e:
    _R_ERROR_MSG = str(e)[:200]

# R_AVAILABLE = False  # TEMPORARY — force Python Piepho for testing # commented in normal usage

def generate_text_report(df, FACTOR_COLS, RESPONSE_COLS, n_factors, n_responses,
                         df_prelim, df_pearson, outlier_results,
                         df_int, full_anova_tables, df_kw,
                         posthoc_dict, posthoc_method,
                         shap_imp, perm_imp, model_comparison, df_mc,
                         beta, beta_ci_lower, beta_ci_upper,
                         pca, pca_var_explained, df_pca_summary, df_pca_loadings,
                         resid_corr, pca_loadings, cv_folds,
                         factor_type_log=None,
                         source_file_name=None, n_rows_raw=None,
                         n_rows_anova=None, n_rows_ml=None, n_rows_dropped=None,
                         df_anova_fdr=None, df_art=None, df_simple=None,
                         stability_results=None, shap_null_results=None,
                         resid_corr_oof=None):
    """Generate a comprehensive text report of all analyses."""
    
    report = []
    sep = "=" * 80
    
    report.append(sep)
    report.append("FAME — FACTORIAL ANALYSIS & MACHINE LEARNING ENGINE")
    report.append("Comprehensive Statistical and Machine Learning Analysis Report")
    report.append(sep)
    report.append(f"Date of analysis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Factors: {', '.join(FACTOR_COLS)}")
    report.append(f"Responses: {', '.join(RESPONSE_COLS)}")
    report.append(f"Observations: {len(df)}")
    report.append(f"Source file: {source_file_name}")
    report.append(f"Total rows in source: {n_rows_raw}")
    report.append(f"Rows used for ANOVA/Pearson: {n_rows_anova}")
    report.append(f"Rows used for ML (after NA drop): {n_rows_ml}")
    if n_rows_dropped > 0:
        report.append(f"Rows dropped for ML due to missing values: {n_rows_dropped}")
    if factor_type_log:
        report.append("Factor type handling:")
        for f, status in factor_type_log.items():
            report.append(f"  {f}: {status}")
        report.append(sep)
   
    
    # ── 1. DESCRIPTIVE STATISTICS ──
    report.append("\n" + "=" * 80)
    report.append("1. DESCRIPTIVE STATISTICS")
    report.append("=" * 80)
    
    for factor in FACTOR_COLS:
        report.append(f"\n--- Factor: {factor} ---")
        report.append(f"{'Statistic':<15} {'Value':>10}")
        report.append("-" * 25)
        try:
            for stat_name in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                val = df[factor].describe()[stat_name]
                report.append(f"{stat_name:<15} {val:>10.4f}")
        except:
            report.append(f"  (categorical: {df[factor].unique().tolist()})")
    
    for resp in RESPONSE_COLS:
        report.append(f"\n--- Response: {resp} ---")
        report.append(f"{'Statistic':<15} {'Value':>10}")
        report.append("-" * 25)
        try:
            for stat_name in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                val = pd.to_numeric(df[resp], errors='coerce').describe()[stat_name]
                report.append(f"{stat_name:<15} {val:>10.4f}")
        except:
            pass
    
    # Treatment combination means
    if n_factors >= 2:
        report.append(f"\n--- Treatment Combination Means ---")
        df_temp = df.copy()
        if 'Treatment' in df_temp.columns:
            for resp in RESPONSE_COLS:
                report.append(f"\n{resp}:")
                try:
                    means = df_temp.groupby('Treatment')[resp].mean().sort_values(ascending=False)
                    for trt, val in means.items():
                        report.append(f"  {trt:<30} {val:>10.4f}")
                except:
                    pass
    
    # ── 2. ASSUMPTION CHECKS ──
    report.append("\n\n" + "=" * 80)
    report.append("2. STATISTICAL ASSUMPTION CHECKS")
    report.append("=" * 80)
    report.append("\nTable 1: Normality (Shapiro-Wilk) and Homogeneity of Variances (Levene)")
    report.append(df_prelim.to_string(index=False))
    
    report.append("\n\nOutlier Detection Summary:")
    for resp in RESPONSE_COLS:
        if resp in outlier_results:
            n_out = outlier_results[resp]['n_outliers']
            bounds = outlier_results[resp]['IQR_bounds']
            report.append(f"  {resp}: {n_out} outlier(s), IQR bounds: {bounds}")
            if n_out > 0:
                for detail in outlier_results[resp]['details'][:10]:
                    report.append(f"    Row {detail['Row']}: value={detail['Value']}, {detail['Detected by']}")
    
    # ── 3. PEARSON CORRELATIONS ──
    report.append("\n\n" + "=" * 80)
    report.append("3. PEARSON CORRELATION COEFFICIENTS")
    report.append("=" * 80)
    report.append("\nTable 2: Factor-Response Pearson Correlations")
    report.append(df_pearson.to_string(index=False))
    
    report.append("\n\nInterpretation:")
    sig_corr = df_pearson[df_pearson['Sig.'] != 'ns']
    if len(sig_corr) > 0:
       for _, row in sig_corr.iterrows():
            try:
                r_val = float(row['Pearson r'])
                direction = "positive" if r_val > 0 else "negative"
                strength = "strong" if abs(r_val) > 0.7 else "moderate" if abs(r_val) > 0.4 else "weak"
                report.append(f"  • {row['Factor']} vs {row['Response']}: {strength} {direction} correlation "
                             f"(r={r_val:.3f}, p={float(row['p-value']):.4f}, {row['Sig.']})")
            except (ValueError, TypeError):
                report.append(f"  • {row['Factor']} vs {row['Response']}: insufficient data ({row['Sig.']})")
    else:
        report.append("  No significant correlations found.")
    
    # ── 4. ANOVA RESULTS ──
    report.append("\n\n" + "=" * 80)
    report.append("4. N-WAY ANOVA RESULTS")
    report.append("=" * 80)
    
    for resp in RESPONSE_COLS:
        if resp in full_anova_tables:
            report.append(f"\n--- {resp} ---")
            cols_show = [c for c in ['sum_sq','df','F','PR(>F)','η²','ω²','ε²','q_within']
                         if c in full_anova_tables[resp].columns]
            report.append(full_anova_tables[resp][cols_show].round(6).to_string())
            
            atbl = full_anova_tables[resp]
            sig_effects = atbl[(atbl['PR(>F)'] < 0.05) & (atbl.index != 'Intercept')]            
            if len(sig_effects) > 0:
                report.append(f"\n  Significant effects (p < 0.05):")
                for idx in sig_effects.index:
                    if idx != 'Residual':
                        eta = atbl.loc[idx, 'η²']
                        effect_size = "large" if eta >= 0.14 else "medium" if eta >= 0.06 else "small"
                        report.append(f"    • {idx}: F={atbl.loc[idx,'F']:.2f}, p={atbl.loc[idx,'PR(>F)']:.4f}, "
                                     f"η²={eta:.4f} ({effect_size} effect)")

    # ---- FDR summary ----
    if df_anova_fdr is not None and len(df_anova_fdr) > 0:
        report.append("\n\n" + "=" * 80)
        report.append("4b. ANOVA WITH FDR CORRECTION (Benjamini-Hochberg)")
        report.append("=" * 80)
        report.append("\nEffect sizes reported: η² (classical eta-squared), ω² (omega-squared,")
        report.append("less biased at small n), ε² (epsilon-squared). Partial η² is not")
        report.append("directly comparable across designs with different numbers of factors")
        report.append("or levels; ω² and ε² are preferred for meta-analytic comparison.")
        report.append("\nq_within = BH correction across terms within a response")
        report.append("q_across = BH correction across responses for the same term\n")
        cols_fdr = [c for c in ['Response','Term','F','p_raw','q_within','q_across','η²','ω²','ε²']
                    if c in df_anova_fdr.columns]
        report.append(df_anova_fdr[cols_fdr].round(6).to_string(index=False))
        
        surviving = df_anova_fdr[(df_anova_fdr['q_within'] < 0.05) | 
                                 (df_anova_fdr['q_across'] < 0.05)]
        report.append(f"\nEffects surviving FDR (q < 0.05, either correction): "
                      f"{len(surviving)} of {len(df_anova_fdr)}")
        # ---- Multiplicity summary (Item 2c) ----
        report.append("\n" + "-" * 80)
        report.append("MULTIPLICITY SAFEGUARD SUMMARY")
        report.append("-" * 80)
        n_total = len(df_anova_fdr)
        n_sig_raw = int((df_anova_fdr['p_raw'] < 0.05).sum())
        n_sig_qwithin = int((df_anova_fdr['q_within'] < 0.05).sum())
        n_sig_qacross = int((df_anova_fdr['q_across'] < 0.05).sum())
        n_sig_either = int(((df_anova_fdr['q_within'] < 0.05) | 
                            (df_anova_fdr['q_across'] < 0.05)).sum())
        
        report.append(f"\nThis report contains {n_total} ANOVA p-values across")
        report.append(f"{len(RESPONSE_COLS)} response(s) and {len(full_anova_tables[RESPONSE_COLS[0]])} term(s).")
        report.append(f"\n   Raw p < 0.05:              {n_sig_raw:>3} of {n_total}")
        report.append(f"   q_within < 0.05 (BH):      {n_sig_qwithin:>3} of {n_total}")
        report.append(f"   q_across < 0.05 (BH):      {n_sig_qacross:>3} of {n_total}")
        report.append(f"   q < 0.05 under either:     {n_sig_either:>3} of {n_total}")
        
        # Effects lost under correction
        lost = df_anova_fdr[(df_anova_fdr['p_raw'] < 0.05) & 
                            (df_anova_fdr['q_within'] >= 0.05) & 
                            (df_anova_fdr['q_across'] >= 0.05)]
        if len(lost) > 0:
            report.append(f"\nEffects significant at raw p < 0.05 but NOT surviving")
            report.append(f"either FDR correction (interpret with caution):")
            for _, row in lost.iterrows():
                report.append(f"   • {row['Response']:<15} {row['Term']:<40} "
                              f"p={row['p_raw']:.4f}  q_within={row['q_within']:.4f}  q_across={row['q_across']:.4f}")
        else:
            report.append("\nAll effects significant at raw p < 0.05 also survive FDR correction.")
        
        report.append("\nNote: q_across is stricter when the same term is tested across many")
        report.append("responses. Use q_within for within-response claims, q_across for")
        report.append("cross-response claims (e.g., 'factor X affects all responses').")
        
    if len(df_int) > 0:
        sig_int_df = df_int[df_int['Sig.'] != 'ns']
        if len(sig_int_df) > 0:
            report.append(f"\nSignificant Two-Way Interactions (p < 0.05):")
            report.append(sig_int_df[['Response','Factor_1','Factor_2','F_int','p_int','Sig.','Type']].to_string(index=False))
    if df_simple is not None and len(df_simple) > 0:
        report.append("\n\nSIMPLE EFFECTS ANALYSIS")
        report.append("=" * 80)
        report.append("For each significant interaction, the effect of each factor")
        report.append("is tested separately at each level of the other factor.\n")
        
        for _, row in df_simple.iterrows():
            report.append(f"\n--- {row['Response']}  |  {row['Interaction']} ---")
            
            for col_name in row.index:
                if col_name in ('Response', 'Interaction'):
                    continue
                report.append(f"\n  {col_name}:")
                se_dict = row[col_name]
                if isinstance(se_dict, dict):
                    for level, stats_ in se_dict.items():
                        if not np.isnan(stats_.get('F', np.nan)):
                            p = stats_.get('p', np.nan)
                            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                            report.append(f"    at {level}: F={stats_['F']:.3f}, p={p:.4f} {sig}")
                        else:
                            report.append(f"    at {level}: insufficient data")    

    if len(df_kw) > 0:
        report.append(f"\n\nKruskal-Wallis Non-Parametric Validation:")
        report.append(df_kw.to_string(index=False))
    if df_art is not None and len(df_art) > 0:
        report.append("\n\nART ANOVA (aligned rank transform) — Item 1g")
        report.append(df_art.round(4).to_string(index=False))
        report.append("\n  Note: ART ANOVA is the design-appropriate non-parametric")
        report.append("  validation for factorial designs; it accounts for all factor")
        report.append("  effects and interactions, unlike Kruskal-Wallis applied")
        report.append("  factor-by-factor.")
    
    # ── 5. POST-HOC MEAN COMPARISONS ──
    report.append("\n\n" + "=" * 80)
    report.append("5. POST-HOC MEAN COMPARISONS")
    report.append("=" * 80)
    report.append(f"Method: {posthoc_method}  |  "
                  f"Engine: {'R agricolae (HSD.test/LSD.test)' if R_AVAILABLE else 'Python fallback (significance matrix)'}")
    if not R_AVAILABLE:
        report.append("Note: The native Python fallback generates Piepho CLD groupings. "
                      "These may differ slightly from the letters assigned by R's 'agricolae' package, "
                      "which is the reference implementation and gold standard for this algorithm.")
    
    for resp in RESPONSE_COLS:
        if resp in posthoc_dict and posthoc_dict[resp]['groups']:
            ph = posthoc_dict[resp]
            report.append(f"\n--- {resp} ({ph['method']}, CD = {ph['critical_diff']:.4f}) ---")
            report.append(f"{'Treatment':<35} {'Mean':>10} {'SD':>10} {'Group':>8}")
            report.append("-" * 65)
            for trt, info in ph['groups'].items():
                report.append(f"{trt:<35} {info['mean']:>10.4f} {info['sd']:>10.4f} {info['letter']:>8}")
            
            if 'sig_matrix' in ph and len(ph['sig_matrix']) > 0:
                report.append(f"\n  Pairwise Significance Matrix:")
                report.append(ph['sig_matrix'].to_string(index=False))
    
    # ── 6. SHAP FEATURE IMPORTANCE ──
    report.append("\n\n" + "=" * 80)
    report.append("6. SHAP FEATURE IMPORTANCE ANALYSIS")
    report.append("=" * 80)
    
    shap_pct = shap_imp / shap_imp.sum(axis=1, keepdims=True) * 100
    report.append("\nTable: SHAP Relative Importance (%)")
    report.append(f"{'Response':<20} " + " ".join([f"{f:>12}" for f in FACTOR_COLS]))
    report.append("-" * (20 + 13 * n_factors))
    for i, resp in enumerate(RESPONSE_COLS):
        report.append(f"{resp:<20} " + " ".join([f"{shap_pct[i,j]:>12.1f}" for j in range(n_factors)]))
    
    report.append("\nInterpretation:")
    for i, resp in enumerate(RESPONSE_COLS):
        dominant_idx = np.argmax(shap_pct[i])
        report.append(f"  • {resp}: Dominant factor is {FACTOR_COLS[dominant_idx]} ({shap_pct[i, dominant_idx]:.1f}%)")
    
    # ── 7. PERMUTATION IMPORTANCE ──
    if n_factors >= 2:
        report.append("\n\n" + "=" * 80)
        report.append("7. PERMUTATION FEATURE IMPORTANCE (Validation)")
        report.append("=" * 80)
        perm_pct = perm_imp / perm_imp.sum(axis=1, keepdims=True) * 100
        report.append("\nTable: Permutation Relative Importance (%)")
        report.append(f"{'Response':<20} " + " ".join([f"{f:>12}" for f in FACTOR_COLS]))
        report.append("-" * (20 + 13 * n_factors))
        for i, resp in enumerate(RESPONSE_COLS):
            report.append(f"{resp:<20} " + " ".join([f"{perm_pct[i,j]:>12.1f}" for j in range(n_factors)]))
    
    # ── 8. MODEL COMPARISON ──
    report.append("\n\n" + "=" * 80)
    report.append(f"8. MODEL PERFORMANCE COMPARISON ({cv_folds}-Fold CV R²)")
    report.append("=" * 80)
    report.append("\nTable: Cross-validated R² (mean ± std) — standard KFold")
    perf_cols = ['Response', 'RF_R2_mean', 'RF_R2_std', 'LR_R2_mean', 'LR_R2_std', 
                'GPR_R2_mean', 'GPR_R2_std']
    report.append(df_mc[perf_cols].round(3).to_string(index=False))
    
    # ---- GroupKFold comparison (Item 1d) ----
    if 'RF_R2_group' in df_mc.columns and df_mc['RF_R2_group'].notna().any():
        report.append("\n\n" + "-" * 80)
        report.append("Table: Cross-validated R² — GroupKFold by treatment combination")
        report.append("(Prevents leakage when replicate samples share a treatment cell)")
        report.append("-" * 80)
        group_cols = ['Response', 'RF_R2_group', 'LR_R2_group', 'GPR_R2_group']
        report.append(df_mc[group_cols].round(3).to_string(index=False))
        
        # Change in performance between the two splitters
        report.append("\n\nΔR² (GroupKFold − KFold):")
        for _, row in df_mc.iterrows():
            if not np.isnan(row.get('RF_R2_group', np.nan)):
                d_rf  = row['RF_R2_group']  - row['RF_R2_mean']
                d_lr  = row['LR_R2_group']  - row['LR_R2_mean']
                d_gpr = row['GPR_R2_group'] - row['GPR_R2_mean']
                report.append(f"  {row['Response']:<15}  "
                              f"RF: {d_rf:+.3f}   "
                              f"LR: {d_lr:+.3f}   "
                              f"GPR: {d_gpr:+.3f}")
        
        # Summary interpretation
        mean_d_rf = (df_mc['RF_R2_group'] - df_mc['RF_R2_mean']).mean()
        report.append(f"\nMean ΔR² across responses — RF: {mean_d_rf:+.3f}")
        if mean_d_rf < -0.05:
            report.append("→ Performance drop under GroupKFold indicates that the "
                          "standard KFold estimates were optimistically biased by "
                          "shared replicate cells between train and test folds.")
        elif mean_d_rf > 0.05:
            report.append("→ Performance increase under GroupKFold suggests the "
                          "standard splitter was not the limiting factor; "
                          "results are conservative.")
        else:
            report.append("→ Performance is stable across splitters, indicating "
                          "standard KFold results were not materially inflated.")
    
    report.append("\n\nTable: Pairwise Model Comparisons — standard KFold (Paired t-test on CV folds)")
    if 'RF_vs_LR' in df_mc.columns:
        test_cols = ['Response', 'RF_vs_LR', 'RF_vs_GPR', 'LR_vs_GPR']
        report.append(df_mc[test_cols].to_string(index=False))
        report.append("\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
    else:
        report.append("  Not available.")

    if 'RF_vs_LR_NB' in df_mc.columns:
        report.append("\n\nTable: Pairwise Model Comparisons — standard KFold, "
                      "Nadeau-Bengio corrected (Item 1e)")
        nb_cols_std = ['Response', 'RF_vs_LR_NB', 'RF_vs_GPR_NB', 'LR_vs_GPR_NB']
        report.append(df_mc[nb_cols_std].to_string(index=False))
        report.append("\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
        
    if 'RF_vs_LR_group' in df_mc.columns and df_mc['RF_vs_LR_group'].notna().any():
        report.append("\n\nTable: Pairwise Model Comparisons — GroupKFold (Paired t-test on CV folds)")
        test_cols_g = ['Response', 'RF_vs_LR_group', 'RF_vs_GPR_group', 'LR_vs_GPR_group']
        report.append(df_mc[test_cols_g].to_string(index=False))
        report.append("\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")

    if 'RF_vs_LR_group_NB' in df_mc.columns and df_mc['RF_vs_LR_group_NB'].notna().any():
        report.append("\n\nTable: Pairwise Model Comparisons — GroupKFold, "
                      "Nadeau-Bengio corrected (Item 1e)")
        nb_cols_g = ['Response', 'RF_vs_LR_group_NB', 'RF_vs_GPR_group_NB', 'LR_vs_GPR_group_NB']
        report.append(df_mc[nb_cols_g].to_string(index=False))
        report.append("\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
    
    report.append("\nInterpretation:")
    for _, row in df_mc.iterrows():
        report.append(f"  • {row['Response']}")
        
        # KFold interpretation
        best_k = 'RF' if row['RF_R2_mean']>=max(row['LR_R2_mean'],row['GPR_R2_mean']) else \
                 'LR' if row['LR_R2_mean']>=max(row['RF_R2_mean'],row['GPR_R2_mean']) else 'GPR'
        report.append(f"      KFold:      Best = {best_k}  "
                      f"(RF={row['RF_R2_mean']:.3f}, LR={row['LR_R2_mean']:.3f}, GPR={row['GPR_R2_mean']:.3f})")
        if 'RF_vs_LR' in row:
            report.append(f"      KFold (naive):  {row['RF_vs_LR']}  |  {row['RF_vs_GPR']}  |  {row['LR_vs_GPR']}")
        if 'RF_vs_LR_NB' in row:
            report.append(f"      KFold (NB):     {row['RF_vs_LR_NB']}  |  {row['RF_vs_GPR_NB']}  |  {row['LR_vs_GPR_NB']}")
            
        # GroupKFold interpretation
        if 'RF_R2_group' in row and not pd.isna(row['RF_R2_group']):
            best_g = 'RF' if row['RF_R2_group']>=max(row['LR_R2_group'],row['GPR_R2_group']) else \
                     'LR' if row['LR_R2_group']>=max(row['RF_R2_group'],row['GPR_R2_group']) else 'GPR'
            report.append(f"      GroupKFold: Best = {best_g}  "
                          f"(RF={row['RF_R2_group']:.3f}, LR={row['LR_R2_group']:.3f}, GPR={row['GPR_R2_group']:.3f})")
            if 'RF_vs_LR_group' in row:
                report.append(f"      GroupKFold (naive): {row['RF_vs_LR_group']}  |  {row['RF_vs_GPR_group']}  |  {row['LR_vs_GPR_group']}")
            if 'RF_vs_LR_group_NB' in row and pd.notna(row['RF_vs_LR_group_NB']):
                report.append(f"      GroupKFold (NB):    {row['RF_vs_LR_group_NB']}  |  {row['RF_vs_GPR_group_NB']}  |  {row['LR_vs_GPR_group_NB']}")

    # ── 8c. MULTI-SEED STABILITY ── 
    if stability_results is not None:
        n_seeds_used = len(stability_results.get('seeds', [])) or 20
        report.append("\n\n" + "=" * 80)
        report.append(f"8c. MULTI-SEED STABILITY ({n_seeds_used} seeds)")
        report.append("=" * 80)
        report.append(f"\nAssesses variability of SHAP importances and CV R² across {n_seeds_used} random seeds.")
        report.append("A small SD indicates results are not dependent on the specific seed.\n")
        
        # SHAP stability
        report.append("SHAP importance (%) — mean ± SD across seeds:")
        header = "Response".ljust(20) + "  " + "  ".join([f"{f:>18}" for f in FACTOR_COLS])
        report.append(header)
        report.append("-" * len(header))
        for j, resp in enumerate(RESPONSE_COLS):
            mean_row = stability_results['shap_mean'][j, :]
            sd_row = stability_results['shap_sd'][j, :]
            cells = [f"{mean_row[i]:.1f} ± {sd_row[i]:.1f}" for i in range(len(FACTOR_COLS))]
            report.append(resp.ljust(20) + "  " + "  ".join([f"{c:>18}" for c in cells]))
        
        # CV R² stability
        report.append("\n\nCV R² (RF) — mean ± SD across seeds:")
        report.append(f"{'Response':<20} {'Mean R²':>10} {'SD R²':>10}")
        report.append("-" * 42)
        for j, resp in enumerate(RESPONSE_COLS):
            report.append(f"{resp:<20} {stability_results['cv_r2_mean'][j]:>10.3f} "
                          f"{stability_results['cv_r2_sd'][j]:>10.3f}")
    
    # ── 8d. SHAP NULL AND BOOTSTRAP (Item 1f) ──
    if shap_null_results is not None:
        report.append("\n\n" + "=" * 80)
        report.append("8d. SHAP IMPORTANCE: NULL DISTRIBUTION AND BOOTSTRAP CI")
        report.append("=" * 80)
        report.append("\nNull distribution: response permuted 50 times; 95th percentile of")
        report.append("null SHAP is the chance threshold. Real SHAP values below this")
        report.append("threshold are not distinguishable from noise.")
        report.append("Bootstrap: data resampled 100 times; 2.5th and 97.5th percentiles")
        report.append("form the 95% CI.")
        report.append("")
        
        for j, resp in enumerate(RESPONSE_COLS):
            report.append(f"\n--- {resp} ---")
            report.append(f"{'Factor':<20} {'Real %':>8} {'Null p95':>10} "
                          f"{'95% CI':>20} {'vs Null':>8}")
            report.append("-" * 70)
            for i, factor in enumerate(FACTOR_COLS):
                real = shap_null_results['real_mean'][j, i]
                null_p95 = shap_null_results['null_p95'][j, i]
                ci_lo = shap_null_results['boot_ci_lower'][j, i]
                ci_hi = shap_null_results['boot_ci_upper'][j, i]
                exceeds = shap_null_results['exceeds_null'][j, i]
                marker = '✓' if exceeds else '✗'
                report.append(f"{factor:<20} {real:>8.2f} {null_p95:>10.2f} "
                              f"[{ci_lo:>6.2f}, {ci_hi:>6.2f}] {'':>6} {marker:>5}")
        
        # Summary
        n_exceeding = int(np.sum(shap_null_results['exceeds_null']))
        n_total = shap_null_results['exceeds_null'].size
        report.append(f"\nFactor–response pairs exceeding null threshold: "
                      f"{n_exceeding} of {n_total}")
    
    # ── 9. STANDARDIZED BETA COEFFICIENTS ──
    report.append("\n\n" + "=" * 80)
    report.append("9. STANDARDIZED REGRESSION COEFFICIENTS (β) with 95% Bootstrap CI")
    report.append("=" * 80)
    
    for i, resp in enumerate(RESPONSE_COLS):
        report.append(f"\n--- {resp} ---")
        report.append(f"{'Factor':<20} {'β':>10} {'95% CI Lower':>14} {'95% CI Upper':>14} {'95% CI ≠ 0':>12}")
        report.append("-" * 70)
        for j, factor in enumerate(FACTOR_COLS):
            sig = "Yes" if (beta_ci_lower[j,i] > 0 or beta_ci_upper[j,i] < 0) else "No"
            report.append(f"{factor:<20} {beta[j,i]:>10.4f} {beta_ci_lower[j,i]:>14.4f} {beta_ci_upper[j,i]:>14.4f} {sig:>12}")
    
    # ── 10. PCA ──
    report.append("\n\n" + "=" * 80)
    report.append("10. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    report.append("=" * 80)
    report.append("\nTable: Variance Explained")
    report.append(df_pca_summary.to_string(index=False))
    
    report.append("\n\nTable: PC Loadings")
    report.append(df_pca_loadings.to_string(index=False))
    
    report.append("\nInterpretation:")
    if len(df_pca_summary) >= 2:
        cum_var = float(df_pca_summary.iloc[1]['Cumulative (%)'])
        report.append(f"  • First 2 PCs explain {cum_var:.1f}% of total variance")
    for i, resp in enumerate(RESPONSE_COLS):
        pc1_load = pca_loadings[i, 0]
        pc2_load = pca_loadings[i, 1] if n_responses >= 2 else 0
        report.append(f"  • {resp}: PC1 loading = {pc1_load:.3f}, PC2 loading = {pc2_load:.3f}")
    
    # ── 11. MORF RESIDUAL CORRELATIONS ──
    report.append("\n\n" + "=" * 80)
    report.append("11. MORF RESIDUAL CORRELATION MATRIX")
    report.append("=" * 80)
    report.append("\nResidual correlations indicate relationships between responses after accounting for factor effects.")
    report.append(f"\n{'':<15} " + " ".join([f"{r:>10}" for r in RESPONSE_COLS]))
    report.append("-" * (15 + 11 * n_responses))
    for i, resp_i in enumerate(RESPONSE_COLS):
        row_str = f"{resp_i:<15} " + " ".join([f"{resid_corr[i,j]:>10.3f}" for j in range(n_responses)])
        report.append(row_str)
    
    strong_resid = []
    for i in range(n_responses):
        for j in range(i+1, n_responses):
            if abs(resid_corr[i,j]) > 0.5:
                strong_resid.append((RESPONSE_COLS[i], RESPONSE_COLS[j], resid_corr[i,j]))
    
    if strong_resid:
        report.append("\nStrong residual correlations (|r| > 0.5):")
        for r1, r2, val in strong_resid:
            report.append(f"  • {r1} ↔ {r2}: r = {val:.3f}")

    # ── 11b. OUT-OF-FOLD MORF RESIDUALS (Item 1j) ──
    if resid_corr_oof is not None and n_responses >= 2:
        report.append("\n\n" + "=" * 80)
        report.append("11b. OUT-OF-FOLD MORF RESIDUAL CORRELATIONS (Item 1j)")
        report.append("=" * 80)
        report.append("\nResiduals computed from held-out fold predictions, not")
        report.append("in-sample predictions. This removes RF memorization bias,")
        report.append("giving an honest estimate of trait coupling after treatment.")
        
        # Format table
        header = "".ljust(15) + " ".join([f"{r[:10]:>10}" for r in RESPONSE_COLS])
        report.append("\n" + header)
        report.append("-" * len(header))
        for i, resp_i in enumerate(RESPONSE_COLS):
            row_str = f"{resp_i[:14]:<15} " + " ".join(
                [f"{resid_corr_oof[i, j]:>10.3f}" for j in range(n_responses)])
            report.append(row_str)
        
        # Strong residual correlations
        strong_oof = []
        for i in range(n_responses):
            for j in range(i+1, n_responses):
                if abs(resid_corr_oof[i, j]) > 0.5:
                    strong_oof.append((RESPONSE_COLS[i], RESPONSE_COLS[j],
                                        resid_corr_oof[i, j]))
        
        if strong_oof:
            report.append("\nStrong out-of-fold correlations (|r| > 0.5):")
            for r1, r2, val in strong_oof:
                report.append(f"  • {r1} ↔ {r2}: r = {val:.3f}")
        else:
            report.append("\nNo out-of-fold correlations exceed |r| = 0.5.")
        
        # Comparison with in-sample
        report.append("\n\nComparison with in-sample correlations:")
        report.append(f"{'Pair':<40} {'In-sample':>10} {'Out-of-fold':>12} {'Δ':>8}")
        report.append("-" * 72)
        comparisons = []
        for i in range(n_responses):
            for j in range(i+1, n_responses):
                r_in = resid_corr[i, j]
                r_oof = resid_corr_oof[i, j]
                if abs(r_in) > 0.4 or abs(r_oof) > 0.4:
                    comparisons.append((RESPONSE_COLS[i], RESPONSE_COLS[j],
                                         r_in, r_oof))
        comparisons.sort(key=lambda x: -abs(x[2]))
        for r1, r2, r_in, r_oof in comparisons[:10]:
            pair = f"{r1[:18]} ↔ {r2[:18]}"
            report.append(f"{pair:<40} {r_in:>10.3f} {r_oof:>12.3f} {r_oof - r_in:>+8.3f}")
    
    # ── SUMMARY ──
    report.append("\n\n" + "=" * 80)
    report.append("SUMMARY OF KEY FINDINGS")
    report.append("=" * 80)
    
    report.append("\nBest predictive models:")
    for _, row in df_mc.iterrows():
        report.append(f"  • {row['Response']}: {row['Best']} (R² = {max(row['RF_R2_mean'], row['LR_R2_mean'], row['GPR_R2_mean']):.3f})")
    
    report.append("\nMost influential factors (SHAP):")
    for i, resp in enumerate(RESPONSE_COLS):
        dominant_idx = np.argmax(shap_pct[i])
        report.append(f"  • {resp}: {FACTOR_COLS[dominant_idx]} ({shap_pct[i, dominant_idx]:.1f}%)")
    
    report.append("\nSignificant ANOVA effects:")
    for resp in RESPONSE_COLS:
        if resp in full_anova_tables:
            atbl = full_anova_tables[resp]
            sig_eff = [idx for idx in atbl.index if idx != 'Residual' and atbl.loc[idx, 'PR(>F)'] < 0.05]
            if sig_eff:
                report.append(f"  • {resp}: {', '.join(sig_eff)}")
    
    if len(df_int) > 0:
        sig_int_df = df_int[df_int['Sig.'] != 'ns']
        if len(sig_int_df) > 0:
            report.append("\nSignificant interactions:")
            for _, row in sig_int_df.iterrows():
                report.append(f"  • {row['Response']}: {row['Factor_1']} × {row['Factor_2']} "
                             f"({row['Type']}, p={row['p_int']:.4f})")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def run_analyzer(n_factors_default, default_factor_names, default_responses, app_title):
    """
    Main analyzer function.
    
    Parameters:
    - n_factors_default: number of factor inputs to show (1, 2, 3, 4, or 5)
    - default_factor_names: list of default factor names
    - default_responses: string with default response names (one per line)
    - app_title: title for the app
    """
    
    st. set_page_config(page_title=f"FAME — {app_title}", page_icon="🌿", layout="wide")
    st.title(f"🌿 {app_title}")
    st.markdown("Upload your experimental data for comprehensive statistical and machine learning analysis.")

    # ============================================================
    # SIDEBAR
    # ============================================================
    st.sidebar.header("⚙️ Settings")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your experimental data as a comma-separated values (.csv) file. "
             "The first row should contain column headers matching your factor and response names."
    )
    
    st.sidebar.subheader(f"Factor Columns ({n_factors_default} factors)")
    st.sidebar.caption("Enter the exact column names of your independent variables (treatments).")
    FACTOR_COLS = []
    for i in range(n_factors_default):
        factor_name = st.sidebar.text_input(
            f"Factor {i+1} name",
            value=default_factor_names[i] if i < len(default_factor_names) else f"Factor{i+1}",
            key=f"factor_{i}",
            help=f"Enter the exact column name for Factor {i+1} as it appears in your CSV file. "
                 f"This is an independent variable (treatment/predictor) in your experiment."
        )
        if factor_name.strip():
            FACTOR_COLS.append(factor_name.strip())
    
    st.sidebar.subheader("Response Columns")
    response_input = st.sidebar.text_area(
        "Response column names (one per line)",
        value=default_responses,
        help="Enter the column names of your response (dependent) variables, one per line. "
             "These are the outcomes or measurements you wish to analyze. "
             "Each line corresponds to one response column in your CSV file."
    )
    
    st.sidebar.subheader("Post-Hoc Test")
    posthoc_method = st.sidebar.selectbox(
        "Mean comparison method",
        options=["Tukey's HSD", "Fisher's LSD"],
        index=0,
        help="Post-hoc test for comparing treatment means after ANOVA. "
             "Tukey's HSD is more conservative (controls family-wise error rate, recommended for ≥3 groups). "
             "Fisher's LSD is more liberal (higher power but greater risk of false positives)."
    )
    if posthoc_method == "Fisher's LSD":
        st.sidebar.warning(
            "⚠️ Fisher's LSD does not control family-wise error rate. "
            "False positive risk increases with the number of groups. "
            "Use Tukey's HSD for ≥3 groups unless you have a specific reason to prefer LSD."
        )
    
    st.sidebar.subheader("Advanced Settings")
    n_categories = st.sidebar.slider(
        "Continuous-factor binning",
        min_value=2, max_value=5, value=3,
        help="Number of bins used ONLY for continuous covariates (factors with more than "
             "10 unique values). Designed treatment factors — such as concentration levels "
             "or genotype — are always analyzed at their natural levels and are NOT affected "
             "by this setting."
    )
    cv_folds = st.sidebar.slider(
        "Cross-validation folds",
        min_value=3, max_value=10, value=5,
        help="Number of folds for K-Fold cross-validation. "
             "Higher values (e.g., 10) give more reliable estimates but take longer. "
             "Lower values (e.g., 3) are faster but have higher variance."
    )
    n_bootstrap = st.sidebar.number_input(
        "Bootstrap iterations",
        min_value=100, max_value=5000, value=1000, step=100,
        help="Number of bootstrap resamples for computing confidence intervals of standardized β coefficients. "
             "More iterations (e.g., 2000-5000) give more stable confidence intervals but take longer."
    )
    rf_estimators = st.sidebar.slider(
        "Random Forest estimators",
        min_value=50, max_value=500, value=200, step=50,
        help="Number of decision trees in the Random Forest model. "
             "More trees (e.g., 300-500) give smoother, more stable results but increase computation time."
    )
    stability_seeds = st.sidebar.selectbox(
        "Stability seed repeats",
        options=[1, 5, 10, 20],
        index=1,  # default = 5
        help="Number of random seeds for SHAP and CV R² stability analysis. "
             "1 = skip stability analysis, 5 = quick check (~1–2 min), "
             "20 = publication-grade (~3–5 min)."
    )
    
    run_shap_null = st.sidebar.checkbox(
        "Run SHAP null distribution and bootstrap CI",
        value=False,
        help="Permutes the response 50× and bootstraps the data 100× to determine "
             "which SHAP importances are distinguishable from chance. "
             "Adds 10–20 minutes for large datasets; recommended only for "
             "final analyses intended for publication."
    )
    random_state = st.sidebar.number_input(
        "Random seed",
        min_value=0, max_value=999, value=42,
        help="Seed for the random number generator. "
             "Using the same seed ensures reproducible results across runs. "
             "Change this value to test stability of your findings."
    )
    
    # ============================================================
    # FIGURE LAYOUT SETTINGS
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Figure Layout for Publication")
    st.sidebar.caption("Single-column (~85 mm) or two-column (~170 mm) A4 journal layout.")
    
    fig_layout = {}
    fig_layout['fig1'] = st.sidebar.selectbox(
        "Fig 1: SHAP Importance",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_fig1",
        help="Bar chart showing the relative importance (%) of each factor for every response, "
             "as determined by SHAP (SHapley Additive exPlanations) values from the Random Forest model."
    )
    fig_layout['fig2'] = st.sidebar.selectbox(
        "Fig 2: SHAP Beeswarm",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_fig2",
        help="Beeswarm plot visualizing individual SHAP values for each factor-response pair. "
             "Each dot is an observation; blue = high factor value, red = low factor value. "
             "The spread shows the direction and magnitude of each factor's effect."
    )
    fig_layout['fig3'] = st.sidebar.selectbox(
        "Fig 3: GPR Contours",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_fig3",
        help="Gaussian Process Regression contour plots for significant two-way interactions. "
             "Shows the predicted response surface as a function of two interacting factors. "
             "Red dots mark actual observations."
    )
    fig_layout['fig4'] = st.sidebar.selectbox(
        "Fig 4: MORF Residuals",
        options=["Single-Column", "Two-Column"],
        index=0, key="layout_fig4",
        help="Multi-Output Random Forest (MORF) residual correlation matrix. "
             "Shows correlations between response residuals after accounting for factor effects. "
             "Strong off-diagonal values suggest shared variance not explained by the factors."
    )
    fig_layout['fig5'] = st.sidebar.selectbox(
        "Fig 5: Standardized Beta",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_fig5",
        help="Standardized regression coefficients (β) with 95% bootstrap confidence intervals. "
             "Bars crossing zero indicate non-significant effects. "
             "Error bars show bootstrap CI uncertainty."
    )
    fig_layout['fig6'] = st.sidebar.selectbox(
        "Fig 6: Model Comparison",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_fig6",
        help="Cross-validated R² comparison of three models: Random Forest (RF), "
             "Linear Regression (LR), and Gaussian Process Regression (GPR). "
             "Error bars show ±1 standard deviation across CV folds."
    )
    fig_layout['fig7'] = st.sidebar.selectbox(
        "Fig 7: PCA Biplot",
        options=["Single-Column", "Two-Column"],
        index=0, key="layout_fig7",
        help="Principal Component Analysis biplot showing treatment groups in PC space. "
             "Red arrows indicate response variable loadings. "
             "Closer groups have more similar response profiles."
    )
    fig_layout['figS1'] = st.sidebar.selectbox(
        "Fig S1: Permutation Importance",
        options=["Two-Column", "Single-Column"],
        index=0, key="layout_figS1",
        help="Permutation-based feature importance as a validation of SHAP results. "
             "Measures the drop in model R² when each factor's values are randomly shuffled. "
             "Higher values indicate greater importance."
    )    
       
    # ============================================================
    # MAIN
    # ============================================================
    if uploaded_file is not None:
        RESPONSE_COLS = [r.strip() for r in response_input.split('\n') if r.strip()]
        FACTOR_COLS = [f for f in FACTOR_COLS if f.strip()]
        
        if len(FACTOR_COLS) < 1:
            st.error("❌ Please specify at least one factor column.")
        elif len(RESPONSE_COLS) < 1:
            st.error("❌ Please specify at least one response column.")
        else:
            try:
                df = pd.read_csv(uploaded_file, na_values=['NA', 'na', 'N/A', 'n/a', ''])
                df.columns = df.columns.str.strip()
                
                missing = [c for c in FACTOR_COLS + RESPONSE_COLS if c not in df.columns]
                if missing:
                    st.error(f"❌ Columns not found in CSV: {missing}")
                    st.info(f"Available columns: {list(df.columns)}")
                else:
                    n_factors = len(FACTOR_COLS)
                    n_responses = len(RESPONSE_COLS)
                    
                    st.success(f"✅ Data loaded: {len(df)} observations | {n_factors} factor(s) | {n_responses} response(s)")
                    
                    with st.expander("📊 Data Preview", expanded=False):
                        st.dataframe(df[FACTOR_COLS + RESPONSE_COLS].head(10))
                        
                        st.markdown("---")
                        st.markdown("#### Distribution by Treatment Group")
                        
                        # Create treatment labels for grouping
                        df_preview = df.copy()
                        if n_factors == 1:
                            try:
                                df_preview['_group_'] = df_preview[FACTOR_COLS[0]].round(1).astype(str)
                            except (TypeError, AttributeError):
                                df_preview['_group_'] = df_preview[FACTOR_COLS[0]].astype(str)
                        else:
                            parts = []
                            for f in FACTOR_COLS:
                                try:
                                    parts.append(df_preview[f].round(1).astype(str))
                                except (TypeError, AttributeError):
                                    parts.append(df_preview[f].astype(str))
                            df_preview['_group_'] = parts[0]
                            for p in parts[1:]:
                                df_preview['_group_'] = df_preview['_group_'] + '_' + p
                        
                        # Sort groups naturally
                        group_order = sorted(df_preview['_group_'].unique(), 
                                           key=lambda x: float(x.split('_')[0]) if x.split('_')[0].replace('.','').replace('-','').isdigit() else x)
                        
                        n_cols_box = min(3, n_responses)
                        n_rows_box = (n_responses + n_cols_box - 1) // n_cols_box
                        fig_box, axes_box = plt.subplots(n_rows_box, n_cols_box, 
                                                         figsize=(3.5 * n_cols_box, 3 * n_rows_box))
                        if n_responses == 1:
                            axes_box = np.array([axes_box])
                        axes_box = axes_box.flatten()
                        
                        for idx, resp in enumerate(RESPONSE_COLS):
                            ax = axes_box[idx]
                            plot_data = [df_preview[df_preview['_group_'] == g][resp].dropna().values 
                                        for g in group_order]
                            bp = ax.boxplot(plot_data, tick_labels=group_order, patch_artist=True,
                                           widths=0.6, showfliers=False)
                            for patch in bp['boxes']:
                                patch.set_facecolor('#5DADE2')
                                patch.set_alpha(0.7)
                            
                            # Add jittered points
                            for gi, g in enumerate(group_order):
                                vals = df_preview[df_preview['_group_'] == g][resp].dropna().values
                                if len(vals) > 0:
                                    jitter = np.random.normal(gi + 1, 0.04, len(vals))
                                    ax.scatter(jitter, vals, color='black', s=12, alpha=0.5, 
                                              edgecolors='none', zorder=5)
                            
                            ax.set_title(resp, fontsize=10, fontweight='bold')
                            ax.set_ylabel(resp if idx % n_cols_box == 0 else '', fontsize=8)
                            ax.tick_params(axis='x', rotation=45, labelsize=7)
                            ax.tick_params(axis='y', labelsize=7)
                            ax.grid(True, axis='y', alpha=0.3)
                        
                        for idx in range(n_responses, len(axes_box)):
                            axes_box[idx].set_visible(False)
                        
                        fig_box.suptitle('Response Distributions by Treatment', fontsize=11, fontweight='bold', y=1.02)
                        fig_box.tight_layout()
                        st.pyplot(fig_box)
                        plt.close(fig_box)
                        df_preview.drop(columns=['_group_'], inplace=True)
                    
                    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                        
                        # Convert response columns to numeric
                        for resp in RESPONSE_COLS:
                            df[resp] = pd.to_numeric(df[resp], errors='coerce')
                        
                        # ---- Factor type handling ----
                        # Designed treatment factors (few unique levels) are treated as categorical
                        # at their natural levels. Only genuinely continuous covariates (many unique
                        # values) are optionally binned into quantile categories for ANOVA.
                        AUTO_CATEGORICAL_THRESHOLD = 10
                        _ALL_CAT_LABELS = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
                        cat_labels = _ALL_CAT_LABELS[:n_categories]
                        if n_categories > len(_ALL_CAT_LABELS):
                            cat_labels = [f'Level {i+1}' for i in range(n_categories)]

                        factor_type_log = {}   # records how each factor was treated (for the report)

                        for factor in FACTOR_COLS:
                            unique_vals = df[factor].dropna().unique()
                            n_unique = len(unique_vals)

                            if df[factor].dtype == 'object' or n_unique <= AUTO_CATEGORICAL_THRESHOLD:
                                # Categorical: designed treatment levels preserved
                                df[f'{factor}_cat'] = df[factor].astype(str)
                                factor_type_log[factor] = f'categorical ({n_unique} levels, natural)'
                            else:
                                # Continuous: optionally binned
                                try:
                                    df[f'{factor}_cat'] = pd.qcut(df[factor], q=n_categories,
                                                              labels=cat_labels, duplicates='drop')
                                    factor_type_log[factor] = f'continuous (binned into {n_categories} quantile groups)'
                                except ValueError:
                                    df[f'{factor}_cat'] = pd.cut(df[factor], bins=n_categories, labels=cat_labels)
                                    factor_type_log[factor] = f'continuous (binned into {n_categories} equal-width groups)'

                        # Display factor type decisions to the user
                        with st.expander("ℹ️ Factor type detection", expanded=False):
                            for f, status in factor_type_log.items():
                                st.write(f"**{f}**: {status}")
                        
                        # Encode categorical factors for ML
                        df_encoded = df.copy()
                        for factor in FACTOR_COLS:
                            try:
                                df_encoded[factor] = pd.to_numeric(df[factor], errors='raise')
                            except (ValueError, TypeError):
                                df_encoded[factor] = LabelEncoder().fit_transform(df[factor].astype(str))
                        
                        X = df_encoded[FACTOR_COLS].values.astype(float)
                        Y = df[RESPONSE_COLS].values.astype(float)
                        
                        # Handle NaN for ML
                        df_ml = df.dropna(subset=RESPONSE_COLS)
                        if len(df_ml) < len(df):
                            st.warning(f"⚠️ Dropped {len(df) - len(df_ml)} rows with missing values "
                                      f"({len(df_ml)} rows retained for ML analyses).")
                        
                        df_ml_encoded = df_encoded.loc[df_ml.index]
                        X_ml = df_ml_encoded[FACTOR_COLS].values.astype(float)
                        Y_ml = df_ml[RESPONSE_COLS].values.astype(float)
                        
                        progress_bar = st.progress(0, text="Starting analysis...")
                        
                        # ============================================================
                        # STATISTICAL PRELIMINARIES
                        # ============================================================
                        prelim_results = {}
                        for resp in RESPONSE_COLS:
                            resp_data = df[resp].dropna().values
                            if len(resp_data) < 3:
                                prelim_results[resp] = {
                                    'Shapiro_W': np.nan, 'Shapiro_p': np.nan,
                                    'Normality': 'Insufficient data',
                                    'Shapiro_W_marginal': np.nan, 'Shapiro_p_marginal': np.nan,
                                    'Levene_W': np.nan, 'Levene_p': np.nan,
                                    'Homogeneity': 'Insufficient data'
                                }
                                continue
                            
                            # Marginal Shapiro (kept for reference only)
                            try:
                                stat_shapiro_marg, p_shapiro_marg = shapiro(resp_data)
                            except:
                                stat_shapiro_marg, p_shapiro_marg = np.nan, np.nan
                            
                            # Residual-based Shapiro (the correct assumption check)
                            try:
                                _main_eff = ' + '.join([f'C({f}_cat)' for f in FACTOR_COLS])
                                if n_factors >= 2:
                                    _inter = ' + '.join([f'C({f1}_cat):C({f2}_cat)'
                                                          for f1, f2 in combinations(FACTOR_COLS, 2)])
                                    _formula = f'Q("{resp}") ~ {_main_eff} + {_inter}'
                                else:
                                    _formula = f'Q("{resp}") ~ {_main_eff}'
                                _df_model = df.dropna(subset=[resp])
                                _model = ols(_formula, data=_df_model).fit()
                                _residuals = _model.resid.values
                                stat_shapiro, p_shapiro = shapiro(_residuals)
                                normality = "Normal (residuals)" if p_shapiro >= 0.05 else "Non-normal (residuals)"
                            except:
                                stat_shapiro, p_shapiro, normality = np.nan, np.nan, "Error"
                            
                            # Levene's test for homogeneity of variance
                            try:
                                if n_factors >= 2:
                                    f1, f2 = FACTOR_COLS[0], FACTOR_COLS[1]
                                    df['_group_'] = df[f1].astype(str) + '_' + df[f2].astype(str)
                                    groups = [g[resp].dropna().values for _, g in df.groupby('_group_') if len(g[resp].dropna()) >= 2]
                                    if len(groups) >= 2:
                                        stat_levene, p_levene = levene(*groups)
                                        homogeneity = "Equal" if p_levene >= 0.05 else "Unequal"
                                    else:
                                        stat_levene, p_levene, homogeneity = np.nan, np.nan, "N/A"
                                    df.drop(columns=['_group_'], inplace=True)
                                elif n_factors == 1:
                                    f1 = FACTOR_COLS[0]
                                    cat_col = f'{f1}_cat'
                                    groups = [g[resp].dropna().values for _, g in df.groupby(cat_col) if len(g[resp].dropna()) >= 2]
                                    if len(groups) >= 2:
                                        stat_levene, p_levene = levene(*groups)
                                        homogeneity = "Equal" if p_levene >= 0.05 else "Unequal"
                                    else:
                                        stat_levene, p_levene, homogeneity = np.nan, np.nan, "N/A"
                                else:
                                    stat_levene, p_levene, homogeneity = np.nan, np.nan, "N/A"
                            except:
                                stat_levene, p_levene, homogeneity = np.nan, np.nan, "Error"
                            
                            prelim_results[resp] = {
                                'Shapiro_W': stat_shapiro, 'Shapiro_p': p_shapiro,
                                'Normality': normality,
                                'Shapiro_W_marginal': stat_shapiro_marg,
                                'Shapiro_p_marginal': p_shapiro_marg,
                                'Levene_W': stat_levene,
                                'Levene_p': p_levene, 'Homogeneity': homogeneity
                            }
                        
                        # Outlier Detection
                        outlier_results = {}
                        for resp in RESPONSE_COLS:
                            all_values = df[resp].dropna().values
                            outlier_details = []; total_outliers = 0
                            if len(all_values) >= 4:
                                Q1_all, Q3_all = np.percentile(all_values, 25), np.percentile(all_values, 75)
                                IQR_all = Q3_all - Q1_all
                                lower_bound_all, upper_bound_all = Q1_all - 1.5*IQR_all, Q3_all + 1.5*IQR_all
                                median_all = np.median(all_values)
                                mad = np.median(np.abs(all_values - median_all))
                                modified_z_scores = 0.6745*(all_values - median_all)/mad if mad > 0 else np.zeros_like(all_values)
                                for idx, val in enumerate(all_values):
                                    reasons = []
                                    if val < lower_bound_all or val > upper_bound_all:
                                        reasons.append(f"IQR [{lower_bound_all:.2f}, {upper_bound_all:.2f}]")
                                    if abs(modified_z_scores[idx]) > 3.5:
                                        reasons.append(f"Z-score={modified_z_scores[idx]:.2f}")
                                    if reasons:
                                        outlier_details.append({'Row': int(idx), 'Value': round(val,3),
                                                              'Detected by': '; '.join(reasons),
                                                              'Z-score': round(modified_z_scores[idx],2)})
                                        total_outliers += 1
                            outlier_results[resp] = {'n_outliers': total_outliers, 'details': outlier_details,
                                                     'IQR_bounds': f'[{lower_bound_all:.2f}, {upper_bound_all:.2f}]' if len(all_values) >= 4 else 'N/A'}
                        
                        # Prelim table
                        prelim_rows = []
                        for resp in RESPONSE_COLS:
                            pr = prelim_results[resp]
                            n_out = outlier_results[resp]['n_outliers']
                            prelim_rows.append({
                                'Response': resp,
                                'Shapiro p (resid)': f"{pr['Shapiro_p']:.4f}" if not np.isnan(pr['Shapiro_p']) else 'N/A',
                                'Residuals': pr['Normality'],
                                'Shapiro p (marg)': f"{pr['Shapiro_p_marginal']:.4f}" if not np.isnan(pr['Shapiro_p_marginal']) else 'N/A',
                                'Levene p': f"{pr['Levene_p']:.4f}" if not np.isnan(pr['Levene_p']) else 'N/A',
                                'Variances': pr['Homogeneity'],
                                'IQR Bounds': outlier_results[resp]['IQR_bounds'],
                                'Outliers': f"⚠️ {n_out}" if n_out > 0 else "✅ None"
                            })
                        df_prelim = pd.DataFrame(prelim_rows)
                        
                        # Pearson Correlations
                        pearson_rows = []
                        for resp in RESPONSE_COLS:
                            for factor in FACTOR_COLS:
                                x_vals = pd.to_numeric(df[factor], errors='coerce')
                                y_vals = pd.to_numeric(df[resp], errors='coerce')
                                mask = x_vals.notna() & y_vals.notna()
                                if mask.sum() >= 3:
                                    r, p = pearsonr(x_vals[mask], y_vals[mask])
                                    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
                                else:
                                    r, p = np.nan, np.nan
                                    sig = 'N/A'
                                pearson_rows.append({'Response': resp, 'Factor': factor,
                                                    'Pearson r': round(r,4) if not np.isnan(r) else 'N/A',
                                                    'p-value': round(p,4) if not np.isnan(p) else 'N/A',
                                                    'Sig.': sig})
                        df_pearson = pd.DataFrame(pearson_rows)
                        
                        progress_bar.progress(5, text="Assumption checks complete...")
                        
                        # ============================================================
                        # N-Way ANOVA
                        # ============================================================
                        progress_bar.progress(8, text="Running N-way ANOVA...")
                        all_int = []
                        full_anova_tables = {}
                        simple_effects_records = []
                        for resp in RESPONSE_COLS:
                            main_effects = ' + '.join([f'C({f}_cat)' for f in FACTOR_COLS])
                            if n_factors >= 2:
                                interactions = ' + '.join([f'C({f1}_cat):C({f2}_cat)' for f1,f2 in combinations(FACTOR_COLS,2)])
                                formula = f'Q("{resp}") ~ {main_effects} + {interactions}'
                            else:
                                formula = f'Q("{resp}") ~ {main_effects}'
                            try:
                                df_anova = df.dropna(subset=[resp])
                                model = ols(formula, data=df_anova).fit()
                                atbl = sm.stats.anova_lm(model, typ=3)
                                ss_error = atbl.loc['Residual', 'sum_sq']
                                eta_sq = {}
                                for idx_row in atbl.index:
                                    if idx_row != 'Residual':
                                        eta_sq[idx_row] = atbl.loc[idx_row,'sum_sq']/(atbl.loc[idx_row,'sum_sq']+ss_error)
                                atbl['η²'] = [eta_sq.get(idx,np.nan) for idx in atbl.index]
                                # ---- Classical and bias-corrected effect sizes (Item 1k) ----
                                ms_error = atbl.loc['Residual', 'sum_sq'] / atbl.loc['Residual', 'df']
                                for idx_row in atbl.index:
                                    if idx_row == 'Residual':
                                        continue
                                    try:
                                        es = compute_effect_sizes(
                                            df_term=atbl.loc[idx_row, 'df'],
                                            df_error=atbl.loc['Residual', 'df'],
                                            ss_term=atbl.loc[idx_row, 'sum_sq'],
                                            ss_error=atbl.loc['Residual', 'sum_sq'],
                                            ms_error=ms_error,
                                            n_total=len(df_anova))
                                        atbl.loc[idx_row, 'ω²'] = es['omega2']
                                        atbl.loc[idx_row, 'ε²'] = es['epsilon2']
                                    except Exception:
                                        atbl.loc[idx_row, 'ω²'] = np.nan
                                        atbl.loc[idx_row, 'ε²'] = np.nan
                                # Remove Intercept row from display (Type III includes it)
                                if 'Intercept' in atbl.index:
                                    atbl = atbl.drop('Intercept')
                                # BH FDR correction within this response
                                pvals = atbl['PR(>F)'].values.astype(float).copy()
                                atbl['q_within'] = benjamini_hochberg(pvals)
                                full_anova_tables[resp] = atbl
                                if n_factors >= 2:
                                    for f1,f2 in combinations(FACTOR_COLS,2):
                                        iterm = f'C({f1}_cat):C({f2}_cat)'
                                        if iterm in atbl.index:
                                            pi = atbl.loc[iterm,'PR(>F)']
                                            fi = atbl.loc[iterm,'F']
                                            sig = '***' if pi<0.001 else '**' if pi<0.01 else '*' if pi<0.05 else 'ns'
                                            itype = 'Synergistic' if (pi<0.05 and fi>10) else ('Moderate' if pi<0.05 else 'Additive')
                                            all_int.append({'Response':resp,'Factor_1':f1,'Factor_2':f2,
                                                          'F_int':fi,'p_int':pi,'Sig.':sig,'Type':itype})
                                            # Simple effects if interaction is significant (Reviewer 3, Suggestion 1)
                                            if pi < 0.05 and n_factors == 2:
                                                try:
                                                    se_f1 = simple_effects(
                                                        df_anova, resp, f'{f1}_cat', f'{f2}_cat')
                                                    se_f2 = simple_effects(
                                                        df_anova, resp, f'{f2}_cat', f'{f1}_cat')
                                                    simple_effects_records.append({
                                                        'Response': resp,
                                                        'Interaction': f'{f1} × {f2}',
                                                        f'{f1} at levels of {f2}': se_f1,
                                                        f'{f2} at levels of {f1}': se_f2,
                                                    })
                                                except Exception as ex:
                                                    pass
                            except Exception as ex:
                                st.warning(f"ANOVA failed for {resp}: {str(ex)[:80]}")
                        df_int = pd.DataFrame(all_int) if all_int else pd.DataFrame()
                        sig_int = df_int[df_int['Sig.']!='ns'] if len(df_int)>0 else df_int
                        df_simple = pd.DataFrame(simple_effects_records) if simple_effects_records else pd.DataFrame()
                        
                        # ---- Cross-response FDR ----
                        # ---- Cross-response FDR ----
                        # For each ANOVA term, correct p-values across responses.
                        anova_fdr_records = []
                        for resp, atbl in full_anova_tables.items():
                            for term in atbl.index:
                                if term == 'Residual':
                                    continue
                                anova_fdr_records.append({
                                    'Response': resp,
                                    'Term': term,
                                    'F': atbl.loc[term, 'F'],
                                    'p_raw': atbl.loc[term, 'PR(>F)'],
                                    'q_within': atbl.loc[term, 'q_within'],
                                    'η²': atbl.loc[term, 'η²'],
                                    'ω²': atbl.loc[term, 'ω²'] if 'ω²' in atbl.columns else np.nan,
                                    'ε²': atbl.loc[term, 'ε²'] if 'ε²' in atbl.columns else np.nan,
                                })
                        df_anova_fdr = pd.DataFrame(anova_fdr_records)
                        
                        # Cross-response correction: group by Term, apply BH
                        if len(df_anova_fdr) > 0:
                            q_across = np.full(len(df_anova_fdr), np.nan)
                            for term, idx in df_anova_fdr.groupby('Term').groups.items():
                                sub_p = df_anova_fdr.loc[idx, 'p_raw'].values.astype(float)
                                sub_q = benjamini_hochberg(sub_p)
                                q_across[df_anova_fdr.index.get_indexer(idx)] = sub_q
                            df_anova_fdr['q_across'] = q_across

                                                
                        # Kruskal-Wallis
                        kw_validation = []
                        for resp in RESPONSE_COLS:
                            for factor in FACTOR_COLS:
                                cat_col = f'{factor}_cat'
                                try:
                                    groups_kw = [g[resp].dropna().values for _,g in df.groupby(cat_col) if len(g[resp].dropna())>=2]
                                    if len(groups_kw) >= 2:
                                        h_stat, p_kw = kruskal(*groups_kw)
                                    else:
                                        continue
                                    p_anova = np.nan
                                    if resp in full_anova_tables:
                                        anova_term = f'C({factor}_cat)'
                                        if anova_term in full_anova_tables[resp].index:
                                            p_anova = full_anova_tables[resp].loc[anova_term,'PR(>F)']
                                    if not np.isnan(p_anova):
                                        if p_anova<0.05 and p_kw<0.05: agreement = "✅ Both significant"
                                        elif p_anova>=0.05 and p_kw>=0.05: agreement = "✅ Both non-significant"
                                        elif p_anova<0.05: agreement = "⚠️ ANOVA only"
                                        else: agreement = "⚠️ KW only"
                                    else:
                                        agreement = "N/A"
                                    kw_validation.append({'Response':resp,'Factor':factor,
                                                        'ANOVA p':f"{p_anova:.4f}" if not np.isnan(p_anova) else 'N/A',
                                                        'KW H-statistic':f"{h_stat:.2f}",'KW p-value':f"{p_kw:.4f}",
                                                        'Agreement':agreement})
                                except: pass
                        df_kw = pd.DataFrame(kw_validation)

                        # ---- ART ANOVA (Item 1g) ----
                        art_rows = []
                        art_cat_cols = [f'{f}_cat' for f in FACTOR_COLS]
                        for resp in RESPONSE_COLS:
                            try:
                                art_res = art_anova(df, resp, art_cat_cols)
                                # For each main effect + interaction, add a row
                                for term, stats_ in art_res.items():
                                    # Extract factor names from term for display
                                    if ':' in term:
                                        f1, f2 = term.split(':')
                                        factor_display = f"{f1.replace('_cat','')} × {f2.replace('_cat','')}"
                                    else:
                                        factor_display = term.replace('_cat','')
                                    art_rows.append({
                                        'Response': resp,
                                        'Effect': factor_display,
                                        'ART_F': stats_['F'],
                                        'ART_p': stats_['p'],
                                    })
                            except Exception as ex:
                                st.warning(f"ART ANOVA failed for {resp}: {str(ex)[:80]}")
                        df_art = pd.DataFrame(art_rows) if art_rows else pd.DataFrame()
                        
                        progress_bar.progress(15, text="Running SHAP analysis...")
                        
                        # ---- SHAP ----
                        shap_imp = np.zeros((n_responses, n_factors))
                        for idx, resp in enumerate(RESPONSE_COLS):
                            try:
                                rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=None,
                                                           min_samples_leaf=2, random_state=random_state, n_jobs=-1)
                                rf.fit(X_ml, Y_ml[:,idx])
                                explainer = shap.TreeExplainer(rf)
                                shap_imp[idx,:] = np.abs(explainer.shap_values(X_ml)).mean(axis=0)
                            except Exception as ex:
                                st.warning(f"SHAP failed for {resp}: {str(ex)[:80]}")
                        
                        progress_bar.progress(25, text="Running permutation importance...")
                        
                        # ---- Permutation ----
                        perm_imp = np.zeros((n_responses, n_factors))
                        perm_imp_std = np.zeros((n_responses, n_factors))
                        for idx, resp in enumerate(RESPONSE_COLS):
                            try:
                                rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=None,
                                                           min_samples_leaf=2, random_state=random_state, n_jobs=-1)
                                rf.fit(X_ml, Y_ml[:,idx])
                                perm_result = permutation_importance(rf, X_ml, Y_ml[:,idx], n_repeats=20,
                                                                    random_state=random_state, scoring='r2')
                                perm_imp[idx,:] = perm_result.importances_mean
                                perm_imp_std[idx,:] = perm_result.importances_std
                            except Exception as ex:
                                st.warning(f"Permutation failed for {resp}: {str(ex)[:80]}")
                        
                        progress_bar.progress(35, text="Running cross-validation...")
                        
                        # ---- CV ----
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                        
                        # ---- GroupKFold by treatment combination (Item 1d) ----
                        # Prevents leakage when replicate samples share a treatment cell.
                        # Encode each unique treatment combination as an integer group ID.
                        # Factor values are already numeric (continuous) or label-encoded
                        # (categorical) at this point, so tuple-as-key works, then we map
                        # each unique tuple to a small integer.
                        _group_tuples = [
                            tuple(row) for row in df_ml_encoded[FACTOR_COLS].values
                        ]
                        _unique_groups = {t: i for i, t in enumerate(set(_group_tuples))}
                        treatment_groups = np.array(
                            [_unique_groups[t] for t in _group_tuples],
                            dtype=int
                        )
                        n_groups = len(_unique_groups)
                   
                        if n_groups >= cv_folds:
                            cv_group = GroupKFold(n_splits=cv_folds)
                        elif n_groups >= 2:
                            cv_group = GroupKFold(n_splits=n_groups)
                            st.info(f"GroupKFold reduced to {n_groups} folds "
                                    f"(number of treatment combinations).")
                        else:
                            cv_group = None
                            st.warning("Too few treatment groups for GroupKFold; "
                                       "only standard KFold will be shown.")
                        
                        model_comparison = []
                        for idx, resp in enumerate(RESPONSE_COLS):
                            try:
                                rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=None,
                                                           min_samples_leaf=2, random_state=random_state, n_jobs=-1)
                                lr = LinearRegression()
                                from sklearn.pipeline import make_pipeline
                                gpr = make_pipeline(
                                    StandardScaler(),
                                    GaussianProcessRegressor(
                                        kernel=RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
                                             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)),
                                        n_restarts_optimizer=10,
                                        normalize_y=True,
                                        random_state=random_state)
                                )
                                
                                # Standard KFold (original)
                                rf_cv   = cross_val_score(rf,  X_ml, Y_ml[:,idx], cv=cv,       scoring='r2')
                                lr_cv   = cross_val_score(lr,  X_ml, Y_ml[:,idx], cv=cv,       scoring='r2')
                                gpr_cv  = cross_val_score(gpr, X_ml, Y_ml[:,idx], cv=cv,       scoring='r2')
                                
                                # GroupKFold (Item 1d) — no leakage across replicate cells
                                if cv_group is not None:
                                    rf_cv_g  = cross_val_score(rf,  X_ml, Y_ml[:,idx],
                                                               cv=cv_group, groups=treatment_groups,
                                                               scoring='r2')
                                    lr_cv_g  = cross_val_score(lr,  X_ml, Y_ml[:,idx],
                                                               cv=cv_group, groups=treatment_groups,
                                                               scoring='r2')
                                    gpr_cv_g = cross_val_score(gpr, X_ml, Y_ml[:,idx],
                                                               cv=cv_group, groups=treatment_groups,
                                                               scoring='r2')
                                else:
                                    rf_cv_g = lr_cv_g = gpr_cv_g = None
                                
                                best = 'RF' if rf_cv.mean()>=max(lr_cv.mean(),gpr_cv.mean()) else \
                                       'LR' if lr_cv.mean()>=max(rf_cv.mean(),gpr_cv.mean()) else 'GPR'
                                
                                def format_ttest(a, b, name_a, name_b):
                                    """Perform paired t-test and return formatted string."""
                                    try:
                                        t_stat, p_val = ttest_rel(a, b)
                                        if p_val < 0.001:
                                            sig = f'*** (p={p_val:.4f})'
                                        elif p_val < 0.01:
                                            sig = f'** (p={p_val:.4f})'
                                        elif p_val < 0.05:
                                            sig = f'* (p={p_val:.4f})'
                                        else:
                                            sig = f'ns (p={p_val:.4f})'
                                        if a.mean() > b.mean():
                                            return f'{name_a} > {name_b}: {sig}'
                                        else:
                                            return f'{name_b} > {name_a}: {sig}'
                                    except:
                                        return 'N/A'
                                
                                test_rf_lr  = format_ttest(rf_cv,  lr_cv,  'RF', 'LR')
                                test_rf_gpr = format_ttest(rf_cv,  gpr_cv, 'RF', 'GPR')
                                test_lr_gpr = format_ttest(lr_cv,  gpr_cv, 'LR', 'GPR')
                                
                                # Nadeau-Bengio corrected t-test (Item 1e)
                                def format_nb(a, b, name_a, name_b):
                                    t_stat, p_val = nadeau_bengio_ttest(a, b, cv_folds)
                                    if np.isnan(p_val):
                                        return 'N/A'
                                    if p_val < 0.001: sig = f'*** (NB p={p_val:.4f})'
                                    elif p_val < 0.01: sig = f'** (NB p={p_val:.4f})'
                                    elif p_val < 0.05: sig = f'* (NB p={p_val:.4f})'
                                    else: sig = f'ns (NB p={p_val:.4f})'
                                    if np.mean(a) > np.mean(b):
                                        return f'{name_a} > {name_b}: {sig}'
                                    else:
                                        return f'{name_b} > {name_a}: {sig}'
                                
                                test_rf_lr_nb  = format_nb(rf_cv,  lr_cv,  'RF', 'LR')
                                test_rf_gpr_nb = format_nb(rf_cv,  gpr_cv, 'RF', 'GPR')
                                test_lr_gpr_nb = format_nb(lr_cv,  gpr_cv, 'LR', 'GPR')
                                
                                # GroupKFold t-tests (Item 1d)
                                if rf_cv_g is not None:
                                    test_rf_lr_g  = format_ttest(rf_cv_g,  lr_cv_g,  'RF', 'LR')
                                    test_rf_gpr_g = format_ttest(rf_cv_g,  gpr_cv_g, 'RF', 'GPR')
                                    test_lr_gpr_g = format_ttest(lr_cv_g,  gpr_cv_g, 'LR', 'GPR')
                                    # GroupKFold Nadeau-Bengio (Item 1e)
                                    test_rf_lr_g_nb  = format_nb(rf_cv_g,  lr_cv_g,  'RF', 'LR')
                                    test_rf_gpr_g_nb = format_nb(rf_cv_g,  gpr_cv_g, 'RF', 'GPR')
                                    test_lr_gpr_g_nb = format_nb(lr_cv_g,  gpr_cv_g, 'LR', 'GPR')
                                else:
                                    test_rf_lr_g = test_rf_gpr_g = test_lr_gpr_g = 'N/A'
                                    test_rf_lr_g_nb = test_rf_gpr_g_nb = test_lr_gpr_g_nb = 'N/A'
                               
                                model_comparison.append({
                                    'Response': resp,
                                    'RF_R2_mean': rf_cv.mean(),  'RF_R2_std': rf_cv.std(),
                                    'LR_R2_mean': lr_cv.mean(),  'LR_R2_std': lr_cv.std(),
                                    'GPR_R2_mean': gpr_cv.mean(), 'GPR_R2_std': gpr_cv.std(),
                                    'Best': best,
                                    'RF_vs_LR': test_rf_lr,
                                    'RF_vs_GPR': test_rf_gpr,
                                    'LR_vs_GPR': test_lr_gpr,
                                    # Nadeau-Bengio (Item 1e) — standard KFold
                                    'RF_vs_LR_NB':  test_rf_lr_nb,
                                    'RF_vs_GPR_NB': test_rf_gpr_nb,
                                    'LR_vs_GPR_NB': test_lr_gpr_nb,
                                    # GroupKFold results (Item 1d)
                                    'RF_R2_group':  rf_cv_g.mean()  if rf_cv_g  is not None else np.nan,
                                    'LR_R2_group':  lr_cv_g.mean()  if lr_cv_g  is not None else np.nan,
                                    'GPR_R2_group': gpr_cv_g.mean() if gpr_cv_g is not None else np.nan,
                                    # GroupKFold t-tests (Item 1d)
                                    'RF_vs_LR_group':  test_rf_lr_g,
                                    'RF_vs_GPR_group': test_rf_gpr_g,
                                    'LR_vs_GPR_group': test_lr_gpr_g,
                                    # Nadeau-Bengio (Item 1e) — GroupKFold
                                    'RF_vs_LR_group_NB':  test_rf_lr_g_nb,
                                    'RF_vs_GPR_group_NB': test_rf_gpr_g_nb,
                                    'LR_vs_GPR_group_NB': test_lr_gpr_g_nb,
                                })
                            except Exception as ex:
                                st.warning(f"CV failed for {resp}: {str(ex)[:80]}")
                                model_comparison.append({
                                    'Response': resp,
                                    'RF_R2_mean': np.nan, 'RF_R2_std': np.nan,
                                    'LR_R2_mean': np.nan, 'LR_R2_std': np.nan,
                                    'GPR_R2_mean': np.nan, 'GPR_R2_std': np.nan,
                                    'Best': 'Error',
                                    'RF_vs_LR': 'N/A', 'RF_vs_GPR': 'N/A', 'LR_vs_GPR': 'N/A',
                                    'RF_vs_LR_NB': 'N/A', 'RF_vs_GPR_NB': 'N/A', 'LR_vs_GPR_NB': 'N/A',
                                    'RF_R2_group': np.nan, 'LR_R2_group': np.nan,
                                    'GPR_R2_group': np.nan,
                                    'RF_vs_LR_group': 'N/A', 'RF_vs_GPR_group': 'N/A', 
                                    'LR_vs_GPR_group': 'N/A',
                                    'RF_vs_LR_group_NB': 'N/A', 'RF_vs_GPR_group_NB': 'N/A',
                                    'LR_vs_GPR_group_NB': 'N/A',
                                })
                        df_mc = pd.DataFrame(model_comparison)
                        
                        progress_bar.progress(45, text="Computing standardized coefficients...")
                        
                        # ---- Beta ----
                        sX, sY = StandardScaler(), StandardScaler()
                        Xs, Ys = sX.fit_transform(X_ml), sY.fit_transform(Y_ml)
                        beta = np.zeros((n_factors, n_responses))
                        for i in range(n_responses):
                            lr = LinearRegression(); lr.fit(Xs, Ys[:,i]); beta[:,i] = lr.coef_
                        beta_bootstrap = np.zeros((n_bootstrap, n_factors, n_responses))
                        np.random.seed(random_state)
                        for b in range(n_bootstrap):
                            idx_boot = np.random.choice(len(Xs), size=len(Xs), replace=True)
                            for i in range(n_responses):
                                lr_b = LinearRegression(); lr_b.fit(Xs[idx_boot], Ys[idx_boot,i])
                                beta_bootstrap[b,:,i] = lr_b.coef_
                        beta_ci_lower = np.percentile(beta_bootstrap, 2.5, axis=0)
                        beta_ci_upper = np.percentile(beta_bootstrap, 97.5, axis=0)
                        
                        # ---- MORF ----
                        try:
                            if n_responses >= 2:
                                # Native multi-output Random Forest: shares tree structure
                                # across responses, unlike MultiOutputRegressor which fits
                                # one independent model per response.
                                mor = RandomForestRegressor(
                                    n_estimators=rf_estimators, max_depth=None,
                                    min_samples_leaf=2, random_state=random_state, n_jobs=-1)
                                mor.fit(X_ml, Y_ml)                     # natively multi-output
                                resid_corr = np.corrcoef((Y_ml - mor.predict(X_ml)).T)
                            else:
                                resid_corr = np.array([[1.0]])
                        except Exception:
                            resid_corr = np.eye(max(n_responses, 1))

                        # ---- Out-of-fold MORF residuals (Item 1j) ----
                        resid_corr_oof = None
                        if n_responses >= 2:
                            try:
                                resid_corr_oof = morf_out_of_fold_residuals(
                                    X_ml, Y_ml, cv_folds=cv_folds,
                                    rf_estimators=rf_estimators,
                                    random_state=random_state)
                            except Exception as ex:
                                st.warning(f"Out-of-fold MORF failed: {str(ex)[:80]}")
                                resid_corr_oof = None

                        # ---- Multi-seed stability analysis (Item 1l) ----
                        stability_results = None
                        if stability_seeds > 1:
                            progress_bar.progress(55, text=f"Running stability analysis ({stability_seeds} seeds)...")
                            try:
                                stability_results = run_seed_stability(
                                    X_ml, Y_ml, FACTOR_COLS, n_responses,
                                    rf_estimators=rf_estimators,
                                    cv_folds=cv_folds, n_seeds=stability_seeds)
                            except Exception as ex:
                                st.warning(f"Stability analysis failed: {str(ex)[:100]}")
                                stability_results = None

                        # ---- SHAP null distribution and bootstrap (Item 1f) ----
                        shap_null_results = None
                        if run_shap_null:
                            progress_bar.progress(57, text="Running SHAP null distribution and bootstrap (this may take several minutes)...")
                            try:
                                shap_null_results = shap_null_and_bootstrap(
                                    X_ml, Y_ml, n_responses,
                                    rf_estimators=100,
                                    n_permutations=50, n_bootstrap=100,
                                    n_jobs=-1, random_state=random_state)
                            except Exception as ex:
                                st.warning(f"SHAP null/bootstrap failed: {str(ex)[:100]}")
                                shap_null_results = None
                        
                        progress_bar.progress(60, text="Running PCA...")
                        
                        # ---- PCA (correlation-based, ddof=1) ----
                        # Manual standardization with ddof=1 so that the covariance
                        # matrix equals the correlation matrix and eigenvalues sum to p.
                        Y_scaled = standardize_ddof1(Y_ml)
                        pca = PCA(n_components=n_responses)   # extract ALL components
                        pca_scores = pca.fit_transform(Y_scaled)
                        pca_loadings = pca.components_.T
                        pca_var_explained = pca.explained_variance_ratio_
                        # pca.explained_variance_ now sums to n_responses, ratios sum to 1.
                        
                        df_pca_summary = pd.DataFrame([{
                            'Component':      f'PC{i+1}',
                            'Eigenvalue':     f"{pca.explained_variance_[i]:.3f}",
                            'Variance (%)':   f"{pca_var_explained[i]*100:.1f}",
                            'Cumulative (%)': f"{np.cumsum(pca_var_explained)[i]*100:.1f}"
                        } for i in range(n_responses)])   # all components, not just 5
                        
                        pca_loading_rows = []
                        for i, resp in enumerate(RESPONSE_COLS):
                            row = {'Response': resp}
                            for j in range(n_responses):          # expose all PCs
                                row[f'PC{j+1}'] = round(pca_loadings[i, j], 3)
                            pca_loading_rows.append(row)
                        df_pca_loadings = pd.DataFrame(pca_loading_rows)
                        
                        progress_bar.progress(65, text="Computing post-hoc comparisons...")
                        
                        # ============================================================
                        # POST-HOC (via R's agricolae subprocess)
                        # ============================================================
                        posthoc_dict = {}
                        if n_factors >= 1:
                            # Create treatment labels from ALL factors
                            treatment_parts = []
                            for f in FACTOR_COLS:
                                try:
                                    treatment_parts.append(df[f].round(1).astype(str))
                                except (TypeError, AttributeError):
                                    treatment_parts.append(df[f].astype(str))
                            df['Treatment'] = treatment_parts[0]
                            for part in treatment_parts[1:]:
                                df['Treatment'] = df['Treatment'] + '_' + part
                        
                            if not R_AVAILABLE:
                                st.warning(f"⚠️ R/agricolae not available. Using Python-based CLD. ({_R_ERROR_MSG})")
                            
                            for resp in RESPONSE_COLS:
                                try:
                                    if R_AVAILABLE:
                                        resp_data = df[['Treatment', resp]].dropna().copy()
                                        
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                                            resp_data.to_csv(f.name, index=False)
                                            csv_path = f.name
                                        
                                        r_script = f'''
                                        library(agricolae)
                                        data <- read.csv("{csv_path}")
                                        data$Treatment <- as.factor(data$Treatment)
                                        model <- aov({resp} ~ Treatment, data=data)
                                        if ("{posthoc_method}" == "Tukey's HSD") {{
                                            test <- HSD.test(model, "Treatment", group=TRUE, console=FALSE)
                                        }} else {{
                                            test <- LSD.test(model, "Treatment", group=TRUE, console=FALSE)
                                        }}
                                        write.csv(test$groups, file="{csv_path}.groups.csv")
                                        cat(test$statistics$MSerror, "\\n", file="{csv_path}.mse.txt")
                                        cat(test$statistics$Df, "\\n", file="{csv_path}.df.txt")
                                        '''
                                        
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                                            f.write(r_script)
                                            r_script_path = f.name
                                        
                                        try:
                                            result = subprocess.run(['Rscript', r_script_path],
                                                                   capture_output=True, text=True, timeout=60)
                                            if result.returncode == 0:
                                                groups_df = pd.read_csv(f"{csv_path}.groups.csv", index_col=0)
                                                with open(f"{csv_path}.mse.txt") as f:
                                                    r_mse = float(f.read().strip())
                                                with open(f"{csv_path}.df.txt") as f:
                                                    r_df = int(f.read().strip())
                                                
                                                method_name = "Tukey's HSD" if posthoc_method=="Tukey's HSD" else "Fisher's LSD"
                                                posthoc_dict[resp] = {'method': method_name, 'critical_diff': np.nan, 'groups': {}}
                                                group_sd = resp_data.groupby('Treatment')[resp].std()
                                                
                                                for trt_name, row_data in groups_df.iterrows():
                                                    trt_name = str(trt_name)
                                                    posthoc_dict[resp]['groups'][trt_name] = {
                                                        'mean': float(row_data.iloc[0]),
                                                        'sd': float(group_sd.get(trt_name, 0)),
                                                        'letter': str(row_data.iloc[1]).strip()
                                                    }
                                                
                                                n_rep = len(resp_data)/max(len(resp_data['Treatment'].unique()), 1)
                                                if posthoc_method == "Tukey's HSD":
                                                    n_trt = len(resp_data['Treatment'].unique())
                                                    q_val = studentized_range.ppf(0.95, n_trt, r_df)
                                                    cd = q_val * np.sqrt(r_mse/n_rep)
                                                else:
                                                    t_val = t_dist.ppf(0.975, r_df)
                                                    cd = t_val * np.sqrt(2*r_mse/n_rep)
                                                posthoc_dict[resp]['critical_diff'] = cd
                                        finally:
                                            for path in [csv_path, r_script_path, f"{csv_path}.groups.csv",
                                                        f"{csv_path}.mse.txt", f"{csv_path}.df.txt"]:
                                                if os.path.exists(path): os.remove(path)

                                    else:
                            
                                        # Python fallback
                                        resp_data = df[['Treatment', resp]].dropna().copy()
                                        group_stats = resp_data.groupby('Treatment')[resp].agg(['mean', 'std', 'count'])
                                        group_means = group_stats['mean'].sort_values(ascending=False)
                                        group_sd = group_stats.loc[group_means.index, 'std']
                                        group_n = group_stats.loc[group_means.index, 'count']
                                        
                                        # Extract MSE and df directly from the factorial ANOVA (Stage 4)
                                        # This ensures the post-hoc error term is mathematically consistent.
                                        try:
                                            atbl = full_anova_tables[resp]
                                            mse = atbl.loc['Residual', 'sum_sq'] / atbl.loc['Residual', 'df']
                                            df_error = int(atbl.loc['Residual', 'df'])
                                        except Exception:
                                            # Fallback to one-way ANOVA only if the factorial table is unavailable
                                            model_1way = smf.ols(f'Q("{resp}") ~ C(Treatment)', data=resp_data).fit()
                                            anova_1way = sm.stats.anova_lm(model_1way, typ=2)
                                            mse = anova_1way.loc['Residual', 'sum_sq'] / anova_1way.loc['Residual', 'df']
                                            df_error = int(anova_1way.loc['Residual', 'df'])
                                        
                                        groups_list = list(group_means.index)
                                        n_grp = len(groups_list)
                                        means_array = group_means.values
                                        n_vals = group_n.values
                                        
                                        # Critical difference (average across pairs for display)
                                        cd_values = []
                                        for i in range(n_grp):
                                            for j in range(i+1, n_grp):
                                                se_diff = np.sqrt(mse * (1.0/n_vals[i] + 1.0/n_vals[j]))
                                                if posthoc_method == "Tukey's HSD":
                                                    q_crit = studentized_range.ppf(0.95, n_grp, df_error)
                                                    cd_values.append(q_crit * se_diff)
                                                else:
                                                    t_crit = t_dist.ppf(0.975, df_error)
                                                    cd_values.append(t_crit * se_diff)
                                        critical_diff = np.mean(cd_values) if cd_values else 0.0
                                        
                                        if posthoc_method == "Tukey's HSD":
                                            method_name = "Tukey's HSD"
                                        else:
                                            method_name = "Fisher's LSD"
                                        
                                        # Build p-value matrix
                                        p_matrix = np.ones((n_grp, n_grp))
                                        for i in range(n_grp):
                                            for j in range(n_grp):
                                                if i != j:
                                                    se_diff = np.sqrt(mse * (1.0/n_vals[i] + 1.0/n_vals[j]))
                                                    t_stat = abs(means_array[i] - means_array[j]) / se_diff
                                                    if posthoc_method == "Tukey's HSD":
                                                        q_stat = t_stat * np.sqrt(2)
                                                        p_matrix[i, j] = 1.0 - studentized_range.cdf(q_stat, n_grp, df_error)
                                                    else:
                                                        p_matrix[i, j] = 2.0 * (1.0 - t_dist.cdf(t_stat, df_error))
                                        
                                        # Store as dict with p-values and significance
                                        sig_results = []
                                        for i in range(n_grp):
                                            for j in range(i+1, n_grp):
                                                p_val = p_matrix[i, j]
                                                if p_val < 0.001:
                                                    sig_str = '***'
                                                elif p_val < 0.01:
                                                    sig_str = '**'
                                                elif p_val < 0.05:
                                                    sig_str = '*'
                                                else:
                                                    sig_str = 'ns'
                                                sig_results.append({
                                                    'Group_1': groups_list[i],
                                                    'Mean_1': round(means_array[i], 4),
                                                    'Group_2': groups_list[j],
                                                    'Mean_2': round(means_array[j], 4),
                                                    'Diff': round(abs(means_array[i] - means_array[j]), 4),
                                                    'p-value': round(p_val, 4),
                                                    'Sig.': sig_str
                                                })
                                        
                                        df_sig = pd.DataFrame(sig_results)
                                        
                                        # Generate native Python CLD
                                        cld_dict = generate_python_cld(groups_list, means_array, p_matrix)
                                        
                                        # Also build groups dict for display consistency
                                        sorted_groups = sorted(groups_list, key=lambda g: group_means[g], reverse=True)
                                        posthoc_dict[resp] = {
                                            'method': method_name,
                                            'critical_diff': critical_diff,
                                            'groups': {},
                                            'sig_matrix': df_sig
                                        }
                                        for g in sorted_groups:
                                            posthoc_dict[resp]['groups'][g] = {
                                                'mean': group_means[g],
                                                'sd': group_sd[g],
                                                'letter': cld_dict[g]
                                            }
                                except Exception as ex:
                                    posthoc_dict[resp] = {'method': posthoc_method, 'critical_diff': np.nan, 'groups': {}, 'error': str(ex)[:150]}                        
                        # ============================================================
                        # GENERATE ALL FIGURES
                        # ============================================================
                        progress_bar.progress(75, text="Generating publication-ready figures...")
                        
                        # Physical widths for journal column formats (in inches)
                        SINGLE_COL = 3.35   # ~85 mm — single-column journal width
                        DOUBLE_COL = 6.7    # ~170 mm — two-column (full page) width
                        DPI = 600           # High resolution for publication
                        TITLE_GAP = 1.07    # Vertical position of suptitle
                        
                        # Reference font sizes (calibrated on two-column 6.7" figures)
                        FONT_BASE_REF = 9
                        FONT_LABEL_REF = 10
                        FONT_TICK_REF = 8
                        FONT_TITLE_REF = 10
                        FONT_MATRIX_REF = 5.5
                        
                        def get_width(fig_key):
                            """Return figure width in inches based on user's layout choice."""
                            return DOUBLE_COL if fig_layout.get(fig_key, 'Two-Column') == 'Two-Column' else SINGLE_COL
                        
                        def get_fonts(fig_key):
                            """Return font sizes scaled to the selected figure width.
                            
                            Single-column figures (~3.35") get fonts scaled to ~55%
                            of the two-column size. Minimum floors prevent illegibility.
                            """
                            w = get_width(fig_key)
                            scale = w / DOUBLE_COL  # 1.0 for two-column, ~0.5 for single-column
                            # Apply a slightly more aggressive scaling curve
                            adjusted_scale = scale ** 1.15
                            return {
                                'base': max(5, FONT_BASE_REF * adjusted_scale),
                                'label': max(5.5, FONT_LABEL_REF * adjusted_scale),
                                'tick': max(5, FONT_TICK_REF * adjusted_scale),
                                'title': max(5.5, FONT_TITLE_REF * adjusted_scale),
                                'matrix': max(3.5, FONT_MATRIX_REF * adjusted_scale),
                            }
                        
                        plt.rcParams.update({
                            'font.size': FONT_BASE_REF, 'axes.titlesize': FONT_TITLE_REF,
                            'axes.labelsize': FONT_LABEL_REF, 'xtick.labelsize': FONT_TICK_REF,
                            'ytick.labelsize': FONT_TICK_REF, 'legend.fontsize': FONT_TICK_REF,
                            'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
                            'font.family': 'sans-serif'
                        })
                        
                        figure_buffers_png = {}
                        figure_buffers_pdf = {}
                        colors_list = plt.cm.Set2(np.linspace(0, 1, max(n_factors, 2)))
                        
                        # --- Fig 1: SHAP Importance ---
                        f1 = get_fonts('fig1')
                        fig1_w = get_width('fig1')
                        fig1, ax1 = plt.subplots(figsize=(fig1_w, fig1_w*0.55))
                        shap_norm = shap_imp/(shap_imp.sum(axis=1,keepdims=True)+1e-10)*100
                        x_pos = np.arange(n_responses)
                        width = 0.8/n_factors
                        for i in range(n_factors):
                            ax1.bar(x_pos+i*width, shap_norm[:,i], width, label=FACTOR_COLS[i],
                                    color=colors_list[i%len(colors_list)], edgecolor='black', linewidth=0.3)
                        ax1.set_xticks(x_pos+width*(n_factors-1)/2)
                        ax1.set_xticklabels(RESPONSE_COLS, rotation=30, ha='right')
                        ax1.set_ylabel('Relative Importance (%)', fontsize=f1['label'], fontweight='bold')
                        ax1.legend(fontsize=f1['tick'], framealpha=0.8)
                        ax1.grid(True, axis='y', alpha=0.3)
                        fig1.suptitle('SHAP Feature Importance Rankings', fontsize=f1['title'], fontweight='bold', y=TITLE_GAP)
                        fig1.tight_layout(rect=[0,0,1,0.95])
                        buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=DPI, bbox_inches='tight'); buf1.seek(0)
                        figure_buffers_png['Fig_1_SHAP_Importance.png'] = buf1
                        buf1p = io.BytesIO(); fig1.savefig(buf1p, format='pdf', bbox_inches='tight'); buf1p.seek(0)
                        figure_buffers_pdf['Fig_1_SHAP_Importance.pdf'] = buf1p
                        
                        # --- Fig 2: SHAP Beeswarm ---
                        f2 = get_fonts('fig2')
                        fig2_w = get_width('fig2')
                        n_cols_b = min(3, n_responses)
                        n_rows_b = (n_responses+n_cols_b-1)//n_cols_b
                        fig2, axes2 = plt.subplots(n_rows_b, n_cols_b, figsize=(fig2_w, fig2_w*0.35*n_rows_b))
                        if n_responses==1: axes2 = np.array([axes2])
                        axes2 = axes2.flatten()
                        for idx, resp in enumerate(RESPONSE_COLS):
                            ax = axes2[idx]
                            try:
                                rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=None,
                                                           min_samples_leaf=2, random_state=random_state, n_jobs=-1)
                                rf.fit(X_ml, Y_ml[:,idx])
                                shap_vals = shap.TreeExplainer(rf).shap_values(X_ml)
                                np.random.seed(42)
                                for fi in range(n_factors):
                                    x_shap = shap_vals[:,fi]
                                    y_jit = np.random.normal(fi, 0.08, len(x_shap))
                                    fv = X_ml[:,fi]; f_min, f_max = fv.min(), fv.max()
                                    fnorm = (fv-f_min)/(f_max-f_min) if f_max>f_min else np.zeros_like(fv)
                                    ax.scatter(x_shap, y_jit, c=plt.cm.RdBu_r(fnorm), s=6, alpha=0.7,
                                              edgecolors='black', linewidth=0.05)
                                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                                max_abs = max([abs(shap_vals[:,i]).max() for i in range(n_factors)])
                                ax.set_xlim(-max_abs*1.15, max_abs*1.15)
                            except:
                                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
                            ax.set_yticks(range(n_factors))
                            if idx//n_cols_b == n_rows_b-1: ax.set_xlabel('SHAP Value', fontsize=f2['label'], fontweight='bold')
                            if idx%n_cols_b==0: ax.set_yticklabels(FACTOR_COLS, fontsize=f2['tick'])
                            else: ax.set_yticklabels([])
                            ax.set_title(resp, fontsize=f2['title'], fontweight='bold')
                            for i in range(n_factors-1): ax.axhline(y=i+0.5, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
                        for idx in range(n_responses, len(axes2)): axes2[idx].set_visible(False)
                        fig2.suptitle('SHAP Beeswarm Plots\n(Blue=Low, Red=High)', fontsize=f2['title'], fontweight='bold', y=TITLE_GAP)
                        fig2.tight_layout(rect=[0,0,1,0.93])
                        buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=DPI, bbox_inches='tight'); buf2.seek(0)
                        figure_buffers_png['Fig_2_SHAP_Beeswarm.png'] = buf2
                        buf2p = io.BytesIO(); fig2.savefig(buf2p, format='pdf', bbox_inches='tight'); buf2p.seek(0)
                        figure_buffers_pdf['Fig_2_SHAP_Beeswarm.pdf'] = buf2p
                        
                        # --- Fig 3: GPR Contours ---
                        if n_factors>=2 and len(sig_int)>0:
                            f3 = get_fonts('fig3')
                            fig3_w = get_width('fig3')
                            n_sig = len(sig_int)
                            n_cols_g = min(3, n_sig)
                            n_rows_g = (n_sig+n_cols_g-1)//n_cols_g
                            fig3, axes3 = plt.subplots(n_rows_g, n_cols_g, figsize=(fig3_w, fig3_w*0.38*n_rows_g))
                            if n_sig==1: axes3 = np.array([axes3])
                            axes3 = axes3.flatten()
                            for idx, (_, row) in enumerate(sig_int.iterrows()):
                                ax = axes3[idx]
                                resp = row['Response']; f1 = row['Factor_1']; f2 = row['Factor_2']
                                Xp = df_encoded[[f1,f2]].values; yp = df[resp].dropna().values
                                if len(yp) < 6:
                                    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
                                    continue
                                scaler_X = StandardScaler(); Xp_s = scaler_X.fit_transform(Xp)
                                scaler_y = StandardScaler(); yp_s = scaler_y.fit_transform(yp.reshape(-1,1)).flatten()
                                kernel = ConstantKernel(1.0,(1e-3,1e3))*RBF([1.0,1.0],(0.5,10.0))+WhiteKernel(0.1,(1e-3,1.0))
                                try:
                                    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True, random_state=random_state)
                                    gpr.fit(Xp_s, yp_s)
                                    yp_pred_s = gpr.predict(Xp_s)
                                    ss_res = np.sum((yp_s-yp_pred_s)**2); ss_tot = np.sum((yp_s-yp_s.mean())**2)
                                    r2_train = 1-ss_res/ss_tot if ss_tot>0 else 0
                                    g1 = np.linspace(Xp_s[:,0].min()-0.5, Xp_s[:,0].max()+0.5, 80)
                                    g2 = np.linspace(Xp_s[:,1].min()-0.5, Xp_s[:,1].max()+0.5, 80)
                                    G1, G2 = np.meshgrid(g1, g2)
                                    Zp_s = gpr.predict(np.c_[G1.ravel(),G2.ravel()]).reshape(80,80)
                                    Zp = scaler_y.inverse_transform(Zp_s)
                                    grid_orig = scaler_X.inverse_transform(np.c_[G1.ravel(),G2.ravel()])
                                    G1o = grid_orig[:,0].reshape(80,80); G2o = grid_orig[:,1].reshape(80,80)
                                    cf = ax.contourf(G1o, G2o, Zp, levels=15, cmap='viridis', alpha=0.9)
                                    ax.contour(G1o, G2o, Zp, levels=6, colors='black', linewidths=0.3, alpha=0.5)
                                    ax.scatter(Xp[:,0], Xp[:,1], c='red', s=6, edgecolors='white', linewidth=0.2, alpha=0.8, zorder=5)
                                    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
                                    cbar.set_label(resp, fontsize=f3['tick']); cbar.ax.tick_params(labelsize=f3['tick']-1)
                                    ax.set_title(f'{resp}  |  Fit R² = {r2_train:.3f}', fontsize=f3['title']-1, fontweight='bold')
                                except:
                                    ax.text(0.5, 0.5, 'GPR failed', transform=ax.transAxes, ha='center')
                                ax.set_xlabel(f1, fontsize=f3['base'], fontweight='normal')
                                ax.set_ylabel(f2, fontsize=f3['base'], fontweight='normal')
                            for idx in range(n_sig, len(axes3)): axes3[idx].set_visible(False)
                            fig3.suptitle('GPR Contour Surfaces', fontsize=f3['title'], fontweight='bold', y=TITLE_GAP)
                            fig3.tight_layout(rect=[0,0,1,0.93])
                            buf3 = io.BytesIO(); fig3.savefig(buf3, format='png', dpi=DPI, bbox_inches='tight'); buf3.seek(0)
                            figure_buffers_png['Fig_3_GPR_Contours.png'] = buf3
                            buf3p = io.BytesIO(); fig3.savefig(buf3p, format='pdf', bbox_inches='tight'); buf3p.seek(0)
                            figure_buffers_pdf['Fig_3_GPR_Contours.pdf'] = buf3p
                        
                        # --- Fig 4: MORF Residuals ---
                        if n_responses >= 2:
                            f4 = get_fonts('fig4')
                            fig4_w = get_width('fig4')
                            matrix_size = fig4_w*(1.0 if fig4_w==SINGLE_COL else 0.65)
                            fig4, ax4 = plt.subplots(figsize=(matrix_size, matrix_size*1.05))
                            im4 = ax4.imshow(resid_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                            for i in range(n_responses):
                                for j in range(n_responses):
                                    if i!=j:
                                        tc = 'white' if abs(resid_corr[i,j])>0.5 else 'black'
                                        ax4.text(j, i, f'{resid_corr[i,j]:.2f}', ha='center', va='center',
                                                fontsize=f4['matrix'], color=tc, fontweight='bold')
                                    else:
                                        ax4.text(j, i, '1.00', ha='center', va='center', fontsize=f4['matrix'], color='gray')
                            ax4.set_xticks(range(n_responses)); ax4.set_yticks(range(n_responses))
                            short_names = [r[:10]+'…' if len(r)>10 else r for r in RESPONSE_COLS]
                            ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
                            ax4.set_yticklabels(short_names, fontsize=6)
                            cbar = plt.colorbar(im4, ax=ax4, shrink=0.85)
                            cbar.set_label('Pearson r', fontsize=6); cbar.ax.tick_params(labelsize=f4['matrix'])
                            fig4.suptitle('MORF Residual Correlation Matrix', fontsize=f4['title'], fontweight='bold', y=TITLE_GAP)
                            fig4.tight_layout(rect=[0,0,1,0.94])
                            buf4 = io.BytesIO(); fig4.savefig(buf4, format='png', dpi=DPI, bbox_inches='tight'); buf4.seek(0)
                            figure_buffers_png['Fig_4_MORF_Correlations.png'] = buf4
                            buf4p = io.BytesIO(); fig4.savefig(buf4p, format='pdf', bbox_inches='tight'); buf4p.seek(0)
                            figure_buffers_pdf['Fig_4_MORF_Correlations.pdf'] = buf4p
                        
                        # --- Fig 5: Standardized Beta ---
                        f5 = get_fonts('fig5')
                        fig5_w = get_width('fig5')
                        fig5, ax5 = plt.subplots(figsize=(fig5_w, fig5_w*0.45))
                        x_pos_b = np.arange(n_responses); width_b = 0.8/n_factors if n_factors>0 else 0.4
                        for i in range(n_factors):
                            means = beta[i,:]
                            ax5.bar(x_pos_b+i*width_b, means, width_b,
                                    yerr=[means-beta_ci_lower[i,:], beta_ci_upper[i,:]-means],
                                    label=FACTOR_COLS[i], capsize=2, edgecolor='black', linewidth=0.3)
                        ax5.axhline(y=0, color='black', linewidth=0.5)
                        ax5.set_xticks(x_pos_b+width_b*(n_factors-1)/2)
                        ax5.set_xticklabels(RESPONSE_COLS, rotation=30, ha='right')
                        ax5.set_ylabel('Standardized β Coefficient', fontsize=f5['label'], fontweight='bold')
                        ax5.legend(fontsize=f5['tick'], framealpha=0.8)
                        ax5.grid(True, axis='y', alpha=0.3)
                        fig5.suptitle('Standardized Regression Coefficients\nwith 95% Bootstrap CI', fontsize=f5['title'], fontweight='bold', y=TITLE_GAP)
                        fig5.tight_layout(rect=[0,0,1,0.95])
                        buf5 = io.BytesIO(); fig5.savefig(buf5, format='png', dpi=DPI, bbox_inches='tight'); buf5.seek(0)
                        figure_buffers_png['Fig_5_Standardized_Beta.png'] = buf5
                        buf5p = io.BytesIO(); fig5.savefig(buf5p, format='pdf', bbox_inches='tight'); buf5p.seek(0)
                        figure_buffers_pdf['Fig_5_Standardized_Beta.pdf'] = buf5p
                        
                        # --- Fig 6: Model Comparison ---
                        f6 = get_fonts('fig6')
                        fig6_w = get_width('fig6')
                        fig6, ax6 = plt.subplots(figsize=(fig6_w, fig6_w*0.40))
                        x_m = np.arange(n_responses); w_m = 0.25
                        model_colors = {'RF':'#2196F3', 'LR':'#FF9800', 'GPR':'#4CAF50'}
                        for i_mn, mn in enumerate(['RF','LR','GPR']):
                            m_vals = [model_comparison[j][f'{mn}_R2_mean'] for j in range(n_responses)]
                            s_vals = [model_comparison[j][f'{mn}_R2_std'] for j in range(n_responses)]
                            ax6.bar(x_m+i_mn*w_m, m_vals, w_m, yerr=s_vals, label=mn,
                                    color=model_colors[mn], edgecolor='black', linewidth=0.3, capsize=2)
                        ax6.set_xticks(x_m+w_m); ax6.set_xticklabels(RESPONSE_COLS, rotation=30, ha='right')
                        ax6.set_ylabel('Cross-Validated R²', fontsize=f6['label'], fontweight='bold')
                        ax6.legend(fontsize=f6['tick'], framealpha=0.8)
                        ax6.grid(True, axis='y', alpha=0.3); ax6.set_ylim(0, 1.1)
                        fig6.suptitle(f'Model Performance Comparison ({cv_folds}-Fold CV R²)', fontsize=f6['title'], fontweight='bold', y=TITLE_GAP)
                        fig6.tight_layout(rect=[0,0,1,0.95])
                        buf6 = io.BytesIO(); fig6.savefig(buf6, format='png', dpi=DPI, bbox_inches='tight'); buf6.seek(0)
                        figure_buffers_png['Fig_6_Model_Comparison.png'] = buf6
                        buf6p = io.BytesIO(); fig6.savefig(buf6p, format='pdf', bbox_inches='tight'); buf6p.seek(0)
                        figure_buffers_pdf['Fig_6_Model_Comparison.pdf'] = buf6p
                        
                        # --- Fig S1: Permutation ---
                        if n_factors>=2:
                            fS1 = get_fonts('figS1')
                            figS1_w = get_width('figS1')
                            figS1, axS1 = plt.subplots(figsize=(figS1_w, figS1_w*0.55))
                            perm_norm = perm_imp/(perm_imp.sum(axis=1,keepdims=True)+1e-10)*100
                            x_pp = np.arange(n_responses); w_pp = 0.8/n_factors
                            for i in range(n_factors):
                                axS1.bar(x_pp+i*w_pp, perm_norm[:,i], w_pp, label=FACTOR_COLS[i],
                                        color=colors_list[i%len(colors_list)], edgecolor='black', linewidth=0.3,
                                        yerr=perm_imp_std[:,i]/(perm_imp.sum(axis=1)+1e-10)*100, capsize=2)
                            axS1.set_xticks(x_pp+w_pp*(n_factors-1)/2)
                            axS1.set_xticklabels(RESPONSE_COLS, rotation=30, ha='right')
                            axS1.set_ylabel('Relative Importance (%)', fontsize=fS1['label'], fontweight='bold')
                            axS1.legend(fontsize=fS1['tick'], framealpha=0.8)
                            axS1.grid(True, axis='y', alpha=0.3)
                            figS1.suptitle('Permutation Feature Importance', fontsize=fS1['title'], fontweight='bold', y=TITLE_GAP)
                            figS1.tight_layout(rect=[0,0,1,0.95])
                            bufS1 = io.BytesIO(); figS1.savefig(bufS1, format='png', dpi=DPI, bbox_inches='tight'); bufS1.seek(0)
                            figure_buffers_png['Fig_S1_Permutation_Importance.png'] = bufS1
                            bufS1p = io.BytesIO(); figS1.savefig(bufS1p, format='pdf', bbox_inches='tight'); bufS1p.seek(0)
                            figure_buffers_pdf['Fig_S1_Permutation_Importance.pdf'] = bufS1p
                        
                        # --- Fig 7: PCA Biplot ---
                        if n_responses >= 2:
                            f7 = get_fonts('fig7')
                            fig7_w = get_width('fig7')
                            fig7_w_actual = SINGLE_COL*1.15 if fig7_w==SINGLE_COL else fig7_w*0.80
                            fig7, ax7 = plt.subplots(figsize=(fig7_w_actual, fig7_w_actual*1.20))
                            
                            try:
                                df['_pca_group_'] = df[FACTOR_COLS[0]].astype(str)+'_'+df[FACTOR_COLS[1]].astype(str)
                            except:
                                df['_pca_group_'] = df[FACTOR_COLS[0]].astype(str)
                            unique_groups = sorted(df['_pca_group_'].unique())
                            n_unique = len(unique_groups)
                            colors_pca = plt.cm.tab20(np.linspace(0,1,n_unique))
                            df_ml_idx = df_ml.index
                            for i, grp in enumerate(unique_groups):
                                mask = df.loc[df_ml_idx, '_pca_group_']==grp
                                if mask.sum()>0:
                                    ax7.scatter(pca_scores[mask.values,0], pca_scores[mask.values,1], c=[colors_pca[i]],
                                               s=30, edgecolors='black', linewidth=0.3, label=grp, zorder=5)
                            scale_factor = max(abs(pca_scores[:,0]).max(), abs(pca_scores[:,1]).max())*0.75
                            label_positions = []
                            for i, resp in enumerate(RESPONSE_COLS):
                                lx = pca_loadings[i,0]*scale_factor; ly = pca_loadings[i,1]*scale_factor
                                ax7.arrow(0,0,lx,ly, head_width=scale_factor*0.03, head_length=scale_factor*0.06,
                                         fc='red', ec='red', alpha=0.7, linewidth=0.6)
                                text_x, text_y = lx*1.22, ly*1.22
                                for prev_x, prev_y in label_positions:
                                    if np.sqrt((text_x-prev_x)**2+(text_y-prev_y)**2) < scale_factor*0.15:
                                        text_x, text_y = lx*1.40, ly*1.40; break
                                label_positions.append((text_x, text_y))
                                ax7.text(text_x, text_y, resp, fontsize=6, color='darkred', fontweight='bold', ha='center', va='center')
                            ax7.set_xlabel(f'PC1 ({pca_var_explained[0]:.1%} variance)', fontsize=f7['base'], fontweight='normal')
                            ax7.set_ylabel(f'PC2 ({pca_var_explained[1]:.1%} variance)', fontsize=f7['base'], fontweight='normal')
                            ax7.axhline(y=0, color='gray', linewidth=0.5); ax7.axvline(x=0, color='gray', linewidth=0.5)
                            ax7.grid(True, alpha=0.3)
                            ncol_legend = min(5, n_unique)
                            ax7.legend(fontsize=5, ncol=ncol_legend, loc='lower left',
                                       bbox_to_anchor=(0,1.01,1,0.12), framealpha=0.8,
                                       mode='expand', borderaxespad=0, title='Groups', title_fontsize=5.5,
                                       handletextpad=0.3, columnspacing=0.5)
                            fig7.suptitle('PCA Biplot — Treatment Groups & Response Loadings',
                                         fontsize=f7['title'], fontweight='bold', y=TITLE_GAP+0.06)
                            fig7.tight_layout(rect=[0,0,1,0.88])
                            buf7 = io.BytesIO(); fig7.savefig(buf7, format='png', dpi=DPI, bbox_inches='tight'); buf7.seek(0)
                            figure_buffers_png['Fig_7_PCA_Biplot.png'] = buf7
                            buf7p = io.BytesIO(); fig7.savefig(buf7p, format='pdf', bbox_inches='tight'); buf7p.seek(0)
                            figure_buffers_pdf['Fig_7_PCA_Biplot.pdf'] = buf7p
                            df.drop(columns=['_pca_group_'], inplace=True)
                            
                            plt.close('all')
                            figure_buffers = {**figure_buffers_png, **figure_buffers_pdf}
                            progress_bar.progress(90, text="Preparing display...")                        
                        # ============================================================
                        # DISPLAY RESULTS
                        # ============================================================
                        st.header("📊 Results")
                        
                        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📋 Preliminaries", "📈 ANOVA & Interactions", "🔤 Post-Hoc",
                            "🤖 ML Results", "📊 Figures", "📥 Download"
                        ])
                        
                        with tab0:
                            st.subheader("Statistical Assumption Checks")
                            st.dataframe(df_prelim, use_container_width=True)
                            non_normal = [r['Response'] for _,r in df_prelim.iterrows() if 'Non-normal' in str(r['Residuals'])]
                            unequal_var = [r['Response'] for _,r in df_prelim.iterrows() if r['Variances']=='Unequal']
                            if non_normal: st.warning(f"⚠️ Non-normal distribution: {', '.join(non_normal)}")
                            if unequal_var: st.warning(f"⚠️ Unequal variances: {', '.join(unequal_var)}")
                            if not non_normal and not unequal_var: st.success("✅ All responses meet assumptions.")
                            st.markdown("---")
                            st.markdown("#### Outlier Detection")
                            any_outliers = False
                            for resp in RESPONSE_COLS:
                                if outlier_results[resp]['n_outliers']>0:
                                    any_outliers = True
                                    with st.expander(f"⚠️ {resp}: {outlier_results[resp]['n_outliers']} outlier(s)"):
                                        st.dataframe(pd.DataFrame(outlier_results[resp]['details']), use_container_width=True)
                            if not any_outliers: st.success("✅ No outliers detected.")
                            st.markdown("---")
                            st.markdown("#### Pearson Correlations")
                            st.dataframe(df_pearson, use_container_width=True)
                        
                        with tab1:
                            if len(sig_int)>0:
                                st.subheader("Significant Interactions")
                                st.dataframe(sig_int[['Response','Factor_1','Factor_2','F_int','p_int','Sig.']].round(4), use_container_width=True)
                            else: st.info("No significant interactions found.")
                            st.subheader("Full ANOVA Tables")
                            for resp in RESPONSE_COLS:
                                if resp in full_anova_tables:
                                    with st.expander(f"ANOVA: {resp}"):
                                        st.dataframe(full_anova_tables[resp].round(4), use_container_width=True)
                            st.markdown("---")
                            st.subheader("ART ANOVA (aligned rank transform)")
                            if len(df_art) > 0:
                                st.dataframe(df_art.round(4), use_container_width=True)
                            else:
                                st.info("ART ANOVA not available.")
                            st.markdown("---")
                            st.subheader("Kruskal-Wallis Validation")
                            if len(df_kw)>0: st.dataframe(df_kw, use_container_width=True)
                            else: st.info("Not available.")
                        
                        with tab2:
                            if posthoc_dict:
                                method_used = list(posthoc_dict.values())[0]['method']
                                st.subheader(f"Mean Comparisons ({method_used})")
                                for resp in RESPONSE_COLS:
                                    if resp in posthoc_dict and posthoc_dict[resp]['groups']:
                                        cd = posthoc_dict[resp]['critical_diff']
                                        # Check if we have a significance matrix (Python fallback)
                                    if resp in posthoc_dict and posthoc_dict[resp]['groups']:
                                        cd = posthoc_dict[resp]['critical_diff']
                                        with st.expander(f"{resp} (CD = {cd:.4f})"):
                                            st.markdown("**Treatment Means**")
                                            rows = [{'Treatment':t, 'Mean':f"{i['mean']:.4f}",
                                                    'SD':f"{i['sd']:.4f}", 'Group':i['letter']}
                                                   for t,i in posthoc_dict[resp]['groups'].items()]
                                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                                            
                                            # Show the pairwise matrix ONLY if the Python fallback was used
                                            if 'sig_matrix' in posthoc_dict[resp]:
                                                st.markdown("**Pairwise Significance Matrix**")
                                                st.dataframe(posthoc_dict[resp]['sig_matrix'], use_container_width=True)
                            else: st.info("Post-hoc comparisons not available.")
                        
                        with tab3:
                            st.subheader("SHAP Feature Importance (%)")
                            shap_pct_df = pd.DataFrame(shap_imp/(shap_imp.sum(axis=1,keepdims=True)+1e-10)*100,
                                                       columns=FACTOR_COLS, index=RESPONSE_COLS).round(1)
                            st.dataframe(shap_pct_df, use_container_width=True)
                            st.subheader(f"Model Comparison ({cv_folds}-Fold CV R²)")

                            # Display performance metrics
                            perf_cols = ['Response', 'RF_R2_mean', 'RF_R2_std', 'LR_R2_mean', 'LR_R2_std', 
                                        'GPR_R2_mean', 'GPR_R2_std',
                                        'RF_R2_group', 'LR_R2_group', 'GPR_R2_group']
                            st.dataframe(df_mc[perf_cols].round(3), use_container_width=True)
                            
                            # Display pairwise statistical tests
                            st.markdown("**Pairwise Model Comparisons (Paired t-test on CV folds)**")
                            test_cols = ['Response', 'RF_vs_LR', 'RF_vs_GPR', 'LR_vs_GPR',
                                         'RF_vs_LR_NB', 'RF_vs_GPR_NB', 'LR_vs_GPR_NB',
                                         'RF_vs_LR_group', 'RF_vs_GPR_group', 'LR_vs_GPR_group',
                                         'RF_vs_LR_group_NB', 'RF_vs_GPR_group_NB', 'LR_vs_GPR_group_NB']
                            st.dataframe(df_mc[test_cols], use_container_width=True)
                            st.caption("* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
                            st.subheader("Standardized Regression Coefficients (β)")
                            beta_display = pd.DataFrame(beta, index=FACTOR_COLS, columns=RESPONSE_COLS).round(4)
                            st.dataframe(beta_display, use_container_width=True)
                            st.markdown("---")
                            st.subheader("PCA")
                            st.dataframe(df_pca_summary, use_container_width=True)
                            st.dataframe(df_pca_loadings, use_container_width=True)
                        
                        with tab4:
                            st.subheader("📊 All Figures")
                            fig_display_list = [
                                ('Fig 1: SHAP Importance', 'Fig_1_SHAP_Importance.png'),
                                ('Fig 2: SHAP Beeswarm', 'Fig_2_SHAP_Beeswarm.png'),
                                ('Fig 3: GPR Contours', 'Fig_3_GPR_Contours.png'),
                                ('Fig 4: MORF Residuals', 'Fig_4_MORF_Correlations.png') if n_responses >= 2 else None,
                                ('Fig 5: Standardized Beta', 'Fig_5_Standardized_Beta.png'),
                                ('Fig 6: Model Comparison', 'Fig_6_Model_Comparison.png'),
                                ('Fig 7: PCA Biplot', 'Fig_7_PCA_Biplot.png') if n_responses >= 2 else None,
                                ('Fig S1: Permutation Importance', 'Fig_S1_Permutation_Importance.png'),
                            ]
                            for fig_title, fig_key in [f for f in fig_display_list if f is not None]:
                                if fig_key in figure_buffers_png:
                                    with st.expander(fig_title, expanded=False):
                                        st.image(figure_buffers_png[fig_key], use_container_width=True)
                                elif fig_key=='Fig_3_GPR_Contours.png':
                                    with st.expander(fig_title): st.info("No significant interactions.")
                                elif fig_key == 'Fig_S1_Permutation_Importance.png':
                                    if n_factors < 2:
                                        with st.expander(fig_title):
                                            st.info("Requires ≥2 factors.")
                        
                        with tab5:
                            st.subheader("📥 Download All Results")
                            
                            text_report = generate_text_report(
                                df, FACTOR_COLS, RESPONSE_COLS, n_factors, n_responses,
                                df_prelim, df_pearson, outlier_results,
                                df_int, full_anova_tables, df_kw,
                                posthoc_dict, posthoc_method,
                                shap_imp, perm_imp, model_comparison, df_mc,
                                beta, beta_ci_lower, beta_ci_upper,
                                pca, pca_var_explained, df_pca_summary, df_pca_loadings,
                                resid_corr, pca_loadings, cv_folds,
                                factor_type_log=factor_type_log,
                                source_file_name=uploaded_file.name,
                                n_rows_raw=len(df),
                                n_rows_anova=len(df.dropna(subset=RESPONSE_COLS)),
                                n_rows_ml=len(df_ml),
                                n_rows_dropped=len(df) - len(df_ml),
                                df_anova_fdr=df_anova_fdr,
                                df_art=df_art,
                                df_simple=df_simple,
                                stability_results=stability_results,
                                shap_null_results=shap_null_results,
                                resid_corr_oof=resid_corr_oof
                            )
                            
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                zf.writestr('Tables/Preliminaries.csv', df_prelim.to_csv(index=False))
                                zf.writestr('Tables/Pearson_correlations.csv', df_pearson.to_csv(index=False))
                                zf.writestr('Tables/ANOVA_interactions.csv', df_int.to_csv(index=False))
                                zf.writestr('Tables/Kruskal_Wallis_validation.csv', df_kw.to_csv(index=False))
                                zf.writestr('Tables/Model_comparison.csv', df_mc.to_csv(index=False))
                                zf.writestr('Tables/SHAP_importance_pct.csv', shap_pct_df.to_csv())
                                zf.writestr('Tables/Standardized_beta.csv', beta_display.to_csv())
                                zf.writestr('Tables/PCA_summary.csv', df_pca_summary.to_csv(index=False))
                                zf.writestr('Tables/PCA_loadings.csv', df_pca_loadings.to_csv(index=False))
                                zf.writestr('Tables/Permutation_importance.csv',
                                           pd.DataFrame(perm_imp, columns=FACTOR_COLS, index=RESPONSE_COLS).to_csv())
                                zf.writestr('analysis_report.txt', text_report)
                                for fn, fb in figure_buffers_png.items():
                                    fb.seek(0); zf.writestr(f'Figures/PNG/{fn}', fb.read())
                                for fn, fb in figure_buffers_pdf.items():
                                    fb.seek(0); zf.writestr(f'Figures/PDF/{fn}', fb.read())
                            zip_buffer.seek(0)
                            st.download_button("📥 Download Complete Results (ZIP)", data=zip_buffer,
                                             file_name="analysis_results.zip", mime="application/zip", use_container_width=True)
                            
                            st.markdown("---")
                            st.download_button("📄 Download Text Report (TXT)", data=text_report,
                                             file_name="analysis_report.txt", mime="text/plain", use_container_width=True)
                            
                            st.markdown("---")
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                png_zip = io.BytesIO()
                                with zipfile.ZipFile(png_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                                    for fn, fb in figure_buffers_png.items():
                                        fb.seek(0); zf.writestr(fn, fb.read())
                                png_zip.seek(0)
                                st.download_button("📥 All PNG Figures (600 DPI)", data=png_zip,
                                                 file_name="figures_png.zip", mime="application/zip", use_container_width=True)
                            with col_dl2:
                                pdf_zip = io.BytesIO()
                                with zipfile.ZipFile(pdf_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                                    for fn, fb in figure_buffers_pdf.items():
                                        fb.seek(0); zf.writestr(fn, fb.read())
                                pdf_zip.seek(0)
                                st.download_button("📥 All PDF Figures (Vector)", data=pdf_zip,
                                                 file_name="figures_pdf.zip", mime="application/zip", use_container_width=True)
                        
                        progress_bar.progress(100, text="Analysis complete!")
                        st.success("✅ Analysis complete! All results and figures are ready.")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    else:
        st.info("👈 Please upload a CSV file in the sidebar to begin.")
        st.markdown("""
        ### Expected CSV Format
        Your CSV file should have **factor columns** (treatment variables) and **response columns** (measured traits).
        """)
