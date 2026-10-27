import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.metrics import r2_score

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
print('statsmodel version:', statsmodels.__version__)

import scipy
from scipy.stats import spearmanr
print('scipy version:', scipy.__version__)

# Spearman Correlation analysis
def spearman_filter(df_collapsed, im_cols, target_col, rho_threshold=0.3):
    """
    Compute Spearman correlation between each IM and ethanol
    on condition-collapsed data (one row per condition).
 
    Parameters
    ----------
    df_collapsed : pd.DataFrame
        Condition-averaged data (one row per condition).
    im_cols : list of str
        IMs to test (should be the FDR-significant IMs from MixedLM).
    target_col : str
        Ethanol phenotype column.
    rho_threshold : float
        Minimum |rho| to pass effect size filter (default 0.3).
 
    Returns
    -------
    spearman_df : pd.DataFrame
        Per-IM Spearman rho, p-value, FDR q-value, pass/fail.
    passing_ims : list
        IMs passing |rho| > rho_threshold and FDR < 0.05.
    """
    rows = []
    for im in im_cols:
        rho, pval = spearmanr(
            df_collapsed[im],
            df_collapsed[target_col]
        )
        rows.append({"IM": im, "spearman_rho": rho, "pval": pval})
 
    spearman_df = pd.DataFrame(rows)
 
    # FDR correct
    _, padj, _, _ = multipletests(spearman_df["pval"], method="fdr_bh")
    spearman_df["qval"] = padj
    spearman_df["strong_effect"] = spearman_df["spearman_rho"].abs() >= rho_threshold
    spearman_df["passes_strong_effect"] = (
        spearman_df["strong_effect"] & (spearman_df["pval"] < 0.05)
    )
    spearman_df["passes_all"] = (
        spearman_df["strong_effect"] & (spearman_df["qval"] < 0.05)
    )
    spearman_df = spearman_df.sort_values("spearman_rho", key=abs, ascending=False)
 
    passing_ims = spearman_df.loc[spearman_df["passes_strong_effect"], "IM"].tolist()
    passing_ims_fdr = spearman_df.loc[spearman_df["passes_all"], "IM"].tolist()
 
    print("\n" + "=" * 60)
    print(f"Spearman Effect Size Filter (|rho| > {rho_threshold})")
    print("=" * 60)
    print(spearman_df.to_string(index=False))
    print(f"\nIMs passing effect size filter: {passing_ims}")
 
    return spearman_df, passing_ims, passing_ims_fdr


# ── LOO-CV for OLS ─────────────────────────────────────────────────────────────

def loo_cv_ols(df, feature_cols, target_col):
    """
    Standard LOO-CV for OLS on condition-collapsed data.

    Parameters
    ----------
    df           : pd.DataFrame — condition-collapsed data (no reference rows)
    feature_cols : list of str  — IM columns to use as predictors
    target_col   : str          — ethanol phenotype column

    Returns
    -------
    loo_r2 : float — LOO-CV R²
    """
    y_true_all = []
    y_pred_all = []

    for idx in df.index:
        train = df.drop(index=idx)
        test  = df.loc[[idx]]

        X_train = sm.add_constant(train[feature_cols], has_constant='add')
        y_train = train[target_col]
        X_test  = sm.add_constant(test[feature_cols],  has_constant='add')

        model   = sm.OLS(y_train, X_train).fit()
        y_pred  = model.predict(X_test).values[0]

        y_true_all.append(df.loc[idx, target_col])
        y_pred_all.append(y_pred)

    return r2_score(y_true_all, y_pred_all)


# ── VIF calculation ────────────────────────────────────────────────────────────

def compute_vif(df, feature_cols):
    """
    Compute Variance Inflation Factor for each predictor.
    Returns max VIF across all predictors in the model.
    """

    if len(feature_cols) == 1:
        return np.nan   # VIF undefined for single predictor

    X = sm.add_constant(df[feature_cols], has_constant='add')
    vifs = [
        variance_inflation_factor(X.values, i + 1)
        for i in range(len(feature_cols))
    ]
    return max(vifs)


# ── Main model comparison ──────────────────────────────────────────────────────

def compare_ols_combinations(df, im_cols, target_col, loo=True):
    """
    Fit OLS for every possible combination of IMs (size 1 to len(im_cols))
    and compare by AIC, R²_adj, and LOO-CV R².

    Parameters
    ----------
    df         : pd.DataFrame — condition-collapsed data, reference rows removed
    im_cols    : list of str  — candidate IM columns (up to 4)
    target_col : str          — ethanol phenotype column
    loo        : bool         — whether to compute LOO-CV R² (default True)

    Returns
    -------
    summary : pd.DataFrame
        All models ranked by AIC with R²_adj, LOO-CV R², gap, VIF.
    best_model : dict
        The winning model by AIC with its details.
    """
    rows = []
    n    = len(df)

    # Generate all combinations of size 1 to len(im_cols)
    all_combos = []
    for r in range(1, len(im_cols) + 1):
        for combo in combinations(im_cols, r):
            all_combos.append(list(combo))

    print(f"Testing {len(all_combos)} model combinations...")
    print(f"Data: {n} conditions, {len(im_cols)} candidate IMs\n")

    for feature_cols in all_combos:
        try:
            # Fit OLS
            X     = sm.add_constant(df[feature_cols], has_constant='add')
            y     = df[target_col]
            model = sm.OLS(y, X).fit()

            r2_adj = model.rsquared_adj
            aic    = model.aic
            fpval  = model.f_pvalue

            # LOO-CV R²
            loo_r2 = loo_cv_ols(df, feature_cols, target_col) if loo else np.nan
            gap    = r2_adj - loo_r2 if loo else np.nan

            # VIF
            max_vif = compute_vif(df, feature_cols)

            # Individual IM significance in this model
            pvals = {
                im: round(model.pvalues[im], 4)
                for im in feature_cols
                if im in model.pvalues.index
            }
            n_sig = sum(p < 0.05 for p in pvals.values())

            rows.append({
                "model":       " + ".join(feature_cols),
                "n_ims":       len(feature_cols),
                "IMs":         feature_cols,
                "R2_adj":      round(r2_adj, 4),
                "LOO_CV_R2":   round(loo_r2, 4) if loo else np.nan,
                "gap":         round(gap, 4)     if loo else np.nan,
                "AIC":         round(aic, 3),
                "F_pval":      round(fpval, 6),
                "max_VIF":     round(max_vif, 2) if not np.isnan(max_vif) else np.nan,
                "n_sig_ims":   n_sig,
                "im_pvals":    pvals
            })

        except Exception as e:
            rows.append({
                "model":     " + ".join(feature_cols),
                "n_ims":     len(feature_cols),
                "IMs":       feature_cols,
                "R2_adj":    np.nan,
                "LOO_CV_R2": np.nan,
                "gap":       np.nan,
                "AIC":       np.nan,
                "F_pval":    np.nan,
                "max_VIF":   np.nan,
                "n_sig_ims": 0,
                "im_pvals":  {}
            })

    summary = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

    # ── Print results ──────────────────────────────────────────────────────────
    print_cols = ["model", "n_ims", "R2_adj", "LOO_CV_R2", "gap", "AIC",
                  "F_pval", "max_VIF", "n_sig_ims"]

    print("=" * 90)
    print("OLS Model Comparison — All Combinations (sorted by AIC)")
    print("=" * 90)
    print(summary[print_cols].to_string(index=True))

    # ── Best model ─────────────────────────────────────────────────────────────
    best = summary.iloc[0]
    print("\n" + "=" * 90)
    print("BEST MODEL (lowest AIC)")
    print("=" * 90)
    print(f"  Model      : {best['model']}")
    print(f"  N IMs      : {best['n_ims']}")
    print(f"  R²_adj     : {best['R2_adj']}")
    print(f"  LOO-CV R²  : {best['LOO_CV_R2']}")
    print(f"  Gap        : {best['gap']}")
    print(f"  AIC        : {best['AIC']}")
    print(f"  Max VIF    : {best['max_VIF']}")
    print(f"  N sig IMs  : {best['n_sig_ims']}")
    print(f"  IM p-values: {best['im_pvals']}")

    # ── AIC comparison relative to best ───────────────────────────────────────
    summary["delta_AIC"] = summary["AIC"] - best["AIC"]
    print("\n" + "=" * 90)
    print("Delta AIC relative to best model")
    print("=" * 90)
    print(summary[["model", "n_ims", "R2_adj", "AIC", "delta_AIC",
                    "n_sig_ims"]].to_string(index=True))

    # ── Summary by model size ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Best model per size (lowest AIC within each n_ims)")
    print("=" * 90)
    best_per_size = (
        summary.groupby("n_ims")
        .apply(lambda x: x.nsmallest(1, "AIC"))
        .reset_index(drop=True)
    )
    print(best_per_size[["n_ims", "model", "R2_adj",
                          "LOO_CV_R2", "AIC", "delta_AIC"]].to_string(index=False))

    best_model = {
        "model":      best["model"],
        "IMs":        best["IMs"],
        "R2_adj":     best["R2_adj"],
        "LOO_CV_R2":  best["LOO_CV_R2"],
        "gap":        best["gap"],
        "AIC":        best["AIC"],
        "im_pvals":   best["im_pvals"]
    }

    return summary, best_model


# ── Refit and print full summary of best model ─────────────────────────────────

def refit_best_model(df, best_model, target_col):
    """
    Refit OLS on the best model and print full statsmodels summary.

    Parameters
    ----------
    df         : pd.DataFrame — condition-collapsed data, reference rows removed
    best_model : dict         — output from compare_ols_combinations()
    target_col : str          — ethanol phenotype column
    """
    feature_cols = best_model["IMs"]
    X     = sm.add_constant(df[feature_cols], has_constant='add')
    y     = df[target_col]
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 90)
    print(f"Full OLS Summary — Best Model: {best_model['model']}")
    print("=" * 90)
    print(model.summary())

    return model