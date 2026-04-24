"""
Linear Mixed Model Analysis Engine.
=====================================

Implements the statistical model from the project proposal:

    y ~ beta0 + beta1*T + beta2*L + (1|Language) + epsilon

where:
    y         = BERT attention weight (parent-child token pair)
    T         = gold dependency type (e.g. nsubj, obj, amod ...)
    L         = sentence length (confounder)
    (1|Language) = random intercept per language (controls cross-lingual variation)
    epsilon   = residual error

Two global models are fit:
    Model 1: y ~ C(T) + L + (1|Language)           -- proposal formula
    Model 2: y ~ C(T) + L + head + layer + (1|Language)  -- extended model

Additionally, per-language OLS models are fit for each language:
    model1_en / model1_hi / model1_ja / model1_de :
        y ~ C(T) + L   (single language, so random intercept is not needed)

Outputs:
    lmm_results.json             -- coefficients, CI, R2, random effects (web UI)
    graphs/lmm_forest_plot.png              -- global fixed effects forest plot
    graphs/lmm_random_effects.png           -- per-language random intercept chart
    graphs/lmm_forest_plot_en.png           -- English forest plot
    graphs/lmm_forest_plot_hi.png           -- Hindi forest plot
    graphs/lmm_forest_plot_ja.png           -- Japanese forest plot
    graphs/lmm_forest_plot_de.png           -- German forest plot
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
import json
import os

BACKEND_DIR = os.path.dirname(__file__)

# Map numeric group IDs back to language names
LE_LANG_MAP = {0: "English", 1: "German", 2: "Hindi", 3: "Japanese"}

# Per-language configs:  (display name, lang code, language string in CSV)
LANG_CONFIGS = [
    ("English",  "en", "English"),
    ("Hindi",    "hi", "Hindi"),
    ("Japanese", "ja", "Japanese"),
    ("German",   "de", "German"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _display(name):
    """Strip statsmodels C(relation)[T.xxx] wrapper to a clean relation name."""
    if name.startswith("C(relation)[T."):
        return name.replace("C(relation)[T.", "").replace("]", "")
    return name


def extract_stats(fit):
    """Extract fixed-effect coefficients, CIs, p-values and R2 from a fitted MixedLM."""
    params  = fit.params
    pvalues = fit.pvalues
    bse     = fit.bse
    conf    = fit.conf_int()

    coefs = []
    for name in params.index:
        if name == "Group Var":
            continue
        coefs.append({
            "name":         name,
            "display_name": _display(name),
            "coef":     round(float(params[name]), 4),
            "pvalue":   float(pvalues[name]),
            "se":       round(float(bse[name]), 4),
            "ci_low":   round(float(conf.loc[name, 0]), 4),
            "ci_high":  round(float(conf.loc[name, 1]), 4)
        })

    # Nakagawa & Schielzeth (2013) marginal/conditional R2
    fe_pred    = np.dot(fit.model.exog, fit.fe_params)
    var_fixed  = np.var(fe_pred)
    var_random = float(fit.cov_re.iloc[0, 0])
    var_resid  = float(fit.scale)
    total      = var_fixed + var_random + var_resid

    return {
        "coefficients":   sorted(coefs, key=lambda x: x["coef"], reverse=True),
        "r2_marginal":    round(float(var_fixed / total), 4),
        "r2_conditional": round(float((var_fixed + var_random) / total), 4),
        "n_obs":          int(len(fit.model.endog))
    }


def extract_stats_ols(fit):
    """Extract coefficients, CIs, p-values and R2 from a fitted OLS model (per-language)."""
    params  = fit.params
    pvalues = fit.pvalues
    bse     = fit.bse
    conf    = fit.conf_int()

    coefs = []
    for name in params.index:
        coefs.append({
            "name":         name,
            "display_name": _display(name),
            "coef":     round(float(params[name]), 4),
            "pvalue":   float(pvalues[name]),
            "se":       round(float(bse[name]), 4),
            "ci_low":   round(float(conf.loc[name, 0]), 4),
            "ci_high":  round(float(conf.loc[name, 1]), 4)
        })

    return {
        "coefficients":   sorted(coefs, key=lambda x: x["coef"], reverse=True),
        "r2_marginal":    round(float(fit.rsquared), 4),
        "r2_conditional": round(float(fit.rsquared_adj), 4),
        "n_obs":          int(fit.nobs)
    }


def extract_random_effects(fit, le_map):
    """
    Extract (1|Language) random intercepts from the fitted LMM.
    Returns a list of {language, intercept} sorted by intercept value.
    """
    re = fit.random_effects
    out = []
    for group_id, vals in re.items():
        lang_name = le_map.get(int(group_id), f"Group {group_id}")
        try:
            intercept = float(vals.iloc[0])
        except AttributeError:
            intercept = float(vals[0])
        out.append({"language": lang_name, "intercept": round(intercept, 6)})
    return sorted(out, key=lambda x: x["intercept"], reverse=True)


# ── Graph helpers ──────────────────────────────────────────────────────────────

def _forest_plot(plot_df, title, filepath):
    """Save a forest-plot PNG for a given coefficient DataFrame."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.95)

    fig, ax = plt.subplots(figsize=(10, max(8, len(plot_df) * 0.32 + 1)))
    for i, row in enumerate(plot_df.itertuples()):
        color = "#10b981" if row.coef >= 0 else "#e11d48"
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, alpha=0.6, linewidth=2)
        ax.scatter(row.coef, i, color=color, s=50,
                   edgecolors="white", zorder=3, linewidths=0.8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.display_name, fontsize=8)
    ax.set_xlabel("Coefficient beta1 (95% CI)\nRelative to baseline relation (acl)")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(filepath)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    csv_path = os.path.join(BACKEND_DIR, "data", "Word.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    le_language = LabelEncoder()
    df["language_enc"] = le_language.fit_transform(df["language"])
    le_map = dict(enumerate(le_language.classes_))

    # ── Model 1: proposal formula  y ~ T + L + (1|Language) ──────────────────
    print("Fitting Model 1:  y ~ C(relation) + sent_length + (1|Language) ...")
    df1 = df[["attention_weight", "relation", "sent_length", "language_enc"]].dropna().copy()
    df1["relation"] = pd.Categorical(df1["relation"])
    fit1 = smf.mixedlm(
        formula="attention_weight ~ C(relation) + sent_length",
        data=df1,
        groups=df1["language_enc"]
    ).fit(reml=True)

    # ── Model 2: extended formula  y ~ T + L + head + layer + (1|Language) ───
    print("Fitting Model 2:  y ~ C(relation) + sent_length + best_head + best_layer + (1|Language) ...")
    df2 = df[["attention_weight", "relation", "sent_length",
              "best_head", "best_layer", "language_enc"]].dropna().copy()
    df2["relation"] = pd.Categorical(df2["relation"])
    fit2 = smf.mixedlm(
        formula="attention_weight ~ C(relation) + sent_length + best_head + best_layer",
        data=df2,
        groups=df2["language_enc"]
    ).fit(reml=True)

    print("Extracting fixed effects and random effects ...")
    m1_stats = extract_stats(fit1)
    m2_stats = extract_stats(fit2)
    random_effects_m1 = extract_random_effects(fit1, le_map)

    out = {
        "model1":         m1_stats,
        "model2":         m2_stats,
        "random_effects": random_effects_m1,
    }

    # ── Per-language OLS models ────────────────────────────────────────────────
    for lang_name, lang_code, lang_str in LANG_CONFIGS:
        print(f"Fitting per-language OLS for {lang_name} ...")
        df_lang = df[df["language"] == lang_str][
            ["attention_weight", "relation", "sent_length"]
        ].dropna().copy()
        df_lang["relation"] = pd.Categorical(df_lang["relation"])

        if len(df_lang) < 50:
            print(f"  Skipping {lang_name}: insufficient data ({len(df_lang)} rows).")
            continue

        fit_lang = smf.ols(
            formula="attention_weight ~ C(relation) + sent_length",
            data=df_lang
        ).fit()

        stats = extract_stats_ols(fit_lang)
        out[f"model1_{lang_code}"] = stats
        print(f"  {lang_name}: n={stats['n_obs']}, R²={stats['r2_marginal']:.4f}")

    # Save JSON
    out_path = os.path.join(BACKEND_DIR, "lmm_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON saved to {out_path}")

    # ── Static Graph Generation ────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        graphs_dir = os.path.join(BACKEND_DIR, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        # Global forest plot (Model 1)
        print("\nGenerating: lmm_forest_plot.png ...")
        plot_df = pd.DataFrame([
            c for c in m1_stats["coefficients"]
            if c["name"].startswith("C(relation)")
        ]).sort_values("coef", ascending=False)

        _forest_plot(
            plot_df,
            "LMM Fixed Effects: Dependency Type Coefficients — All Languages\n"
            "y ~ beta0 + beta1*T + beta2*L + (1|Language) + epsilon",
            os.path.join(graphs_dir, "lmm_forest_plot.png")
        )

        # Per-language forest plots
        LANG_COLORS = {
            "English":  "#38bdf8",
            "Hindi":    "#f97316",
            "Japanese": "#a78bfa",
            "German":   "#34d399",
        }
        for lang_name, lang_code, _ in LANG_CONFIGS:
            key = f"model1_{lang_code}"
            if key not in out:
                continue
            print(f"Generating: lmm_forest_plot_{lang_code}.png ...")
            plot_df_lang = pd.DataFrame([
                c for c in out[key]["coefficients"]
                if c["name"].startswith("C(relation)")
            ]).sort_values("coef", ascending=False)

            _forest_plot(
                plot_df_lang,
                f"OLS Fixed Effects: Dependency Type Coefficients — {lang_name}\n"
                f"y ~ beta0 + beta1*T + beta2*L  (n={out[key]['n_obs']}, R²={out[key]['r2_marginal']:.4f})",
                os.path.join(graphs_dir, f"lmm_forest_plot_{lang_code}.png")
            )

        # Random effects chart
        print("Generating: lmm_random_effects.png ...")
        re_df = pd.DataFrame(random_effects_m1).sort_values("intercept", ascending=True)
        colors = [LANG_COLORS.get(l, "#888") for l in re_df["language"]]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.set_theme(style="whitegrid", font_scale=0.95)
        bars = ax.barh(re_df["language"], re_df["intercept"],
                       color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, re_df["intercept"]):
            ax.text(val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}",
                    va="center", ha="left" if val >= 0 else "right", fontsize=9)
        ax.set_xlabel("Random Intercept Deviation from Global Mean")
        ax.set_title(
            "LMM Random Effects: (1|Language) Intercepts\n"
            "Per-language attention baseline after controlling for T and L",
            fontweight="bold"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(graphs_dir, "lmm_random_effects.png"), dpi=150)
        plt.close(fig)
        print("  [OK] lmm_random_effects.png")

        print(f"\nAll graphs saved to: {graphs_dir}/")

    except Exception as e:
        print(f"Notice: Could not generate static plots: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
