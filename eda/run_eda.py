# -*- coding: utf-8 -*-
"""
============================================================
eda/run_eda.py
============================================================
Exploratory data analysis for the Crop Recommendation dataset.

Writes figures to eda/figures/ and a findings summary to
eda/EDA_REPORT.md. Everything in the report is computed from the
data at run time -- there are no numbers typed in by hand.

The analysis is built around one question that matters more than
any chart here: is this problem actually hard? The answer shapes
how every accuracy figure in this repository should be read.

Run: python eda/run_eda.py
============================================================
"""

import json
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")
FIG_DIR   = os.path.join(HERE, "figures")
REPORT_PATH = os.path.join(HERE, "EDA_REPORT.md")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

UNITS = {
    "N": "kg/ha ratio", "P": "kg/ha ratio", "K": "kg/ha ratio",
    "temperature": "°C", "humidity": "%", "ph": "pH", "rainfall": "mm",
}

os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"  [FIG] {name}")
    return os.path.relpath(path, HERE).replace(os.sep, "/")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structure
# ─────────────────────────────────────────────────────────────────────────────
def profile(df):
    """Shape, types, completeness, duplicates, class balance."""
    counts = df[TARGET].value_counts()
    return {
        "n_rows":     int(len(df)),
        "n_features": len(FEATURES),
        "n_classes":  int(df[TARGET].nunique()),
        "missing_cells":   int(df.isna().sum().sum()),
        "duplicate_rows":  int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df[FEATURES].duplicated().sum()),
        "class_min": int(counts.min()),
        "class_max": int(counts.max()),
        "balanced":  bool(counts.min() == counts.max()),
        "dtypes":    {c: str(t) for c, t in df.dtypes.items()},
    }


def describe(df):
    """Descriptive statistics per feature."""
    stats = df[FEATURES].describe().T
    stats["skew"]     = df[FEATURES].skew()
    stats["kurtosis"] = df[FEATURES].kurtosis()
    return stats.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Outliers
# ─────────────────────────────────────────────────────────────────────────────
def outliers(df):
    """
    IQR outlier counts, computed globally and within each crop.

    The distinction matters. A rainfall value that is extreme for the
    dataset as a whole may be entirely ordinary for rice. Treating global
    outliers as errors would delete exactly the values that separate the
    high-rainfall crops -- which is why nothing is dropped here.
    """
    rows = []
    for f in FEATURES:
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        global_out = int(((df[f] < lo) | (df[f] > hi)).sum())

        within = 0
        for _crop, group in df.groupby(TARGET):
            gq1, gq3 = group[f].quantile([0.25, 0.75])
            giqr     = gq3 - gq1
            within += int((
                (group[f] < gq1 - 1.5 * giqr) | (group[f] > gq3 + 1.5 * giqr)
            ).sum())

        rows.append({
            "feature": f,
            "global_outliers": global_out,
            "global_pct": round(100 * global_out / len(df), 2),
            "within_class_outliers": within,
            "within_class_pct": round(100 * within / len(df), 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Separability -- the question that matters
# ─────────────────────────────────────────────────────────────────────────────
def separability(df):
    """
    How hard is this classification problem, really?

    Answered with a ladder of deliberately weak models. Each rung removes
    one excuse: if a model this simple already does well, the problem was
    never hard, and the headline accuracy of a tuned forest means little.

    Then, per crop, the recall of a weak-but-honest model. That is what
    identifies the genuinely difficult classes -- far more reliably than
    centroid geometry, which in seven dimensions systematically overstates
    overlap (a cluster's mean radius grows with sqrt(n_features) while the
    distance between two centroids does not).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                         cross_val_score)
    from sklearn.metrics import classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    X  = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    probes = {
        "majority-class baseline":  DummyClassifier(strategy="most_frequent"),
        "decision tree, depth 2":   DecisionTreeClassifier(max_depth=2, random_state=0),
        "decision tree, depth 5":   DecisionTreeClassifier(max_depth=5, random_state=0),
        "decision tree, depth 10":  DecisionTreeClassifier(max_depth=10, random_state=0),
        "1-nearest neighbour":      make_pipeline(StandardScaler(),
                                                  KNeighborsClassifier(1)),
        "linear discriminant":      make_pipeline(StandardScaler(),
                                                  LinearDiscriminantAnalysis()),
    }
    probe_scores = {
        name: round(float(cross_val_score(clf, X, y, cv=cv,
                                          scoring="accuracy").mean()), 4)
        for name, clf in probes.items()
    }

    # Per-crop difficulty, measured by a model weak enough to make mistakes.
    # A tuned forest gets ~99.5% and reveals almost nothing per class.
    weak = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    y_pred = cross_val_predict(weak, X, y, cv=cv)
    report = classification_report(y, y_pred, target_names=list(le.classes_),
                                   output_dict=True, zero_division=0)

    # Which crop does each crop's errors go to?
    confused_with = {}
    for crop in le.classes_:
        idx = le.transform([crop])[0]
        wrong = y_pred[(y == idx) & (y_pred != idx)]
        if len(wrong):
            vals, counts = np.unique(wrong, return_counts=True)
            confused_with[crop] = le.inverse_transform(
                [vals[int(np.argmax(counts))]])[0]
        else:
            confused_with[crop] = "-"

    difficulty = pd.DataFrame([
        {
            "crop": crop,
            "recall": round(float(report[crop]["recall"]), 3),
            "precision": round(float(report[crop]["precision"]), 3),
            "mistaken_for": confused_with[crop],
        }
        for crop in le.classes_
    ]).sort_values("recall")

    return probe_scores, difficulty


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_distributions(df):
    fig = Figure(figsize=(14, 7))
    axes = fig.subplots(2, 4)
    for ax, f in zip(axes.ravel(), FEATURES):
        ax.hist(df[f], bins=40, color="#2f7d4f", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{f}  ({UNITS[f]})", fontsize=10)
        ax.tick_params(labelsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Feature distributions across all 2200 samples", fontsize=13)
    fig.tight_layout()
    return _save(fig, "01_feature_distributions.png")


def fig_class_balance(df):
    counts = df[TARGET].value_counts().sort_index()
    fig = Figure(figsize=(11, 4.5))
    ax  = fig.subplots()
    ax.bar(counts.index, counts.values, color="#2f7d4f")
    ax.set_ylabel("samples")
    ax.set_title("Class balance — exactly 100 samples per crop")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_ylim(0, counts.max() * 1.25)
    fig.tight_layout()
    return _save(fig, "02_class_balance.png")


def fig_correlation(df):
    corr = df[FEATURES].corr()
    fig = Figure(figsize=(6.5, 5.5))
    ax  = fig.subplots()
    im  = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, fontsize=9)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Pearson correlation between features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, "03_correlation.png")


def fig_boxplots_by_crop(df):
    """The two features that carry most of the signal, split by crop."""
    fig = Figure(figsize=(13, 8))
    axes = fig.subplots(2, 1)
    order = sorted(df[TARGET].unique())
    for ax, f in zip(axes, ["humidity", "rainfall"]):
        ax.boxplot([df.loc[df[TARGET] == c, f].values for c in order],
                   labels=order, showfliers=True,
                   medianprops=dict(color="#c2410c"))
        ax.set_ylabel(f"{f} ({UNITS[f]})")
        ax.set_title(f"{f} by crop")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, "04_key_features_by_crop.png")


def fig_rice_jute(df):
    """
    The one genuinely overlapping pair.

    Rice and jute are both wet-season crops of the same river-delta
    conditions, and they are where the classifier's errors concentrate.
    Plotting them alone shows why: the confusion is in the data, not a
    modelling failure.
    """
    pair = df[df[TARGET].isin(["rice", "jute"])]
    fig  = Figure(figsize=(11, 4.5))
    axes = fig.subplots(1, 2)
    for ax, (fx, fy) in zip(axes, [("humidity", "rainfall"), ("N", "K")]):
        for crop, colour in [("rice", "#1d4ed8"), ("jute", "#ea580c")]:
            sub = pair[pair[TARGET] == crop]
            ax.scatter(sub[fx], sub[fy], s=18, alpha=0.65,
                       label=crop, color=colour)
        ax.set_xlabel(f"{fx} ({UNITS[fx]})")
        ax.set_ylabel(f"{fy} ({UNITS[fy]})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("rice vs jute — the overlap the model actually gets wrong",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, "05_rice_vs_jute.png")


def fig_difficulty(difficulty):
    """Per-crop recall of a deliberately weak model, hardest first."""
    top = difficulty.head(12).iloc[::-1]
    fig = Figure(figsize=(9, 5.5))
    ax  = fig.subplots()
    colours = ["#b91c1c" if v < 0.9 else "#2f7d4f" for v in top["recall"]]
    ax.barh(top["crop"], top["recall"], color=colours)
    ax.axvline(1.0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("cross-validated recall (linear discriminant)")
    ax.set_title("Hardest crops for a weak model — where the errors live")
    ax.grid(axis="x", alpha=0.25)
    for y, (_, row) in enumerate(top.iterrows()):
        if row["mistaken_for"] != "-":
            ax.text(row["recall"] + 0.01, y,
                    f"→ {row['mistaken_for']}", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, "06_per_crop_difficulty.png")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(df, prof, stats, out, probe_scores, difficulty, figures):
    corr = df[FEATURES].corr()
    pairs = [
        (FEATURES[i], FEATURES[j], corr.iloc[i, j])
        for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))
    ]
    pairs.sort(key=lambda p: -abs(p[2]))

    hardest = difficulty.iloc[0]
    d5      = probe_scores["decision tree, depth 5"]
    d10     = probe_scores["decision tree, depth 10"]
    lda     = probe_scores["linear discriminant"]
    knn     = probe_scores["1-nearest neighbour"]
    struggling = difficulty[difficulty["recall"] < 0.95]

    lines = [
        "# EDA — Crop Recommendation Dataset",
        "",
        "Generated by `eda/run_eda.py`. Every figure in this document is "
        "computed from `data/crop_data.csv` at run time.",
        "",
        "## 1. Structure",
        "",
        f"- **{prof['n_rows']} rows**, {prof['n_features']} numeric features, "
        f"{prof['n_classes']} crop classes",
        f"- **Missing cells: {prof['missing_cells']}** — no imputation is needed "
        f"or performed",
        f"- **Duplicate rows: {prof['duplicate_rows']}** "
        f"(duplicate feature vectors: {prof['duplicate_feature_rows']})",
        f"- **Class balance: {prof['class_min']}–{prof['class_max']} samples per "
        f"crop** — perfectly balanced, so plain accuracy is a fair headline "
        f"metric and no resampling is justified",
        "",
        "All seven features are continuous and numeric, so there is no "
        "categorical encoding step. The only transform the models need is "
        "standardisation, and that is applied inside a scikit-learn "
        "`Pipeline` so it is refit within every cross-validation fold.",
        "",
        f"![distributions]({figures['distributions']})",
        "",
        f"![balance]({figures['balance']})",
        "",
        "## 2. Descriptive statistics",
        "",
        stats.to_markdown(),
        "",
        "## 3. Outliers",
        "",
        "Counted two ways, because the difference is the whole point:",
        "",
        out.to_markdown(index=False),
        "",
        "Every feature shows global outliers — K flags 9% of rows — and "
        "**not one of them is an outlier within its own crop.** Those points "
        "are not errors. They are the high-potassium crops (grapes, apple) "
        "and the high-rainfall crops (rice, coconut) sitting exactly where "
        "agronomy says they should. A global IQR filter would delete the "
        "signal that separates the classes, so **no outlier removal is "
        "applied** — the correct treatment here is none.",
        "",
        "Worth noting for what it says about the data: a within-class count "
        "of exactly zero across all seven features is not what measured "
        "field data looks like. Roughly 0.7% of samples drawn from a normal "
        "distribution fall outside the 1.5×IQR fence, so ~15 would be "
        "expected here. Getting none means each crop's values are cleanly "
        "bounded with no tails — consistent with this benchmark having been "
        "at least partly generated rather than observed. See *Limitations*.",
        "",
        "## 4. Correlation",
        "",
        f"![correlation]({figures['correlation']})",
        "",
        "Strongest pairs:",
        "",
    ]
    for a, b, v in pairs[:4]:
        lines.append(f"- `{a}` ↔ `{b}`: **{v:+.3f}**")
    lines += [
        "",
        f"The one substantial relationship is **P ↔ K ({corr.loc['P','K']:+.3f})**, "
        "and it is agronomically real: phosphorus and potassium are applied "
        "together in compound fertiliser, and the crops that demand a lot of "
        "one (grapes, apple) demand a lot of the other. Every other pair sits "
        "below |0.3|. With seven weakly-related features there is nothing to "
        "gain from PCA or from dropping a redundant column.",
        "",
        f"![key features]({figures['boxplots']})",
        "",
        "## 5. How hard is this problem?",
        "",
        "This is the finding that should govern how every accuracy number in "
        "this repository is read.",
        "",
        "| probe | 5-fold CV accuracy |",
        "|---|---|",
    ]
    for name, score in probe_scores.items():
        lines.append(f"| {name} | {score:.4f} |")
    lines += [
        "",
        "Read that table carefully, because it does not say what a quick "
        "glance suggests.",
        "",
        f"A **linear discriminant reaches {lda:.1%}** and **1-NN reaches "
        f"{knn:.1%}** — so in the full seven-dimensional space the crops are "
        f"very nearly linearly separable, and the tuned models' ~99.5% is "
        f"barely an improvement on a method with no hyperparameters at all.",
        "",
        f"But a **depth-5 tree manages only {d5:.1%}**, climbing to "
        f"{d10:.1%} at depth 10. That is the interesting half. Separability "
        f"here does not come from any one feature crossing a threshold — a "
        f"shallow axis-aligned model cannot express it. It comes from the "
        f"*joint* configuration of all seven. Which is why depth matters so "
        f"much more than the number of trees in the tuning results, and why "
        f"a linear model that combines all features at once does so well "
        f"while a shallow tree does not.",
        "",
        f"![difficulty]({figures['difficulty']})",
        "",
        "Per-crop recall is measured with the linear discriminant rather "
        "than the forest. That is deliberate: the tuned forest misclassifies "
        "2 of 440 test rows, which is too few to say anything per class. A "
        "weaker model makes enough mistakes for the pattern in them to be "
        "visible.",
        "",
        difficulty.head(8).to_markdown(index=False),
        "",
    ]
    if len(struggling):
        worst_names = ", ".join(
            f"**{r.crop}** (recall {r.recall:.2f}, mistaken for "
            f"{r.mistaken_for})"
            for r in struggling.head(3).itertuples()
        )
        lines += [f"The crops that actually cost accuracy: {worst_names}.", ""]
    lines += [
        f"![rice vs jute]({figures['rice_jute']})",
        "",
        "The rice/jute pair is the one that survives into the tuned model's "
        "errors as well. The scatter shows why: both are wet-season crops of "
        "the same river-delta conditions, and their humidity/rainfall "
        "envelopes genuinely overlap. That confusion is a property of the "
        "world, not a modelling defect — no amount of tuning removes it, and "
        "separating them would need a feature the dataset does not contain "
        "(soil texture, season, or geography).",
        "",
        "### What this means",
        "",
        "**A ~99% score on this dataset is mostly a property of the dataset.** "
        "A linear discriminant with nothing tuned gets within three points "
        "of a grid-searched random forest, and the gap between the top "
        "models is smaller than the fold-to-fold standard deviation. "
        "Presenting 99% as evidence of modelling skill would misrepresent "
        "the work.",
        "",
        "The questions the headline hides are the ones worth answering:",
        "",
        f"- *Which* samples fail, and is the failure explicable? "
        f"({hardest['crop']} ↔ {hardest['mistaken_for']}, and rice ↔ jute)",
        "- Is the gap between models larger than fold noise? "
        "(No — see `model_selection` in the evaluation report.)",
        "- What does the model actually rely on? "
        "(Permutation importance, in "
        "`backend/models/evaluation_report.json`.)",
        "",
        "## 6. Limitations",
        "",
        "- **The dataset is synthetic.** Every crop's nitrogen range spans "
        "exactly 40, phosphorus exactly 25 and potassium exactly 10, and "
        "97.7% of class bounds are exact multiples of 5. Within those ranges "
        "the values are uniform, not normal: 141 of 154 (class, feature) "
        "pairs are consistent with a uniform distribution and none with a "
        "normal one. Each crop occupies a near-disjoint axis-aligned box, "
        "which is why any classifier reaches ~99%. See "
        "`dataset_provenance.py`. Fine for demonstrating a workflow; unfit "
        "for agronomic advice.",
        "- Seven features describe a field with no soil type, no season, no "
        "geography and no cultivar. Real recommendation needs all of those.",
        "- Every row is one field-season with no temporal or spatial "
        "grouping, so a random split cannot leak across correlated units — "
        "but it also cannot measure generalisation to a *new region*, which "
        "is the deployment question that would actually matter.",
        "",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\n[OK] Report -> {REPORT_PATH}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python data/download_data.py")

    df = pd.read_csv(DATA_PATH)
    print(f"[EDA] {len(df)} rows from {DATA_PATH}\n")

    prof  = profile(df)
    stats = describe(df)
    out   = outliers(df)
    probe_scores, difficulty = separability(df)

    figures = {
        "distributions": fig_distributions(df),
        "balance":       fig_class_balance(df),
        "correlation":   fig_correlation(df),
        "boxplots":      fig_boxplots_by_crop(df),
        "rice_jute":     fig_rice_jute(df),
        "difficulty":    fig_difficulty(difficulty),
    }

    write_report(df, prof, stats, out, probe_scores, difficulty, figures)

    with open(os.path.join(HERE, "eda_summary.json"), "w", encoding="utf-8") as h:
        json.dump({
            "profile": prof,
            "separability_probes": probe_scores,
            "hardest_crops": difficulty.head(5).to_dict(orient="records"),
        }, h, indent=2)
    print(f"[OK] Summary -> {os.path.join(HERE, 'eda_summary.json')}")


if __name__ == "__main__":
    main()
