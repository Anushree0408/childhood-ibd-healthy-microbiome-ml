# file: supervised_algo/ml_models_nestedcv_leakageproof_with_inner_plots_timed.py
"""
Leakage-proof ML for pediatric gut microbiome (amplicon) with age bins:
- 0–4 years: IBD vs CONTROL
- 4–12 years: IBD vs CONTROL

Models included:
- SVM (linear), SVM (RBF)
- Logistic Regression (L2)
- Elastic Net Logistic Regression
- KNN
- Naive Bayes (GaussianNB)

Leakage control:
- Genus collapse is label-free.
- ALL preprocessing/filtering (rel-abundance, prevalence, CLR, variance, impute, scale) happens inside pipelines.
- Hyperparameter tuning happens only inside inner CV.
- Metrics are computed from OUTER-CV OOF probabilities only.

IMPORTANT FIX (for thesis figures):
- Outer CV uses RepeatedStratifiedKFold, so each sample is tested multiple times.
- We now STORE + AVERAGE OOF probabilities per sample across repeats
  (instead of overwriting). This makes PR/ROC/confusion stable and correct.

Inner-CV visualization:
- Option A: mean/std inner-CV PR-AUC surface across outer folds (1D or 2D depending on model grid)
- Option B: best-parameter frequency across outer folds

Plot styling:
- Confusion matrix: blue colormap, NO colorbar, NO title, axis labels + tick labels bold + bigger.
- Learning curve: NO title, axis labels + tick labels bold + bigger.
- PR curve: NO title, axis labels + tick labels bold + bigger, PR-AUC shown in legend.
- ROC curve: NO title, axis labels + tick labels bold + bigger, ROC-AUC shown in legend.

Timing:
- Times each (bin, model) and total runtime.
- n_jobs defaults to 8.

Defaults:
- Uses your absolute input paths by default (you can still override via CLI).

Run (using defaults):
  python ml_models_nestedcv_leakageproof_with_inner_plots_timed.py --outdir out_ml --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# -------------------------
# Defaults (your paths)
# -------------------------

DEFAULT_FEATURE_TABLE = (
    "data/feature_table_final_filtered.tsv"
)
DEFAULT_METADATA = (
    "data/ibd_meta_filtered.txt"
)
DEFAULT_TAXONOMY = (
    "data/taxonomy_230_220_final.tsv"
)


# -------------------------
# Global plot style
# -------------------------

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 16


def _bold_ticks(ax) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(TICK_LABEL_SIZE)


# -------------------------
# Path cleanup helpers
# -------------------------

def _maybe_fix_path_spaces(path: str) -> str:
    """
    Your pasted paths contain '.../understand /supervised_algo /file.tsv' (spaces before '/').
    If the path doesn't exist, try a few safe normalizations.
    """
    path = str(path)
    if os.path.exists(path):
        return path

    candidates = []
    candidates.append(path.replace(" /", "/").replace("/ ", "/"))
    candidates.append(re.sub(r"\s+/", "/", path))
    candidates.append(re.sub(r"/\s+", "/", path))
    candidates.append(re.sub(r"\s*/\s*", "/", path))

    for p in candidates:
        if os.path.exists(p):
            return p

    return path


# -------------------------
# IO helpers
# -------------------------

def _read_table_auto_sep(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if df.shape[1] <= 2:
            raise ValueError("TSV parse too few columns")
        return df
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")


def read_feature_table_as_samples_x_features(path: str) -> pd.DataFrame:
    df = _read_table_auto_sep(path)

    cols = list(df.columns)
    if len(cols) >= 2 and cols[0] == "#OTU" and cols[1] == "ID":
        df = df.rename(columns={"#OTU": "#OTU_ID"})
        df["#OTU_ID"] = df["#OTU_ID"].astype(str)
        df = df.drop(columns=["ID"])
        df = df.rename(columns={"#OTU_ID": "#OTU ID"})

    feature_id_col = df.columns[0]
    df = df.rename(columns={feature_id_col: "FeatureID"}).set_index("FeatureID")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    return df.T


def read_metadata(path: str) -> pd.DataFrame:
    md = _read_table_auto_sep(path)
    colmap = {}
    for c in md.columns:
        if c.lower() == "sample":
            colmap[c] = "Sample"
        elif c.lower() == "group":
            colmap[c] = "Group"
        elif c.lower() in ("age", "ages"):
            colmap[c] = "AGE"
    md = md.rename(columns=colmap)

    required = {"Sample", "Group", "AGE"}
    if required - set(md.columns):
        raise ValueError(f"Metadata must contain columns: {sorted(required)}")

    md["AGE"] = pd.to_numeric(md["AGE"], errors="coerce")
    md = md.dropna(subset=["AGE"])
    md["Group"] = md["Group"].astype(str).str.upper()
    return md.set_index("Sample")


def read_taxonomy(path: str) -> pd.DataFrame:
    tx = _read_table_auto_sep(path)
    if "Feature" in tx.columns and "ID" in tx.columns:
        tx["Feature ID"] = tx["Feature"].astype(str) + tx["ID"].astype(str)
        tx = tx.drop(columns=["Feature", "ID"])
    if "Feature ID" not in tx.columns:
        tx = tx.rename(columns={tx.columns[0]: "Feature ID"})
    if "Taxon" not in tx.columns:
        raise ValueError("Taxonomy file must contain a Taxon column.")
    tx = tx.rename(columns={"Feature ID": "FeatureID"})
    tx["FeatureID"] = tx["FeatureID"].astype(str)
    tx["Taxon"] = tx["Taxon"].astype(str)
    return tx[["FeatureID", "Taxon"]]


# -------------------------
# Taxonomy -> Genus collapse (label-free)
# -------------------------

_GENUS_RX = re.compile(r"(?:^|;\s*)g__([^;]+)")


def taxon_to_genus(taxon: str) -> str:
    m = _GENUS_RX.search(taxon or "")
    if not m:
        return "Unassigned"
    g = m.group(1).strip()
    if not g or g.lower() in {"uncultured", "unassigned", "unknown"}:
        return "Unassigned"
    return g


def collapse_features_to_genus(X: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    feat_ids = X.columns.astype(str)
    tx = taxonomy[taxonomy["FeatureID"].isin(feat_ids)].copy()
    if tx.empty:
        raise ValueError("No overlap between feature IDs in feature table and taxonomy file.")
    tx["Genus"] = tx["Taxon"].apply(taxon_to_genus)
    fmap = dict(zip(tx["FeatureID"], tx["Genus"]))

    genera = [fmap.get(str(fid), "Unassigned") for fid in feat_ids]
    Xg = X.copy()
    Xg.columns = genera
    Xg = Xg.groupby(axis=1, level=0).sum()
    Xg = Xg.loc[:, (Xg.sum(axis=0) > 0)]
    return Xg


# -------------------------
# Leakage-safe transformers
# -------------------------

class RelativeAbundance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return X / rs


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_prevalence: float = 0.05):
        self.min_prevalence = float(min_prevalence)
        self._mask: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        prev = (X > 0).mean(axis=0)
        self._mask = prev >= self.min_prevalence
        if not np.any(self._mask):
            self._mask = np.ones(X.shape[1], dtype=bool)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self._mask]


class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_variance: float = 1e-8):
        self.min_variance = float(min_variance)
        self._mask: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        var = X.var(axis=0)
        self._mask = var >= self.min_variance
        if not np.any(self._mask):
            self._mask = np.ones(X.shape[1], dtype=bool)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self._mask]


class CLRTransform(BaseEstimator, TransformerMixin):
    def __init__(self, pseudocount: float = 1e-6):
        self.pseudocount = float(pseudocount)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float) + self.pseudocount
        logX = np.log(X)
        gm = logX.mean(axis=1, keepdims=True)
        return logX - gm


# -------------------------
# Metrics
# -------------------------

@dataclass(frozen=True)
class Metrics:
    pr_auc: float
    roc_auc: float
    bal_acc: float
    recall: float
    specificity: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return Metrics(
        pr_auc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


# -------------------------
# Plot helpers (styled)
# -------------------------

def save_confusion_png(path: str, tn: int, fp: int, fn: int, tp: int) -> None:
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots()
    ax.imshow(mat, interpolation="nearest", cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["CONTROL", "IBD"])
    ax.set_yticklabels(["CONTROL", "IBD"])

    ax.set_xlabel("Predicted", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("True", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _bold_ticks(ax)

    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center", fontweight="bold", fontsize=TICK_LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_pr_curve(path: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR-AUC={ap:.3f}")

    ax.set_xlabel("Recall", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _bold_ticks(ax)

    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve(path: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _bold_ticks(ax)

    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_learning_curve(
    out_png: str,
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    n_jobs: int,
    scoring: str = "average_precision",
) -> None:
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.2, 1.0, 6),
        shuffle=True,
        random_state=42,
        error_score="raise",
    )
    tr_m, tr_s = train_scores.mean(axis=1), train_scores.std(axis=1)
    te_m, te_s = test_scores.mean(axis=1), test_scores.std(axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, tr_m, marker="o", label="Train")
    ax.plot(train_sizes, te_m, marker="o", label="CV")
    ax.fill_between(train_sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.2)
    ax.fill_between(train_sizes, te_m - te_s, te_m + te_s, alpha=0.2)

    ax.set_xlabel("Training samples", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(scoring, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _bold_ticks(ax)

    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def save_heatmap(
    out_png: str,
    Z: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(Z, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    _bold_ticks(ax)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# Inner-CV aggregation (Option A/B) for 1D or 2D grids
# -------------------------

@dataclass(frozen=True)
class GridSpec:
    """Describes which hyperparameters to extract from GridSearch for inner plotting."""
    x_param: str
    x_vals: List[Any]
    x_label: str
    y_param: Optional[str] = None
    y_vals: Optional[List[Any]] = None
    y_label: Optional[str] = None


def _key(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _extract_surface_df(cv_results: Dict[str, Any], grid_spec: GridSpec) -> pd.DataFrame:
    rows = []
    for score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        x = params.get(grid_spec.x_param)
        y = params.get(grid_spec.y_param) if grid_spec.y_param else None
        rows.append({"x": x, "y": y, "score": float(score)})
    return pd.DataFrame(rows)


def _aggregate_surfaces(
    fold_surfaces: List[pd.DataFrame],
    grid_spec: GridSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    x_vals = grid_spec.x_vals
    y_vals = grid_spec.y_vals or [None]

    x_index = {_key(v): i for i, v in enumerate(x_vals)}
    y_index = {_key(v): i for i, v in enumerate(y_vals)}

    cube = np.full((len(fold_surfaces), len(y_vals), len(x_vals)), np.nan, dtype=float)

    for fi, df in enumerate(fold_surfaces):
        for _, r in df.iterrows():
            xi = x_index.get(_key(r["x"]))
            yi = y_index.get(_key(r["y"])) if grid_spec.y_param else y_index.get(_key(None))
            if xi is None or yi is None:
                continue
            cube[fi, yi, xi] = float(r["score"])

    return np.nanmean(cube, axis=0), np.nanstd(cube, axis=0)


def _best_param_frequency(
    best_params: List[Dict[str, Any]],
    grid_spec: GridSpec,
) -> pd.DataFrame:
    rows = []
    for bp in best_params:
        rows.append({
            "x": bp.get(grid_spec.x_param),
            "y": bp.get(grid_spec.y_param) if grid_spec.y_param else None,
        })
    df = pd.DataFrame(rows)
    return df.value_counts().reset_index(name="count").sort_values("count", ascending=False)


# -------------------------
# Nested CV with inner capture (FIXED: average probs per sample across repeats)
# -------------------------

def nested_oof_predictions_with_inner(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Pipeline,
    param_grid: Dict[str, List],
    grid_spec: Optional[GridSpec],
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[pd.DataFrame]]:
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state
    )

    # FIX: accumulate and average probabilities per sample across repeats
    sum_prob = np.zeros_like(y, dtype=float)
    count_pred = np.zeros_like(y, dtype=int)

    best_params: List[Dict[str, Any]] = []
    inner_surfaces: List[pd.DataFrame] = []

    for tr, te in outer_cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="average_precision",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        )
        gs.fit(Xtr, ytr)

        prob = gs.predict_proba(Xte)[:, 1]
        sum_prob[te] += prob
        count_pred[te] += 1

        best_params.append(gs.best_params_)
        if grid_spec is not None:
            inner_surfaces.append(_extract_surface_df(gs.cv_results_, grid_spec))

    # average per sample (should usually be constant for all samples, but keep safe)
    count_pred[count_pred == 0] = 1
    y_prob = sum_prob / count_pred

    return y_prob, best_params, inner_surfaces


# -------------------------
# Model factory (ALL models)
# -------------------------

def build_models(
    pseudocount: float,
    min_prevalence: float,
    min_variance: float,
    seed: int,
) -> Dict[str, Tuple[Pipeline, Dict[str, List], Optional[GridSpec]]]:
    pre = [
        ("rel", RelativeAbundance()),
        ("prev", PrevalenceFilter(min_prevalence=min_prevalence)),
        ("clr", CLRTransform(pseudocount=pseudocount)),
        ("var", VarianceFilter(min_variance=min_variance)),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]

    # ---- SVM
    C_vals = np.logspace(-3, 3, 13).tolist()
    gamma_vals = np.logspace(-4, 2, 13).tolist()

    svm_linear = Pipeline(pre + [
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed))
    ])
    svm_linear_grid = {"clf__C": C_vals}
    svm_linear_spec = GridSpec(
        x_param="clf__C", x_vals=C_vals, x_label="C",
        y_param=None, y_vals=None, y_label=None
    )

    svm_rbf = Pipeline(pre + [
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))
    ])
    svm_rbf_grid = {"clf__C": C_vals, "clf__gamma": gamma_vals}
    svm_rbf_spec = GridSpec(
        x_param="clf__C", x_vals=C_vals, x_label="C",
        y_param="clf__gamma", y_vals=gamma_vals, y_label="gamma"
    )

    # ---- Logistic Regression (L2)
    lr = Pipeline(pre + [
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=20000,
            random_state=seed,
        ))
    ])
    lr_grid = {"clf__C": C_vals}
    lr_spec = GridSpec(x_param="clf__C", x_vals=C_vals, x_label="C")

    # ---- Elastic Net Logistic Regression
    # IMPORTANT: set l1_ratio default in estimator to avoid errors in learning_curve.
    l1_ratios = [0.1, 0.5, 0.9]
    enet = Pipeline(pre + [
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=20000,
            random_state=seed,
            l1_ratio=0.5,
        ))
    ])
    enet_grid = {"clf__C": C_vals, "clf__l1_ratio": l1_ratios}
    enet_spec = GridSpec(
        x_param="clf__C", x_vals=C_vals, x_label="C",
        y_param="clf__l1_ratio", y_vals=l1_ratios, y_label="l1_ratio"
    )

    # ---- KNN
    ks = [3, 5, 7, 11, 15, 21, 31]
    weights = ["uniform", "distance"]
    knn = Pipeline(pre + [
        ("clf", KNeighborsClassifier())
    ])
    knn_grid = {"clf__n_neighbors": ks, "clf__weights": weights}
    knn_spec = GridSpec(
        x_param="clf__n_neighbors", x_vals=ks, x_label="n_neighbors",
        y_param="clf__weights", y_vals=weights, y_label="weights"
    )

    # ---- Naive Bayes
    vs = np.logspace(-12, -6, 7).tolist()
    nb = Pipeline(pre + [
        ("clf", GaussianNB())
    ])
    nb_grid = {"clf__var_smoothing": vs}
    nb_spec = GridSpec(x_param="clf__var_smoothing", x_vals=vs, x_label="var_smoothing")

    return {
        "svm_linear": (svm_linear, svm_linear_grid, svm_linear_spec),
        "svm_rbf": (svm_rbf, svm_rbf_grid, svm_rbf_spec),
        "logistic_regression": (lr, lr_grid, lr_spec),
        "elastic_net": (enet, enet_grid, enet_spec),
        "knn": (knn, knn_grid, knn_spec),
        "naive_bayes": (nb, nb_grid, nb_spec),
    }


# -------------------------
# Age bins
# -------------------------

def make_age_bin_mask(age: pd.Series, bin_name: str) -> pd.Series:
    if bin_name == "0-4":
        return (age >= 0.0) & (age < 4.0)
    if bin_name == "4-12":
        return (age >= 4.0) & (age <= 12.0)
    raise ValueError(f"Unknown bin: {bin_name}")


# -------------------------
# Timing helpers
# -------------------------

def fmt_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (3600 * h + 60 * m)
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


# -------------------------
# Main
# -------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE)
    p.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    p.add_argument("--metadata", default=DEFAULT_METADATA)
    p.add_argument("--outdir", default="out_ml")
    p.add_argument("--n-jobs", type=int, default=8, help="Threads (default 8)")

    p.add_argument("--pseudocount", type=float, default=1e-6)
    p.add_argument("--min-prevalence", type=float, default=0.05)
    p.add_argument("--min-variance", type=float, default=1e-8)

    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=10)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    feature_table = _maybe_fix_path_spaces(args.feature_table)
    taxonomy_path = _maybe_fix_path_spaces(args.taxonomy)
    metadata_path = _maybe_fix_path_spaces(args.metadata)

    total_t0 = time.perf_counter()

    X_raw = read_feature_table_as_samples_x_features(feature_table)
    md = read_metadata(metadata_path)
    tx = read_taxonomy(taxonomy_path)

    common = X_raw.index.intersection(md.index)
    if common.empty:
        raise ValueError("No overlapping samples between feature table and metadata.")
    X_raw = X_raw.loc[common]
    md = md.loc[common]

    X_genus = collapse_features_to_genus(X_raw, tx)
    y_all = (md["Group"] == "IBD").astype(int)

    models = build_models(
        pseudocount=args.pseudocount,
        min_prevalence=args.min_prevalence,
        min_variance=args.min_variance,
        seed=args.random_state,
    )

    summary_rows = []
    timing_rows = []

    for bin_name in ["0-4", "4-12"]:
        mask = make_age_bin_mask(md["AGE"], bin_name)
        Xb_df = X_genus.loc[mask]
        yb = y_all.loc[mask].to_numpy()
        Xb = Xb_df.to_numpy(dtype=float)
        samples = Xb_df.index.astype(str).to_numpy()

        if len(np.unique(yb)) < 2:
            print(f"[WARN] Bin {bin_name}: only one class present. Skipping.")
            continue

        print(f"[{bin_name}] n={len(yb)} IBD={int(yb.sum())} CONTROL={int((1-yb).sum())}")

        lc_cv = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.random_state)

        for model_name, (pipe, grid, grid_spec) in models.items():
            block_t0 = time.perf_counter()
            print(f"[{bin_name} | {model_name}] start...")

            y_prob, best_params, inner_surfaces = nested_oof_predictions_with_inner(
                X=Xb,
                y=yb,
                estimator=pipe,
                param_grid=grid,
                grid_spec=grid_spec,
                outer_splits=args.outer_splits,
                outer_repeats=args.outer_repeats,
                inner_splits=args.inner_splits,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )

            m = compute_metrics(yb, y_prob, threshold=0.5)

            # OOF predictions (NOW averaged across repeats)
            pd.DataFrame({"Sample": samples, "y_true": yb, "y_prob": y_prob}).to_csv(
                os.path.join(outdir, f"oof_predictions_{bin_name}_{model_name}.csv"),
                index=False,
            )

            # Option A: inner surface mean/std across outer folds (PR-AUC)
            if grid_spec is not None and inner_surfaces:
                mean_Z, std_Z = _aggregate_surfaces(inner_surfaces, grid_spec)

                mean_csv = os.path.join(outdir, f"innercv_pr_auc_mean_{bin_name}_{model_name}.csv")
                std_csv = os.path.join(outdir, f"innercv_pr_auc_std_{bin_name}_{model_name}.csv")

                x_labels = [_key(v) for v in grid_spec.x_vals]
                y_labels = [_key(v) for v in (grid_spec.y_vals or [None])]

                pd.DataFrame(mean_Z, index=y_labels, columns=x_labels).to_csv(mean_csv)
                pd.DataFrame(std_Z, index=y_labels, columns=x_labels).to_csv(std_csv)

                save_heatmap(
                    out_png=os.path.join(outdir, f"innercv_pr_auc_mean_{bin_name}_{model_name}.png"),
                    Z=mean_Z,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    xlabel=grid_spec.x_label,
                    ylabel=grid_spec.y_label or "",
                )
                save_heatmap(
                    out_png=os.path.join(outdir, f"innercv_pr_auc_std_{bin_name}_{model_name}.png"),
                    Z=std_Z,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    xlabel=grid_spec.x_label,
                    ylabel=grid_spec.y_label or "",
                )

                # Option B: best param frequency
                freq_df = _best_param_frequency(best_params, grid_spec)
                freq_df.to_csv(
                    os.path.join(outdir, f"best_param_freq_{bin_name}_{model_name}.csv"),
                    index=False,
                )

            # After-training evaluation plots (outer OOF averaged)
            save_confusion_png(
                os.path.join(outdir, f"confusion_{bin_name}_{model_name}.png"),
                m.tn, m.fp, m.fn, m.tp,
            )
            save_pr_curve(
                os.path.join(outdir, f"pr_curve_{bin_name}_{model_name}.png"),
                yb, y_prob,
            )
            save_roc_curve(
                os.path.join(outdir, f"roc_curve_{bin_name}_{model_name}.png"),
                yb, y_prob,
            )

            # During-training visualization: learning curve
            save_learning_curve(
                out_png=os.path.join(outdir, f"learning_curve_{bin_name}_{model_name}.png"),
                estimator=pipe,
                X=Xb,
                y=yb,
                cv=lc_cv,
                n_jobs=args.n_jobs,
                scoring="average_precision",
            )

            elapsed = time.perf_counter() - block_t0
            print(f"[{bin_name} | {model_name}] done in {fmt_seconds(elapsed)}")

            timing_rows.append({
                "bin": bin_name,
                "model": model_name,
                "seconds": elapsed,
                "human": fmt_seconds(elapsed),
                "n_jobs": args.n_jobs,
                "outer_splits": args.outer_splits,
                "outer_repeats": args.outer_repeats,
                "inner_splits": args.inner_splits,
            })

            # Metrics txt
            met_path = os.path.join(outdir, f"metrics_{bin_name}_{model_name}.txt")
            with open(met_path, "w", encoding="utf-8") as f:
                f.write(f"Bin: {bin_name}\nModel: {model_name}\n\n")
                f.write(f"Runtime: {fmt_seconds(elapsed)}\nThreads(n_jobs): {args.n_jobs}\n\n")
                f.write(f"PR-AUC: {m.pr_auc:.6f}\n")
                f.write(f"ROC-AUC: {m.roc_auc:.6f}\n")
                f.write(f"Balanced accuracy: {m.bal_acc:.6f}\n")
                f.write(f"Recall/Sensitivity: {m.recall:.6f}\n")
                f.write(f"Specificity: {m.specificity:.6f}\n")
                f.write(f"F1: {m.f1:.6f}\n\n")
                f.write("Confusion matrix (threshold=0.5):\n")
                f.write(f"TN={m.tn} FP={m.fp}\nFN={m.fn} TP={m.tp}\n\n")
                f.write("Best params per outer fold:\n")
                for i, bp in enumerate(best_params, start=1):
                    f.write(f"  fold{i}: {bp}\n")

            summary_rows.append({
                "bin": bin_name,
                "model": model_name,
                "pr_auc": m.pr_auc,
                "roc_auc": m.roc_auc,
                "balanced_accuracy": m.bal_acc,
                "recall_sensitivity": m.recall,
                "specificity": m.specificity,
                "f1": m.f1,
                "tn": m.tn, "fp": m.fp, "fn": m.fn, "tp": m.tp,
                "runtime_seconds": elapsed,
                "runtime_human": fmt_seconds(elapsed),
            })

    summary = pd.DataFrame(summary_rows).sort_values(["bin", "pr_auc"], ascending=[True, False])
    summary_path = os.path.join(outdir, "results_all_bins.csv")
    summary.to_csv(summary_path, index=False)

    timing = pd.DataFrame(timing_rows).sort_values(["bin", "seconds"])
    timing_path = os.path.join(outdir, "timing.csv")
    timing.to_csv(timing_path, index=False)

    total_elapsed = time.perf_counter() - total_t0
    with open(os.path.join(outdir, "timing_total.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total runtime: {fmt_seconds(total_elapsed)}\n")
        f.write(f"Threads(n_jobs): {args.n_jobs}\n")
        f.write(f"outer_splits={args.outer_splits} outer_repeats={args.outer_repeats} inner_splits={args.inner_splits}\n")

    print(f"Saved summary: {summary_path}")
    print(f"Saved timing: {timing_path}")
    print(f"Total runtime: {fmt_seconds(total_elapsed)}")
    print("Done.")


if __name__ == "__main__":
    main()
