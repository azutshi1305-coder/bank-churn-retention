"""Metrics and plots shared by training and the comparison notebook."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def lift_at(y, p, fraction):
    """Churn rate among the top `fraction` of customers by score, divided by the overall churn rate."""
    k = int(np.ceil(len(y) * fraction))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def score_all(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        f"precision_at_{threshold}": precision_score(y, pred, zero_division=0),
        f"recall_at_{threshold}": recall_score(y, pred, zero_division=0),
        "lift_top10": lift_at(y, p, 0.10),
        "lift_top20": lift_at(y, p, 0.20),
    }


BOOTSTRAP_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier": brier_score_loss}


def bootstrap_metrics(y, probs, n_boot=1000, seed=42):
    """Resample test rows with replacement; every model sees the same resamples so differences can be paired.

    Captures test-set sampling noise only: the fitted models are held fixed.
    """
    rng = np.random.default_rng(seed)
    draws = {name: {m: np.empty(n_boot) for m in BOOTSTRAP_METRICS} for name in probs}
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for name, p in probs.items():
            for m, fn in BOOTSTRAP_METRICS.items():
                draws[name][m][b] = fn(y[idx], p[idx])
    return draws


def percentile_ci(values, level=0.95):
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def _axes(title, xlabel, ylabel, figsize=(5.2, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    return fig, ax


def plot_roc(y, p, title="ROC curve (test)"):
    fig, ax = _axes(title, "False positive rate", "True positive rate")
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.text(0.97, 0.05, f"AUC = {roc_auc_score(y, p):.3f}", ha="right", color=INK)
    return fig


def plot_pr(y, p, title="Precision-recall curve (test)"):
    fig, ax = _axes(title, "Recall", "Precision")
    prec, rec, _ = precision_recall_curve(y, p)
    ax.axhline(np.mean(y), ls="--", color=MUTED, lw=1)
    ax.text(0.02, np.mean(y) + 0.02, f"baseline = churn rate {np.mean(y):.2f}", color=MUTED, fontsize=8.5)
    ax.plot(rec, prec, color=BLUE, lw=2)
    ax.set_ylim(0, 1.02)
    ax.text(0.97, 0.95, f"PR-AUC = {average_precision_score(y, p):.3f}", ha="right", color=INK)
    return fig


def plot_calibration(y, curves, title="Calibration (test)", n_bins=10):
    """curves: {label: probabilities}. One curve -> no legend; two or more -> legend."""
    fig, ax = _axes(title, "Mean predicted probability", "Observed churn rate")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    for (label, p), color in zip(curves.items(), (BLUE, ORANGE)):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, lw=2, color=color, label=label)
    if len(curves) > 1:
        ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_confusion(y, p, threshold=0.5, title=None):
    cm = confusion_matrix(y, (p >= threshold).astype(int))
    fig, ax = _axes(title or f"Confusion matrix at threshold {threshold} (test)", "Predicted", "Actual",
                    figsize=(4.4, 4))
    ax.grid(False)
    ax.imshow(cm, cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    return fig
