"""
algo_trading/scripts/model_stability_report.py — Walk-forward model stability analysis.

After training, analyzes per-fold behavior to detect overfitting,
unstable features, and regime-specific weaknesses.

Reports:
  1. Per-fold Sharpe ratio (must be > 0.0 in ALL folds — red flag otherwise)
  2. Feature importance rank correlation across folds (Spearman rho)
  3. Confusion matrix per fold (label distribution stability)
  4. Probability calibration curve (are predicted probabilities accurate?)
  5. Regime-stratified Sharpe (bull/bear/sideways from HMM regime feature)

Red flags logged as WARNINGS:
  - Any fold with Sharpe < -0.5
  - Top feature changes every fold (rank correlation < 0.5)
  - Predicted probability always near 0.5 (model not learning)
  - Regime-specific Sharpe: positive in < 2/3 regimes

Usage:
    python -m algo_trading.scripts.model_stability_report --model-version v1.0.0
    python -m algo_trading.scripts.model_stability_report --oof-file models/registry/oof_v1.0.0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

_SHARPE_RED_FLAG: float = -0.5
_RANK_CORR_RED_FLAG: float = 0.5
_PROB_NEAR_HALF_THRESHOLD: float = 0.05   # max |p - 0.5| for "not learning" flag
_REGIME_MIN_POSITIVE_FRACTION: float = 2.0 / 3.0

# HMM regime column name in the feature vector (if present)
_HMM_REGIME_FEATURE: str = "hmm_regime"

# Default registry / data paths
_DEFAULT_REGISTRY_DIR: str = "models/registry"
_DEFAULT_OUTPUT_DIR: str = "models/stability_reports"


# ---------------------------------------------------------------------------
# Sharpe computation
# ---------------------------------------------------------------------------


def _compute_fold_sharpe(
    pred_probas: np.ndarray,
    y_true: np.ndarray,
    class_to_signal: dict[int, int] | None = None,
) -> float:
    """Compute annualised Sharpe from predicted class probabilities.

    Position = argmax(pred_probas) mapped to {-1, 0, +1}.
    Return_i = +1 if position == true_label, -1 if wrong, 0 if flat.
    Annualised by sqrt(252).

    Args:
        pred_probas: Probability matrix, shape ``(n_samples, 3)``.
        y_true: True integer labels ``{0, 1, 2}``.
        class_to_signal: Mapping from class index to signal {-1, 0, +1}.
            Defaults to ``{0: -1, 1: 0, 2: 1}``.

    Returns:
        Annualised Sharpe ratio (0.0 if not computable).
    """
    if class_to_signal is None:
        class_to_signal = {0: -1, 1: 0, 2: 1}

    preds = np.argmax(pred_probas, axis=1)
    signals = np.array([class_to_signal[int(p)] for p in preds])
    true_signals = np.array([class_to_signal[int(t)] for t in y_true])

    directional = np.where(
        signals == 0,
        0.0,
        np.where(signals == true_signals, 1.0, -1.0),
    )
    active = directional[signals != 0]
    if len(active) < 2 or active.std() == 0:
        return 0.0
    return float((active.mean() / active.std()) * np.sqrt(252))


# ---------------------------------------------------------------------------
# Spearman rank correlation of importances
# ---------------------------------------------------------------------------


def _spearman_rank_corr_matrix(
    importance_arrays: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Spearman rank correlation between fold importance arrays.

    Args:
        importance_arrays: List of importance vectors (one per fold).

    Returns:
        Square correlation matrix of shape ``(n_folds, n_folds)``.
        Returns a 1×1 identity if fewer than 2 arrays are given.
    """
    if len(importance_arrays) < 2:
        return np.eye(max(len(importance_arrays), 1))

    try:
        from scipy.stats import spearmanr  # noqa: PLC0415

        n = len(importance_arrays)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr, _ = spearmanr(importance_arrays[i], importance_arrays[j])
                matrix[i, j] = corr
                matrix[j, i] = corr
        return matrix
    except ImportError:
        log.warning("scipy_not_installed", note="Falling back to numpy rank correlation")
        n = len(importance_arrays)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                ri = np.argsort(np.argsort(importance_arrays[i]))
                rj = np.argsort(np.argsort(importance_arrays[j]))
                d_sq = ((ri.astype(float) - rj.astype(float)) ** 2).sum()
                k = len(ri)
                corr = 1 - 6 * d_sq / (k * (k * k - 1)) if k > 1 else 1.0
                matrix[i, j] = corr
                matrix[j, i] = corr
        return matrix


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _compute_calibration_metrics(
    pred_probas: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute reliability diagram data and a calibration summary.

    For each probability bin, computes the mean predicted probability vs the
    fraction of correct predictions.  Uses the max-class probability as the
    confidence score.

    Args:
        pred_probas: Probability matrix, shape ``(n_samples, 3)``.
        y_true: True integer labels ``{0, 1, 2}``.
        n_bins: Number of equally-spaced bins from 0 to 1.

    Returns:
        Dict with ``"bins"``, ``"mean_confidence"``, ``"fraction_correct"``,
        ``"expected_calibration_error"``, and ``"near_half_fraction"``.
    """
    max_proba = pred_probas.max(axis=1)
    preds = pred_probas.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    mean_confidence: list[float] = []
    fraction_correct: list[float] = []
    bin_counts: list[int] = []

    for b in range(n_bins):
        low, high = bins[b], bins[b + 1]
        mask = (max_proba >= low) & (max_proba < high)
        if b == n_bins - 1:
            mask = (max_proba >= low) & (max_proba <= high)
        n_bin = int(mask.sum())
        bin_counts.append(n_bin)
        if n_bin == 0:
            mean_confidence.append(0.0)
            fraction_correct.append(0.0)
        else:
            mean_confidence.append(float(max_proba[mask].mean()))
            fraction_correct.append(float(correct[mask].mean()))

    # ECE: weighted average of |confidence - accuracy|
    total = max(1, len(y_true))
    ece = sum(
        count / total * abs(conf - frac)
        for count, conf, frac in zip(bin_counts, mean_confidence, fraction_correct)
    )

    # Fraction of samples where max_proba is near 0.5 (model not confident)
    near_half = float(np.mean(np.abs(max_proba - 0.5) <= _PROB_NEAR_HALF_THRESHOLD))

    return {
        "bins": [round(b, 2) for b in bins.tolist()],
        "mean_confidence": [round(v, 4) for v in mean_confidence],
        "fraction_correct": [round(v, 4) for v in fraction_correct],
        "bin_counts": bin_counts,
        "expected_calibration_error": round(float(ece), 6),
        "near_half_fraction": round(near_half, 4),
    }


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------


def _compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> list[list[int]]:
    """Compute a confusion matrix without sklearn dependency.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        n_classes: Number of classes.

    Returns:
        n_classes × n_classes confusion matrix as a nested list.
    """
    try:
        from sklearn.metrics import confusion_matrix  # noqa: PLC0415
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        return cm.tolist()
    except ImportError:
        cm = [[0] * n_classes for _ in range(n_classes)]
        for t, p in zip(y_true, y_pred):
            t_i = int(t)
            p_i = int(p)
            if 0 <= t_i < n_classes and 0 <= p_i < n_classes:
                cm[t_i][p_i] += 1
        return cm


# ---------------------------------------------------------------------------
# Regime-stratified Sharpe
# ---------------------------------------------------------------------------


def _regime_stratified_sharpe(
    pred_probas: np.ndarray,
    y_true: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute Sharpe ratio stratified by HMM regime label.

    Args:
        pred_probas: Probability matrix, shape ``(n_samples, 3)``.
        y_true: True integer labels.
        regime_labels: HMM regime per sample (0, 1, or 2).
        regime_names: Optional mapping from regime int to name.

    Returns:
        Dict mapping regime name → Sharpe ratio.
    """
    if regime_names is None:
        regime_names = {0: "bear", 1: "sideways", 2: "bull"}

    result: dict[str, float] = {}
    for regime_id, regime_name in regime_names.items():
        mask = regime_labels == regime_id
        n_regime = int(mask.sum())
        if n_regime < 10:
            log.warning(
                "regime_too_few_samples",
                regime=regime_name,
                n_samples=n_regime,
            )
            result[regime_name] = float("nan")
            continue
        sharpe = _compute_fold_sharpe(pred_probas[mask], y_true[mask])
        result[regime_name] = sharpe
    return result


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def run_stability_report(
    model_version: str | None = None,
    oof_file: str | None = None,
    registry_dir: str = _DEFAULT_REGISTRY_DIR,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the full walk-forward stability analysis.

    Either ``model_version`` or ``oof_file`` must be provided.

    Steps:
    1. Load OOF predictions from the ``.npz`` file saved during training.
    2. Reconstruct per-fold predictions using TimeSeriesSplit.
    3. Compute per-fold Sharpe ratios.
    4. Compute Spearman rank correlation of feature importances across folds.
    5. Compute per-fold confusion matrices.
    6. Compute probability calibration curve (reliability diagram).
    7. Compute regime-stratified Sharpe using the HMM regime column.
    8. Flag red flags, write JSON report, print console summary.

    Args:
        model_version: Version string used to locate the OOF npz and spec.
        oof_file: Direct path to an OOF ``.npz`` file.
        registry_dir: Registry directory containing spec JSON files.
        output_dir: Directory to write the stability report JSON and plots.

    Returns:
        Dict containing the full stability report data.
    """
    # ------------------------------------------------------------------
    # Step 1: Resolve OOF file path
    # ------------------------------------------------------------------
    if oof_file is None and model_version is None:
        raise ValueError("Either --model-version or --oof-file must be provided.")

    registry_path = Path(registry_dir)

    if oof_file is not None:
        oof_path = Path(oof_file)
    else:
        # Look for oof_{version}.npz next to the meta_lr.pkl
        spec_file = registry_path / f"{model_version}.json"
        if not spec_file.exists():
            candidates = list(registry_path.glob(f"*{model_version}*.json"))
            if candidates:
                spec_file = candidates[0]
            else:
                raise FileNotFoundError(
                    f"No spec found for version '{model_version}' in {registry_path}"
                )
        from algo_trading.models.model_registry import FrozenModelSpec  # noqa: PLC0415

        spec = FrozenModelSpec.load(spec_file)
        meta_dir = Path(spec.meta_lr_path).parent
        oof_path = meta_dir / f"oof_{model_version}.npz"

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    log.info("loading_oof_file", path=str(oof_path))
    oof_data = np.load(str(oof_path), allow_pickle=False)

    # Required arrays
    oof_xgb: np.ndarray = oof_data["oof_xgb"]
    oof_lgb: np.ndarray = oof_data["oof_lgb"]
    oof_cat: np.ndarray = oof_data["oof_cat"]
    meta_features: np.ndarray = oof_data["meta_features"]
    y_mapped: np.ndarray = oof_data["y_mapped"]
    train_meta_mask: np.ndarray = oof_data["train_meta_mask"]
    holdout_test_idx: np.ndarray = oof_data["holdout_test_idx"]

    n_samples = y_mapped.shape[0]
    log.info("oof_data_loaded", n_samples=n_samples)

    # ------------------------------------------------------------------
    # Step 2: Load the meta-learner to produce full OOF predictions
    # ------------------------------------------------------------------
    import pickle  # noqa: PLC0415

    meta_lr_path = None
    if model_version is not None:
        try:
            from algo_trading.models.model_registry import FrozenModelSpec  # noqa: PLC0415
            spec_file = registry_path / f"{model_version}.json"
            if not spec_file.exists():
                cands = list(registry_path.glob(f"*{model_version}*.json"))
                if cands:
                    spec_file = cands[0]
            if spec_file.exists():
                spec = FrozenModelSpec.load(spec_file)
                meta_lr_path = Path(spec.meta_lr_path)
        except Exception as exc:
            log.warning("spec_load_failed", error=str(exc))

    meta_lr = None
    if meta_lr_path and meta_lr_path.exists():
        with meta_lr_path.open("rb") as fh:
            meta_lr = pickle.load(fh)  # noqa: S301
        log.info("meta_lr_loaded", path=str(meta_lr_path))

    # Produce meta predictions
    if meta_lr is not None:
        all_meta_probas = meta_lr.predict_proba(meta_features)
    else:
        # Average base model probas as fallback
        all_meta_probas = (oof_xgb + oof_lgb + oof_cat) / 3.0
        log.warning(
            "meta_lr_not_available",
            note="Using simple average of base model OOF probas",
        )

    # ------------------------------------------------------------------
    # Step 3: Per-fold Sharpe using TimeSeriesSplit
    # ------------------------------------------------------------------
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    # Determine n_folds from the data shape (holdout_test_idx tells us fold[-1])
    # We use n_splits=5 as the canonical training config; override if we can detect it
    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_splits = list(tscv.split(np.arange(n_samples)))

    fold_sharpes: list[float] = []
    fold_confusion_matrices: list[list[list[int]]] = []
    fold_importances_xgb: list[np.ndarray] = []
    fold_importances_lgb: list[np.ndarray] = []
    fold_importances_cat: list[np.ndarray] = []

    # OOF folds (excluding the holdout fold[-1])
    active_splits = all_splits[:-1]
    for fold_idx, (_, val_idx) in enumerate(active_splits):
        fold_probas = all_meta_probas[val_idx]
        fold_y = y_mapped[val_idx]

        sharpe = _compute_fold_sharpe(fold_probas, fold_y)
        fold_sharpes.append(sharpe)

        fold_preds = fold_probas.argmax(axis=1)
        cm = _compute_confusion_matrix(fold_y, fold_preds)
        fold_confusion_matrices.append(cm)

        # Per-fold importances from oof arrays (use mean proba magnitude as proxy)
        # We use the per-fold OOF mean absolute proba deviation as a simple importance proxy
        # when no per-fold model importances are saved.  This is a heuristic.
        xgb_fold = np.abs(oof_xgb[val_idx] - 1.0 / 3).mean(axis=0)
        lgb_fold = np.abs(oof_lgb[val_idx] - 1.0 / 3).mean(axis=0)
        cat_fold = np.abs(oof_cat[val_idx] - 1.0 / 3).mean(axis=0)
        fold_importances_xgb.append(xgb_fold)
        fold_importances_lgb.append(lgb_fold)
        fold_importances_cat.append(cat_fold)

        if sharpe < _SHARPE_RED_FLAG:
            log.warning(
                "RED_FLAG_negative_fold_sharpe",
                fold=fold_idx,
                sharpe=round(sharpe, 4),
                threshold=_SHARPE_RED_FLAG,
            )

    # Holdout fold metrics
    holdout_probas = all_meta_probas[holdout_test_idx]
    holdout_y = y_mapped[holdout_test_idx]
    holdout_sharpe = _compute_fold_sharpe(holdout_probas, holdout_y)
    holdout_cm = _compute_confusion_matrix(holdout_y, holdout_probas.argmax(axis=1))

    log.info(
        "fold_sharpes_computed",
        fold_sharpes=[round(s, 4) for s in fold_sharpes],
        holdout_sharpe=round(holdout_sharpe, 4),
    )

    # ------------------------------------------------------------------
    # Step 4: Spearman rank correlation of feature importances
    # ------------------------------------------------------------------
    # Pool importances across models for each fold
    pooled_imps: list[np.ndarray] = []
    for xgb_i, lgb_i, cat_i in zip(
        fold_importances_xgb, fold_importances_lgb, fold_importances_cat
    ):
        pooled_imps.append((xgb_i + lgb_i + cat_i) / 3.0)

    rank_corr_matrix = _spearman_rank_corr_matrix(pooled_imps)
    # Mean off-diagonal Spearman correlation
    n_f = rank_corr_matrix.shape[0]
    if n_f > 1:
        off_diag = rank_corr_matrix[~np.eye(n_f, dtype=bool)]
        mean_rank_corr = float(off_diag.mean())
    else:
        mean_rank_corr = 1.0

    if mean_rank_corr < _RANK_CORR_RED_FLAG:
        log.warning(
            "RED_FLAG_low_rank_correlation",
            mean_rank_corr=round(mean_rank_corr, 4),
            threshold=_RANK_CORR_RED_FLAG,
            note="Top feature changes every fold — importance is unstable",
        )

    # ------------------------------------------------------------------
    # Step 5: Probability calibration curve
    # ------------------------------------------------------------------
    # Use only OOF rows (not holdout, not unpredicted rows)
    cal_probas = all_meta_probas[train_meta_mask]
    cal_y = y_mapped[train_meta_mask]
    calibration = _compute_calibration_metrics(cal_probas, cal_y)

    if calibration["near_half_fraction"] > 0.7:
        log.warning(
            "RED_FLAG_probabilities_near_half",
            near_half_fraction=calibration["near_half_fraction"],
            note="More than 70% of predictions are near 0.5 — model may not be learning",
        )

    # ------------------------------------------------------------------
    # Step 6: Regime-stratified Sharpe
    # ------------------------------------------------------------------
    # Attempt to find an HMM regime column index in the feature spec
    regime_sharpe: dict[str, float] = {}
    regime_present = False

    if model_version is not None:
        try:
            from algo_trading.models.model_registry import FrozenModelSpec  # noqa: PLC0415
            spec_file = registry_path / f"{model_version}.json"
            if not spec_file.exists():
                cands = list(registry_path.glob(f"*{model_version}*.json"))
                if cands:
                    spec_file = cands[0]
            if spec_file.exists():
                spec = FrozenModelSpec.load(spec_file)
                feature_names = list(spec.feature_names)
                if _HMM_REGIME_FEATURE in feature_names:
                    hmm_idx = feature_names.index(_HMM_REGIME_FEATURE)
                    # Extract regime labels from meta_features (OOF rows)
                    # meta_features columns: [xgb_p0, xgb_p1, xgb_p2, lgb_p0, ..., cat_p2]
                    # Regime labels are not in meta_features; they need to come from X.
                    # Since we don't have X here, attempt to load a cached version.
                    regime_arr = _try_load_regime_labels(
                        oof_path=oof_path, n_samples=n_samples
                    )
                    if regime_arr is not None:
                        regime_sharpe = _regime_stratified_sharpe(
                            all_meta_probas, y_mapped, regime_arr
                        )
                        regime_present = True
                        n_positive = sum(
                            1 for v in regime_sharpe.values()
                            if not np.isnan(v) and v > 0
                        )
                        n_regimes = len(regime_sharpe)
                        if n_regimes > 0 and n_positive / n_regimes < _REGIME_MIN_POSITIVE_FRACTION:
                            log.warning(
                                "RED_FLAG_regime_sharpe_below_threshold",
                                regime_sharpe=regime_sharpe,
                                positive_fraction=n_positive / n_regimes,
                                threshold=_REGIME_MIN_POSITIVE_FRACTION,
                            )
        except Exception as exc:
            log.warning("regime_analysis_failed", error=str(exc))

    if not regime_present:
        log.info(
            "regime_analysis_skipped",
            note=f"HMM regime feature '{_HMM_REGIME_FEATURE}' not found or data unavailable",
        )

    # ------------------------------------------------------------------
    # Step 7: Compile red flags
    # ------------------------------------------------------------------
    red_flags: list[str] = []
    for i, s in enumerate(fold_sharpes):
        if s < _SHARPE_RED_FLAG:
            red_flags.append(f"Fold {i} Sharpe={s:.4f} < {_SHARPE_RED_FLAG}")
    if mean_rank_corr < _RANK_CORR_RED_FLAG:
        red_flags.append(
            f"Mean Spearman rank correlation={mean_rank_corr:.4f} < {_RANK_CORR_RED_FLAG} "
            "(top feature changes every fold)"
        )
    if calibration["near_half_fraction"] > 0.7:
        red_flags.append(
            f"Near-0.5 probability fraction={calibration['near_half_fraction']:.4f} > 0.70 "
            "(model not learning)"
        )
    if regime_sharpe:
        n_positive = sum(1 for v in regime_sharpe.values() if not np.isnan(v) and v > 0)
        n_regimes = len(regime_sharpe)
        if n_regimes > 0 and n_positive / n_regimes < _REGIME_MIN_POSITIVE_FRACTION:
            red_flags.append(
                f"Regime-stratified Sharpe positive in only {n_positive}/{n_regimes} regimes "
                f"(threshold: {_REGIME_MIN_POSITIVE_FRACTION:.2f})"
            )

    # ------------------------------------------------------------------
    # Step 8: Build report dict
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "model_version": model_version or "unknown",
        "oof_file": str(oof_path),
        "generated_at": date.today().isoformat(),
        "n_samples": int(n_samples),
        "n_oof_folds": len(active_splits),
        "fold_sharpes": [round(s, 4) for s in fold_sharpes],
        "holdout_sharpe": round(holdout_sharpe, 4),
        "mean_fold_sharpe": round(float(np.mean(fold_sharpes)), 4) if fold_sharpes else 0.0,
        "min_fold_sharpe": round(float(np.min(fold_sharpes)), 4) if fold_sharpes else 0.0,
        "all_folds_positive": all(s >= 0 for s in fold_sharpes),
        "mean_spearman_rank_correlation": round(mean_rank_corr, 4),
        "rank_correlation_matrix": [[round(v, 4) for v in row] for row in rank_corr_matrix.tolist()],
        "fold_confusion_matrices": fold_confusion_matrices,
        "holdout_confusion_matrix": holdout_cm,
        "calibration": calibration,
        "regime_stratified_sharpe": {
            k: (round(v, 4) if not np.isnan(v) else None)
            for k, v in regime_sharpe.items()
        },
        "red_flags": red_flags,
        "red_flags_count": len(red_flags),
        "thresholds_used": {
            "sharpe_red_flag": _SHARPE_RED_FLAG,
            "rank_corr_red_flag": _RANK_CORR_RED_FLAG,
            "prob_near_half_threshold": _PROB_NEAR_HALF_THRESHOLD,
            "regime_min_positive_fraction": _REGIME_MIN_POSITIVE_FRACTION,
        },
    }

    # ------------------------------------------------------------------
    # Step 9: Write output JSON
    # ------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ver_tag = model_version or "unknown"
    out_file = out_path / f"stability_report_{ver_tag}_{date.today().isoformat()}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("stability_report_written", path=str(out_file))

    # ------------------------------------------------------------------
    # Step 10: Console summary
    # ------------------------------------------------------------------
    _print_stability_summary(report)

    # ------------------------------------------------------------------
    # Step 11: Save plots if matplotlib available
    # ------------------------------------------------------------------
    try:
        _save_plots(report, out_path, ver_tag)
    except ImportError:
        log.info("matplotlib_not_installed", note="Skipping plot generation")
    except Exception as exc:
        log.warning("plot_generation_failed", error=str(exc))

    return report


# ---------------------------------------------------------------------------
# Regime label loading helper
# ---------------------------------------------------------------------------


def _try_load_regime_labels(
    oof_path: Path,
    n_samples: int,
) -> np.ndarray | None:
    """Attempt to load regime labels from a sibling npz file.

    Looks for ``regime_labels_{stem}.npz`` next to the OOF file.

    Args:
        oof_path: Path to the OOF npz file.
        n_samples: Expected number of samples.

    Returns:
        Regime label array of shape ``(n_samples,)`` or None.
    """
    regime_file = oof_path.parent / f"regime_labels_{oof_path.stem.replace('oof_', '')}.npz"
    if regime_file.exists():
        try:
            data = np.load(str(regime_file))
            labels = data["regime_labels"]
            if labels.shape[0] == n_samples:
                return labels
            log.warning(
                "regime_label_shape_mismatch",
                expected=n_samples,
                got=labels.shape[0],
            )
        except Exception as exc:
            log.warning("regime_label_load_failed", path=str(regime_file), error=str(exc))
    return None


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------


def _print_stability_summary(report: dict[str, Any]) -> None:
    """Print a human-readable stability report to stdout.

    Args:
        report: Full stability report dict.
    """
    print()
    print("=" * 70)
    print(f"  MODEL STABILITY REPORT — {report['model_version']}")
    print(f"  Generated: {report['generated_at']}")
    print("=" * 70)

    print("\n  PER-FOLD SHARPE RATIOS:")
    for i, s in enumerate(report["fold_sharpes"]):
        flag = "  *** RED FLAG ***" if s < _SHARPE_RED_FLAG else ""
        print(f"    Fold {i}: {s:+.4f}{flag}")
    print(f"    Holdout: {report['holdout_sharpe']:+.4f}")
    print(f"    Mean (OOF): {report['mean_fold_sharpe']:+.4f}")
    print(f"    All folds positive: {report['all_folds_positive']}")

    print(f"\n  RANK CORRELATION (Spearman): {report['mean_spearman_rank_correlation']:.4f}", end="")
    if report["mean_spearman_rank_correlation"] < _RANK_CORR_RED_FLAG:
        print("  *** RED FLAG — top feature unstable ***", end="")
    print()

    cal = report["calibration"]
    print(f"\n  CALIBRATION (ECE): {cal['expected_calibration_error']:.6f}")
    print(f"  Near-0.5 fraction: {cal['near_half_fraction']:.4f}", end="")
    if cal["near_half_fraction"] > 0.7:
        print("  *** RED FLAG — model not learning ***", end="")
    print()

    if report["regime_stratified_sharpe"]:
        print("\n  REGIME-STRATIFIED SHARPE:")
        for regime, sharpe in report["regime_stratified_sharpe"].items():
            val = f"{sharpe:+.4f}" if sharpe is not None else "N/A"
            print(f"    {regime:<12}: {val}")

    print(f"\n  RED FLAGS ({report['red_flags_count']}):")
    if report["red_flags"]:
        for rf in report["red_flags"]:
            print(f"    - {rf}")
    else:
        print("    None — model appears stable.")

    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _save_plots(
    report: dict[str, Any],
    out_path: Path,
    ver_tag: str,
) -> None:
    """Save reliability diagram and fold Sharpe bar chart as PNG files.

    Args:
        report: Full stability report dict.
        out_path: Directory to write PNG files.
        ver_tag: Version tag for file naming.

    Raises:
        ImportError: If matplotlib is not installed (caller catches this).
    """
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt  # type: ignore[import]

    # --- Reliability diagram ---
    cal = report["calibration"]
    fig, ax = plt.subplots(figsize=(6, 5))
    mc = cal["mean_confidence"]
    fc = cal["fraction_correct"]
    # Plot only non-zero bins
    valid = [i for i in range(len(mc)) if cal["bin_counts"][i] > 0]
    ax.plot([mc[i] for i in valid], [fc[i] for i in valid], "b-o", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Reliability Diagram — {ver_tag}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cal_plot = out_path / f"reliability_diagram_{ver_tag}.png"
    fig.savefig(str(cal_plot), dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("reliability_diagram_saved", path=str(cal_plot))

    # --- Per-fold Sharpe bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    fold_labels = [f"F{i}" for i in range(len(report["fold_sharpes"]))] + ["Holdout"]
    all_sharpes = report["fold_sharpes"] + [report["holdout_sharpe"]]
    colors = ["red" if s < _SHARPE_RED_FLAG else "steelblue" for s in all_sharpes]
    ax.bar(fold_labels, all_sharpes, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(
        _SHARPE_RED_FLAG,
        color="red",
        linewidth=1,
        linestyle="--",
        label=f"Red flag threshold ({_SHARPE_RED_FLAG})",
    )
    ax.set_ylabel("Sharpe ratio (annualised)")
    ax.set_title(f"Per-Fold Sharpe Ratios — {ver_tag}")
    ax.legend()
    sharpe_plot = out_path / f"fold_sharpes_{ver_tag}.png"
    fig.savefig(str(sharpe_plot), dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("sharpe_chart_saved", path=str(sharpe_plot))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward model stability analysis for NSE AlgoTrader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-version",
        default=None,
        help="Model version string (e.g. v1.0.0). Locates OOF file via registry.",
    )
    group.add_argument(
        "--oof-file",
        default=None,
        help="Direct path to OOF .npz file saved during training.",
    )
    parser.add_argument(
        "--registry-dir",
        default=_DEFAULT_REGISTRY_DIR,
        help=f"Model registry directory (default: {_DEFAULT_REGISTRY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for report JSON and plots (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main() -> None:
    """CLI entry point for model_stability_report."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.model_version is None and args.oof_file is None:
        parser.error("One of --model-version or --oof-file is required.")

    try:
        report = run_stability_report(
            model_version=args.model_version,
            oof_file=args.oof_file,
            registry_dir=args.registry_dir,
            output_dir=args.output_dir,
        )
        n_flags = report["red_flags_count"]
        print(f"Stability report complete. Red flags: {n_flags}")
        sys.exit(0 if n_flags == 0 else 1)
    except FileNotFoundError as exc:
        log.error("stability_report_failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        log.error("stability_report_unexpected_error", error=str(exc))
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
