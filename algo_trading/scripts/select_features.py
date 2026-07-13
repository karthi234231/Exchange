"""
algo_trading/scripts/select_features.py — SHAP-guided feature selection.

Loads a trained model from models/registry/, computes SHAP values on
walk-forward OOF predictions, and identifies features to consider removing.

Selection criteria:
  1. Mean |SHAP| < 0.001 across all samples → candidate for removal
  2. Feature importance rank stability < 0.5 (std > 0.5 × mean across folds)
  3. Pairwise Pearson correlation > 0.90 with a higher-importance feature → redundant

Output:
  - data/feature_selection_{version}_{date}.json with:
      - keep: list of features to keep
      - remove_candidates: list with reason
      - correlation_pairs: pairs with correlation > 0.90
  - Console table: feature | mean_shap | rank_std | correlated_with | recommendation

Usage:
    python -m algo_trading.scripts.select_features --model-version v1.0.0 --data-dir data/historical
    python -m algo_trading.scripts.select_features --quick  # use saved SHAP values

Requires: shap, a trained model in models/registry/
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

_SHAP_LOW_THRESHOLD: float = 0.001
_CORR_HIGH_THRESHOLD: float = 0.90
_INSTABILITY_RATIO: float = 0.5
_DEFAULT_REGISTRY_DIR: str = "models/registry"
_DEFAULT_DATA_DIR: str = "data/historical"
_DEFAULT_OUTPUT_DIR: str = "data"
_N_FOLDS_STABILITY: int = 5


# ---------------------------------------------------------------------------
# SHAP computation helpers
# ---------------------------------------------------------------------------


def _compute_shap_values_for_model(
    model: Any,
    X: np.ndarray,
    model_name: str,
) -> np.ndarray | None:
    """Compute per-sample SHAP values for a single tree-based model.

    Uses ``shap.TreeExplainer`` for XGBoost, LightGBM, and CatBoost.
    Falls back to ``model.feature_importances_`` or
    ``model.get_feature_importance()`` if shap is not installed.

    Args:
        model: Fitted tree model (XGB, LGB, or CatBoost instance).
        X: Feature matrix of shape ``(n_samples, n_features)``.
        model_name: Label for logging (``"xgb"``, ``"lgb"``, ``"cat"``).

    Returns:
        Array of mean absolute SHAP values, shape ``(n_features,)``,
        or None if computation fails.
    """
    try:
        import shap  # type: ignore[import]

        explainer = shap.TreeExplainer(model)
        # shap_values shape: (n_samples, n_features, n_classes) for multi-class
        raw_shap = explainer.shap_values(X)
        if isinstance(raw_shap, list):
            # Old shap API: list of (n_samples, n_features) arrays, one per class
            abs_shap = np.mean([np.abs(s) for s in raw_shap], axis=0)
        else:
            # New shap API: (n_samples, n_features, n_classes) or (n_samples, n_features)
            if raw_shap.ndim == 3:
                abs_shap = np.mean(np.abs(raw_shap), axis=(0, 2))
            else:
                abs_shap = np.mean(np.abs(raw_shap), axis=0)
        mean_abs_shap = abs_shap.mean(axis=0) if abs_shap.ndim == 2 else abs_shap
        log.info(
            "shap_computed",
            model=model_name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
        return mean_abs_shap

    except ImportError:
        log.warning(
            "shap_not_installed",
            model=model_name,
            note="Falling back to native feature_importances_",
        )
        return _fallback_importance(model, model_name)
    except Exception as exc:
        log.warning(
            "shap_computation_failed",
            model=model_name,
            error=str(exc),
            note="Falling back to native feature_importances_",
        )
        return _fallback_importance(model, model_name)


def _fallback_importance(model: Any, model_name: str) -> np.ndarray | None:
    """Extract native feature importances when SHAP is unavailable.

    Args:
        model: Fitted tree model.
        model_name: Label for logging.

    Returns:
        Normalised importance array or None on failure.
    """
    try:
        if hasattr(model, "feature_importances_"):
            raw = np.array(model.feature_importances_, dtype=np.float64)
        elif hasattr(model, "get_feature_importance"):
            raw = np.array(model.get_feature_importance(), dtype=np.float64)
        else:
            log.warning("no_feature_importance_available", model=model_name)
            return None
        total = raw.sum()
        return raw / total if total > 0 else raw
    except Exception as exc:
        log.warning("fallback_importance_failed", model=model_name, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Rank stability across folds
# ---------------------------------------------------------------------------


def _compute_rank_stability(
    fold_importance_arrays: list[np.ndarray],
) -> np.ndarray:
    """Compute rank standard deviation across OOF folds for each feature.

    For each fold's importance array, convert to rank (1 = most important).
    The output is the standard deviation of rank across folds per feature.
    High rank_std means the feature's relative importance is unstable.

    Args:
        fold_importance_arrays: List of normalised importance arrays,
            one per fold. Each array has shape ``(n_features,)``.

    Returns:
        Array of rank standard deviations, shape ``(n_features,)``.
        Returns zeros if fewer than 2 folds are provided.
    """
    if len(fold_importance_arrays) < 2:
        if fold_importance_arrays:
            return np.zeros(fold_importance_arrays[0].shape[0])
        return np.zeros(0)

    n_features = fold_importance_arrays[0].shape[0]
    rank_matrix = np.zeros((len(fold_importance_arrays), n_features))
    for i, imp in enumerate(fold_importance_arrays):
        # argsort descending → ranks; add 1 so rank 1 = most important
        order = np.argsort(imp)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n_features + 1)
        rank_matrix[i] = ranks
    return rank_matrix.std(axis=0)


def _recompute_fold_importances(
    model: Any,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> list[np.ndarray]:
    """Re-run walk-forward importance extraction on each fold.

    Trains a fresh copy of the given model on each fold's training slice and
    extracts feature importances.  Used when saved fold importances are not
    available.

    Args:
        model: An unfitted model instance with a ``fit`` and
            ``feature_importances_`` / ``get_feature_importance`` interface.
            The model is cloned using ``sklearn.base.clone`` or re-instantiated
            from its ``get_params()`` if that fails.
        model_name: Label for logging.
        X: Full feature matrix (chronological).
        y: Full label array.
        n_folds: Number of folds to use.

    Returns:
        List of normalised importance arrays, one per fold.
    """
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_imps: list[np.ndarray] = []

    for fold_idx, (train_idx, _) in enumerate(tscv.split(X)):
        try:
            # Clone the model using sklearn if possible, else pickle round-trip
            try:
                from sklearn.base import clone as sklearn_clone  # noqa: PLC0415
                m = sklearn_clone(model)
            except Exception:
                import pickle  # noqa: PLC0415
                m = pickle.loads(pickle.dumps(model))

            m.fit(X[train_idx], y[train_idx])
            imp = _fallback_importance(m, model_name)
            if imp is not None:
                fold_imps.append(imp)
        except Exception as exc:
            log.warning(
                "fold_importance_recompute_failed",
                model=model_name,
                fold=fold_idx,
                error=str(exc),
            )
    return fold_imps


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def _find_correlated_pairs(
    X: np.ndarray,
    feature_names: list[str],
    mean_shap: np.ndarray,
    threshold: float = _CORR_HIGH_THRESHOLD,
) -> list[dict[str, Any]]:
    """Find pairs of features with Pearson correlation above threshold.

    For each correlated pair, the feature with lower mean SHAP importance is
    flagged as redundant.

    Args:
        X: Feature matrix, shape ``(n_samples, n_features)``.
        feature_names: List of feature names.
        mean_shap: Mean absolute SHAP values per feature.
        threshold: Correlation magnitude threshold (default: 0.90).

    Returns:
        List of dicts with keys ``"feature_a"``, ``"feature_b"``,
        ``"correlation"``, ``"redundant_feature"`` (the lower-SHAP one).
    """
    n_features = len(feature_names)
    if X.shape[1] != n_features:
        log.warning(
            "correlation_shape_mismatch",
            X_cols=X.shape[1],
            n_feature_names=n_features,
        )
        n_features = min(X.shape[1], n_features)

    corr_matrix = np.corrcoef(X[:, :n_features].T)
    pairs: list[dict[str, Any]] = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = float(corr_matrix[i, j])
            if abs(corr_val) >= threshold:
                # The feature with lower SHAP is the redundant one
                if mean_shap[i] >= mean_shap[j]:
                    redundant = feature_names[j]
                else:
                    redundant = feature_names[i]
                pairs.append(
                    {
                        "feature_a": feature_names[i],
                        "feature_b": feature_names[j],
                        "correlation": round(corr_val, 4),
                        "redundant_feature": redundant,
                    }
                )
    return pairs


# ---------------------------------------------------------------------------
# Main selection logic
# ---------------------------------------------------------------------------


def run_feature_selection(
    model_version: str,
    data_dir: str = _DEFAULT_DATA_DIR,
    registry_dir: str = _DEFAULT_REGISTRY_DIR,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    quick: bool = False,
    n_shap_samples: int = 2000,
) -> dict[str, Any]:
    """Run the full SHAP-guided feature selection pipeline.

    Steps:
    1. Load model spec from registry and unpickle base models.
    2. Load historical data and compute the feature matrix.
    3. If not ``quick``, compute SHAP values on each base model.
    4. Average SHAP values across XGB, LGB, CatBoost.
    5. Compute rank stability across walk-forward folds.
    6. Compute pairwise Pearson correlation matrix.
    7. Apply selection criteria and produce output JSON + console table.

    Args:
        model_version: Version string, e.g. ``"v1.0.0"``.
        data_dir: Directory containing historical Parquet files.
        registry_dir: Directory containing model spec JSON files.
        output_dir: Directory to write the output JSON file.
        quick: If True, skip SHAP recomputation and use saved OOF SHAP
            values if available; otherwise use native importances.
        n_shap_samples: Maximum number of rows to pass to SHAP explainer
            (random subsample to keep computation tractable).

    Returns:
        Dict with ``"keep"``, ``"remove_candidates"``,
        ``"correlation_pairs"``, and ``"summary"`` keys.
    """
    # ------------------------------------------------------------------
    # Step 1: Load model spec
    # ------------------------------------------------------------------
    registry_path = Path(registry_dir)
    if not registry_path.exists():
        log.error("registry_not_found", path=str(registry_path))
        raise FileNotFoundError(f"Registry directory not found: {registry_path}")

    from algo_trading.models.model_registry import FrozenModelSpec  # noqa: PLC0415

    spec_file = registry_path / f"{model_version}.json"
    if not spec_file.exists():
        # Try alternate naming (version without prefix)
        candidates = list(registry_path.glob(f"*{model_version}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No spec file found for version '{model_version}' in {registry_path}"
            )
        spec_file = candidates[0]

    spec = FrozenModelSpec.load(spec_file)
    feature_names = list(spec.feature_names)
    n_features = len(feature_names)
    log.info("spec_loaded", version=spec.version, n_features=n_features)

    # ------------------------------------------------------------------
    # Step 2: Load base models from pickle
    # ------------------------------------------------------------------
    import pickle  # noqa: PLC0415

    models: dict[str, Any] = {}
    model_paths = {
        "xgb": Path(spec.treelite_xgb_path).parent / "xgb_model.pkl",
        "lgb": Path(spec.treelite_lgb_path).parent / "lgb_model.pkl",
        "cat": Path(spec.treelite_cat_path).parent / "cat_model.pkl",
    }
    for model_name, pkl_path in model_paths.items():
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                models[model_name] = pickle.load(fh)  # noqa: S301
            log.info("model_loaded", model=model_name, path=str(pkl_path))
        else:
            log.warning(
                "model_pkl_not_found",
                model=model_name,
                path=str(pkl_path),
                note="Will skip SHAP for this model",
            )

    # ------------------------------------------------------------------
    # Step 3: Load historical data and build feature matrix
    # ------------------------------------------------------------------
    X, y, loaded_feature_names = _load_feature_matrix(
        data_dir=data_dir,
        feature_names=feature_names,
    )
    if X is None or X.shape[0] == 0:
        log.error("no_data_loaded", data_dir=data_dir)
        raise RuntimeError(f"No data could be loaded from {data_dir}")

    # Subsample for SHAP computation efficiency
    if X.shape[0] > n_shap_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=n_shap_samples, replace=False)
        idx.sort()
        X_shap = X[idx]
        y_shap = y[idx] if y is not None else None
    else:
        X_shap = X
        y_shap = y

    log.info("data_loaded", n_rows=X.shape[0], n_shap_rows=X_shap.shape[0])

    # ------------------------------------------------------------------
    # Step 4: Compute SHAP values (or quick fallback)
    # ------------------------------------------------------------------
    shap_arrays: list[np.ndarray] = []
    fold_importance_arrays_by_model: dict[str, list[np.ndarray]] = {}

    # Check for saved OOF file when quick mode
    saved_oof_path = Path(spec.meta_lr_path).parent / f"oof_{model_version}.npz"
    if quick and saved_oof_path.exists():
        log.info("quick_mode_oof_loaded", path=str(saved_oof_path))
        # Use native importances only — no SHAP recompute
        for model_name, model in models.items():
            imp = _fallback_importance(model, model_name)
            if imp is not None:
                shap_arrays.append(imp)
                fold_importance_arrays_by_model[model_name] = [imp]
    else:
        for model_name, model in models.items():
            shap_imp = _compute_shap_values_for_model(model, X_shap, model_name)
            if shap_imp is not None:
                shap_arrays.append(shap_imp)

            # Recompute per-fold importances for stability (only when not quick)
            if not quick and y_shap is not None:
                fold_imps = _recompute_fold_importances(
                    model, model_name, X, y, n_folds=_N_FOLDS_STABILITY
                )
                fold_importance_arrays_by_model[model_name] = fold_imps
            else:
                if model in models.values():
                    native_imp = _fallback_importance(model, model_name)
                    fold_importance_arrays_by_model[model_name] = (
                        [native_imp] if native_imp is not None else []
                    )

    if not shap_arrays:
        log.error("no_shap_values_computed")
        raise RuntimeError(
            "Could not compute any SHAP or importance values. "
            "Ensure models are available and data is valid."
        )

    # Average SHAP values across available models
    mean_shap = np.mean(np.vstack(shap_arrays), axis=0)
    log.info("shap_averaged", n_models=len(shap_arrays), n_features=len(mean_shap))

    # ------------------------------------------------------------------
    # Step 5: Compute rank stability
    # ------------------------------------------------------------------
    all_fold_imps: list[np.ndarray] = []
    for imps in fold_importance_arrays_by_model.values():
        all_fold_imps.extend(imps)

    rank_stds = _compute_rank_stability(all_fold_imps)
    if rank_stds.shape[0] == 0:
        rank_stds = np.zeros(n_features)

    # ------------------------------------------------------------------
    # Step 6: Pairwise correlation
    # ------------------------------------------------------------------
    corr_pairs = _find_correlated_pairs(X, feature_names, mean_shap)
    redundant_features: set[str] = {p["redundant_feature"] for p in corr_pairs}

    # ------------------------------------------------------------------
    # Step 7: Apply selection criteria
    # ------------------------------------------------------------------
    # Compute mean importance for instability ratio
    mean_imp_all = mean_shap if len(mean_shap) == n_features else np.zeros(n_features)

    keep: list[str] = []
    remove_candidates: list[dict[str, Any]] = []

    for i, feat in enumerate(feature_names):
        shap_val = float(mean_shap[i]) if i < len(mean_shap) else 0.0
        rank_std = float(rank_stds[i]) if i < len(rank_stds) else 0.0
        imp_val = float(mean_imp_all[i]) if i < len(mean_imp_all) else 0.0

        reasons: list[str] = []
        corr_with: list[str] = []

        # Criterion 1: low SHAP
        if shap_val < _SHAP_LOW_THRESHOLD:
            reasons.append(f"mean_|SHAP|={shap_val:.6f} < {_SHAP_LOW_THRESHOLD}")

        # Criterion 2: rank instability (using actual fold rank stds)
        # We compute ratio relative to mean rank across all features
        mean_rank = float(n_features / 2)  # expected value of a uniform rank
        if mean_rank > 0 and rank_std > _INSTABILITY_RATIO * mean_rank:
            reasons.append(
                f"rank_std={rank_std:.2f} > {_INSTABILITY_RATIO} × mean_rank={mean_rank:.1f}"
            )

        # Criterion 3: redundant (highly correlated with better feature)
        if feat in redundant_features:
            corr_with = [
                p["feature_a"] if p["feature_b"] == feat else p["feature_b"]
                for p in corr_pairs
                if p["redundant_feature"] == feat
            ]
            reasons.append(
                f"corr > {_CORR_HIGH_THRESHOLD} with {corr_with}"
            )

        if reasons:
            remove_candidates.append(
                {
                    "feature": feat,
                    "mean_shap": round(shap_val, 6),
                    "rank_std": round(rank_std, 4),
                    "correlated_with": corr_with,
                    "reasons": reasons,
                    "recommendation": "REMOVE_CANDIDATE",
                }
            )
        else:
            keep.append(feat)

    # ------------------------------------------------------------------
    # Console table output
    # ------------------------------------------------------------------
    _print_selection_table(feature_names, mean_shap, rank_stds, corr_pairs, keep)

    # ------------------------------------------------------------------
    # Write output JSON
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "model_version": model_version,
        "generated_at": date.today().isoformat(),
        "n_features_total": n_features,
        "n_keep": len(keep),
        "n_remove_candidates": len(remove_candidates),
        "keep": keep,
        "remove_candidates": remove_candidates,
        "correlation_pairs": corr_pairs,
        "summary": {
            "shap_threshold": _SHAP_LOW_THRESHOLD,
            "instability_ratio": _INSTABILITY_RATIO,
            "correlation_threshold": _CORR_HIGH_THRESHOLD,
            "n_models_averaged": len(shap_arrays),
            "n_samples_used_for_shap": int(X_shap.shape[0]),
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"feature_selection_{model_version}_{date.today().isoformat()}.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info(
        "feature_selection_output_written",
        path=str(out_file),
        n_keep=len(keep),
        n_remove=len(remove_candidates),
    )
    return output


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------


def _load_feature_matrix(
    data_dir: str,
    feature_names: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Load historical data and run feature engineering to build X.

    Attempts to load all Parquet files from ``data_dir``, compute features via
    ``TechnicalFeatures``, and return the feature matrix aligned to
    ``feature_names``.

    Args:
        data_dir: Directory containing Parquet files.
        feature_names: Expected feature column names.

    Returns:
        Tuple ``(X, y, actual_feature_names)`` where X may be None on failure.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        log.error("data_dir_not_found", path=str(data_path))
        return None, None, []

    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        log.error("no_parquet_files", data_dir=str(data_path))
        return None, None, []

    frames: list[Any] = []
    labels: list[Any] = []

    try:
        import polars as pl  # noqa: PLC0415
        from algo_trading.features.technical import TechnicalFeatures  # noqa: PLC0415
    except ImportError as exc:
        log.error("import_failed", error=str(exc))
        return None, None, []

    for pq_file in parquet_files:
        try:
            df = pl.read_parquet(pq_file)
            enriched = TechnicalFeatures().compute(df)
            feat_cols = [c for c in enriched.columns if c.startswith("feat_")]
            if not feat_cols:
                continue
            X_part = enriched[feat_cols].to_numpy()
            frames.append((feat_cols, X_part))
            if "label" in enriched.columns:
                labels.append(enriched["label"].to_numpy())
        except Exception as exc:
            log.warning("parquet_load_failed", file=str(pq_file), error=str(exc))

    if not frames:
        return None, None, []

    # Use the feature set from the first successful file
    ref_cols, X = frames[0]
    y = labels[0] if labels else None

    if len(frames) > 1:
        # Concatenate frames that share the same columns
        compatible = [
            arr for cols, arr in frames if cols == ref_cols
        ]
        if len(compatible) > 1:
            X = np.vstack(compatible)
            if labels:
                y = np.concatenate(labels[: len(compatible)])

    return X, y, ref_cols


# ---------------------------------------------------------------------------
# Console table printer
# ---------------------------------------------------------------------------


def _print_selection_table(
    feature_names: list[str],
    mean_shap: np.ndarray,
    rank_stds: np.ndarray,
    corr_pairs: list[dict[str, Any]],
    keep: list[str],
) -> None:
    """Print a formatted console table summarising feature selection.

    Args:
        feature_names: All feature names.
        mean_shap: Mean absolute SHAP per feature.
        rank_stds: Rank standard deviation per feature.
        corr_pairs: List of correlated pair dicts.
        keep: Features marked as keep.
    """
    redundant = {p["redundant_feature"] for p in corr_pairs}
    corr_lookup: dict[str, list[str]] = {}
    for p in corr_pairs:
        rf = p["redundant_feature"]
        other = p["feature_a"] if p["feature_b"] == rf else p["feature_b"]
        corr_lookup.setdefault(rf, []).append(other)

    header = f"{'Feature':<35} {'MeanSHAP':>10} {'RankStd':>9} {'CorrelatedWith':<25} {'Recommendation'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, feat in enumerate(feature_names):
        shap_v = float(mean_shap[i]) if i < len(mean_shap) else 0.0
        rstd = float(rank_stds[i]) if i < len(rank_stds) else 0.0
        corr_with = ", ".join(corr_lookup.get(feat, []))[:24]
        rec = "KEEP" if feat in keep else "REMOVE_CANDIDATE"
        print(f"{feat:<35} {shap_v:>10.6f} {rstd:>9.3f} {corr_with:<25} {rec}")

    print("-" * len(header))
    print(f"Total: {len(feature_names)}  |  Keep: {len(keep)}  |  Remove candidates: {len(feature_names) - len(keep)}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHAP-guided feature selection for NSE AlgoTrader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-version",
        default="v1.0.0",
        help="Model version string to load from registry (default: v1.0.0)",
    )
    parser.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help=f"Directory with historical Parquet files (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--registry-dir",
        default=_DEFAULT_REGISTRY_DIR,
        help=f"Model registry directory (default: {_DEFAULT_REGISTRY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for output JSON (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use saved SHAP values / native importances instead of recomputing",
    )
    parser.add_argument(
        "--n-shap-samples",
        type=int,
        default=2000,
        help="Max rows to pass to SHAP explainer (default: 2000)",
    )
    return parser


def main() -> None:
    """CLI entry point for select_features."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = run_feature_selection(
            model_version=args.model_version,
            data_dir=args.data_dir,
            registry_dir=args.registry_dir,
            output_dir=args.output_dir,
            quick=args.quick,
            n_shap_samples=args.n_shap_samples,
        )
        print(
            f"\nFeature selection complete: {result['n_keep']} kept, "
            f"{result['n_remove_candidates']} removal candidates."
        )
        sys.exit(0)
    except FileNotFoundError as exc:
        log.error("select_features_failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        log.error("select_features_unexpected_error", error=str(exc))
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
