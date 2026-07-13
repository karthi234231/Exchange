"""SHAP-based explainability for the XGBoost + LightGBM + CatBoost ensemble.

SEBI compliance requirement: SHAP values must be computed and saved after every
training run.  The ``generate_sebi_report`` method produces the mandatory JSON
artefact for regulatory submission.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import shap
import structlog

matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt  # noqa: E402  (import after backend set)

logger = structlog.get_logger(__name__)

# Suppress noisy SHAP warnings about slow exact TreeExplainer paths
logging.getLogger("shap").setLevel(logging.ERROR)


class EnsembleExplainer:
    """SHAP explainability for the XGBoost + LightGBM + CatBoost ensemble.

    SHAP values are computed independently for each model and then averaged to
    produce ensemble-level feature importance scores.  All per-model explainers
    use ``shap.TreeExplainer`` for speed and accuracy.

    Args:
        xgb_model: Trained XGBoost classifier (``XGBClassifier``).
        lgb_model: Trained LightGBM classifier (``LGBMClassifier``).
        cat_model: Trained CatBoost classifier (``CatBoostClassifier``).
        feature_names: Ordered list of feature column names matching the
            training matrix column order.
    """

    #: Default class index → human-readable label mapping used in SEBI reports.
    #: Matches ensemble._CLASS_TO_SIGNAL: 0=bear, 1=sideways, 2=bull. [P0-FIX-11]
    _DEFAULT_CLASS_LABELS: dict[int, str] = {0: "bear", 1: "sideways", 2: "bull"}

    def __init__(
        self,
        xgb_model: Any,
        lgb_model: Any,
        cat_model: Any,
        feature_names: list[str],
        class_labels: dict[int, str] | None = None,
    ) -> None:
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.feature_names = feature_names
        self.CLASS_LABELS: dict[int, str] = (
            class_labels if class_labels is not None else self._DEFAULT_CLASS_LABELS
        )
        self._log = structlog.get_logger(self.__class__.__name__)

        # Build TreeExplainer instances once at construction time
        self._log.info("explainer.init", n_features=len(feature_names))
        self._xgb_explainer = shap.TreeExplainer(xgb_model)
        self._lgb_explainer = shap.TreeExplainer(lgb_model)
        self._cat_explainer = shap.TreeExplainer(cat_model)

    # ------------------------------------------------------------------
    # Core SHAP computation
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Compute SHAP values for all three models.

        Uses ``shap.TreeExplainer`` which is CPU-only and exact for tree models.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Dict with keys ``"xgb"``, ``"lgb"``, ``"cat"``.  Each value is a
            numpy array of shape ``(n_samples, n_features, n_classes)`` where
            n_classes is the number of output classes.
        """
        self._log.info("compute_shap_values.start", n_samples=X.shape[0])

        # Use a background sample to avoid OOM on large datasets.
        # Never pass the full training matrix to the explainers.
        background = shap.sample(X, min(500, len(X)))

        def _to_3d(raw: Any) -> np.ndarray:
            """Normalise SHAP output to (n_samples, n_features, n_classes)."""
            if isinstance(raw, list):
                # shap returns list[array(n_samples, n_features)] per class
                return np.stack(raw, axis=2)
            arr = np.array(raw)
            if arr.ndim == 2:
                # Binary or single-output: expand to (n_samples, n_features, 1)
                return arr[:, :, np.newaxis]
            return arr  # already (n_samples, n_features, n_classes)

        xgb_shap = _to_3d(self._xgb_explainer.shap_values(background))
        lgb_shap = _to_3d(self._lgb_explainer.shap_values(background))
        cat_shap = _to_3d(self._cat_explainer.shap_values(background))

        result = {"xgb": xgb_shap, "lgb": lgb_shap, "cat": cat_shap}
        self._log.info(
            "compute_shap_values.done",
            shapes={k: v.shape for k, v in result.items()},
        )
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def compute_average_importance(self, X: np.ndarray) -> dict[str, float]:
        """Compute ensemble-averaged mean-absolute SHAP importance.

        Averages the absolute SHAP values across all three models and all
        classes, then returns a feature-name-sorted dict.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Dict mapping feature name → importance score, sorted descending by
            importance.
        """
        shap_dict = self.compute_shap_values(X)

        # Mean absolute importance per model: sum over classes, mean over samples
        importances: list[np.ndarray] = []
        for model_shap in shap_dict.values():
            # model_shap: (n_samples, n_features, n_classes)
            per_sample_abs = np.abs(model_shap).sum(axis=2)  # (n_samples, n_features)
            importances.append(per_sample_abs.mean(axis=0))  # (n_features,)

        avg_importance = np.mean(np.stack(importances, axis=0), axis=0)  # (n_features,)

        importance_dict: dict[str, float] = {
            name: float(score)
            for name, score in zip(self.feature_names, avg_importance)
        }
        # Sort descending
        sorted_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        self._log.debug(
            "compute_average_importance.done",
            top_5=list(sorted_dict.items())[:5],
        )
        return sorted_dict

    # ------------------------------------------------------------------
    # SEBI compliance report
    # ------------------------------------------------------------------

    def generate_sebi_report(
        self,
        X: np.ndarray,
        output_path: Path,
        model_version: str,
        symbol: str,
    ) -> Path:
        """Generate a SEBI-compliant model explainability report as JSON.

        The report satisfies SEBI's algorithmic trading explainability
        requirement and must be produced and saved after every training run.

        Report structure::

            {
                "model_version": "v2.3.1",
                "symbol": "NIFTY50",
                "generated_at": "2026-06-06T12:00:00Z",
                "top_10_features": [{"feature": ..., "importance": ...}, ...],
                "per_class_importance": {
                    "bull":  {"feature": importance, ...},
                    "bear":  {"feature": importance, ...},
                    "sideways": {"feature": importance, ...}
                },
                "sample_explanations": [
                    {"sample_index": ..., "feature_contributions": {...}},
                    ...
                ]
            }

        Args:
            X: Feature matrix used to compute SHAP values.
            output_path: Directory where the JSON report will be saved.  The
                file is named ``sebi_report_{symbol}_{model_version}.json``.
            model_version: Version string (e.g. ``"v2.3.1"``).
            symbol: NSE trading symbol (e.g. ``"NIFTY50"``).

        Returns:
            Absolute path to the saved JSON report file.
        """
        self._log.info(
            "generate_sebi_report.start",
            symbol=symbol,
            model_version=model_version,
            n_samples=X.shape[0],
        )

        shap_dict = self.compute_shap_values(X)
        avg_importance = self.compute_average_importance(X)

        # Top-10 features
        top_10 = [
            {"feature": name, "importance": round(score, 6)}
            for name, score in list(avg_importance.items())[:10]
        ]

        # Per-class importance: average absolute SHAP per class across all models
        n_classes = next(iter(shap_dict.values())).shape[2]
        per_class: dict[str, dict[str, float]] = {}
        for cls_idx in range(n_classes):
            label = self.CLASS_LABELS.get(cls_idx, str(cls_idx))
            class_importances: list[np.ndarray] = []
            for model_shap in shap_dict.values():
                cls_shap = model_shap[:, :, cls_idx]  # (n_samples, n_features)
                class_importances.append(np.abs(cls_shap).mean(axis=0))
            avg_cls = np.mean(np.stack(class_importances, axis=0), axis=0)
            per_class[label] = dict(
                sorted(
                    {name: round(float(v), 6) for name, v in zip(self.feature_names, avg_cls)}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        # Sample explanations: 3 evenly-spaced indices (deterministic — no random)
        n = X.shape[0]
        sample_indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        sample_explanations = []
        for idx in sample_indices:
            contributions: dict[str, float] = {}
            for model_shap in shap_dict.values():
                for feat_idx, feat_name in enumerate(self.feature_names):
                    # Average contribution across classes for this sample
                    contrib = float(np.mean(model_shap[idx, feat_idx, :]))
                    contributions[feat_name] = contributions.get(feat_name, 0.0) + contrib
            # Average across the 3 models
            n_models = len(shap_dict)
            contributions = {k: round(v / n_models, 6) for k, v in contributions.items()}
            contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            sample_explanations.append(
                {"sample_index": int(idx), "feature_contributions": contributions}
            )

        report: dict[str, Any] = {
            "model_version": model_version,
            "symbol": symbol,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "top_10_features": top_10,
            "per_class_importance": per_class,
            "sample_explanations": sample_explanations,
        }

        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / f"sebi_report_{symbol}_{model_version}.json"
        report_file.write_text(json.dumps(report, indent=2))

        self._log.info(
            "generate_sebi_report.done",
            path=str(report_file),
            top_feature=top_10[0]["feature"] if top_10 else None,
        )
        return report_file

    # ------------------------------------------------------------------
    # Summary plot
    # ------------------------------------------------------------------

    def plot_summary(self, X: np.ndarray, output_path: Path) -> None:
        """Save a SHAP summary bar plot (ensemble-averaged) as a PNG.

        Uses ``shap.summary_plot`` with ``show=False`` to suppress the
        interactive display, then saves via ``matplotlib.pyplot.savefig``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            output_path: Full file path for the output PNG
                (e.g. ``Path("reports/shap_summary.png")``).
        """
        self._log.info("plot_summary.start", output_path=str(output_path))

        shap_dict = self.compute_shap_values(X)

        # Average absolute SHAP across models and classes → (n_samples, n_features)
        model_avgs: list[np.ndarray] = []
        for model_shap in shap_dict.values():
            # Collapse classes by mean absolute → (n_samples, n_features)
            model_avgs.append(np.abs(model_shap).mean(axis=2))
        ensemble_shap = np.mean(np.stack(model_avgs, axis=0), axis=0)  # (n_samples, n_features)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            ensemble_shap,
            features=X,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=min(20, len(self.feature_names)),
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close("all")

        self._log.info("plot_summary.done", output_path=str(output_path))

    # ------------------------------------------------------------------
    # Single prediction explanation
    # ------------------------------------------------------------------

    def explain_single_prediction(self, x: np.ndarray) -> dict[str, float]:
        """Explain a single prediction for live trade monitoring.

        SHAP contributions are averaged across all three models and reported
        for the class with the highest predicted probability (the predicted
        class).

        Args:
            x: 1-D feature vector of shape ``(n_features,)`` or 2-D row array
                of shape ``(1, n_features)``.

        Returns:
            Dict mapping feature name → SHAP contribution for the predicted
            class, sorted by absolute contribution descending.  Positive values
            indicate the feature pushed the prediction toward the predicted
            class; negative values pushed it away.
        """
        x_2d = x.reshape(1, -1) if x.ndim == 1 else x

        # Determine predicted class from the ensemble mean probability
        xgb_proba = self.xgb_model.predict_proba(x_2d)
        lgb_proba = self.lgb_model.predict_proba(x_2d)
        cat_proba = self.cat_model.predict_proba(x_2d)
        mean_proba = (xgb_proba + lgb_proba + cat_proba) / 3.0
        predicted_class = int(np.argmax(mean_proba[0]))

        # Compute SHAP values and extract contributions for predicted class
        shap_dict = self.compute_shap_values(x_2d)  # each: (1, n_features, n_classes)

        contributions: dict[str, float] = {}
        for feat_idx, feat_name in enumerate(self.feature_names):
            contrib_sum = 0.0
            for model_shap in shap_dict.values():
                contrib_sum += float(model_shap[0, feat_idx, predicted_class])
            contributions[feat_name] = round(contrib_sum / len(shap_dict), 6)

        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        self._log.debug(
            "explain_single_prediction",
            predicted_class=self.CLASS_LABELS.get(predicted_class, str(predicted_class)),
            top_feature=next(iter(sorted_contributions)) if sorted_contributions else None,
        )
        return sorted_contributions
