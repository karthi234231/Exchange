"""MLflow experiment tracking and DVC model/data versioning.

Every training run must be logged via ExperimentTracker.  DVCManager handles
artifact versioning so that any model version can be restored for audit or
regulatory review.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import structlog

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")

_MLFLOW_FALLBACK_DIR = Path("state")


def _mlflow_with_retry(
    operation: Callable[[], _T],
    fallback_payload: dict[str, Any] | None = None,
    operation_name: str = "mlflow_write",
) -> _T | None:
    """Call an MLflow write operation with 3 attempts and exponential backoff.

    Retry schedule: attempt 1 immediately, wait 1 s before attempt 2,
    wait 2 s before attempt 3, wait 4 s (then give up).

    On persistent failure after 3 attempts, if ``fallback_payload`` is provided
    the payload is written to ``state/mlflow_fallback_{timestamp}.json``.

    Args:
        operation: Zero-argument callable that performs the MLflow write.
        fallback_payload: Optional dict to persist as JSON on permanent failure.
        operation_name: Label used in log messages.

    Returns:
        Return value of ``operation()`` on success, or ``None`` on failure.
    """
    delays = [0, 1, 2, 4]  # sleep before attempt index 1, 2, 3 (index 0 = no sleep)
    last_exc: Exception | None = None
    for attempt, delay in enumerate(delays[:3], start=1):
        if delay:
            time.sleep(delay)
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "mlflow_write.retry",
                operation=operation_name,
                attempt=attempt,
                error=str(exc),
            )

    # All 3 attempts failed
    logger.error(
        "mlflow_write.permanent_failure",
        operation=operation_name,
        error=str(last_exc),
    )
    if fallback_payload is not None:
        _MLFLOW_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fallback_path = _MLFLOW_FALLBACK_DIR / f"mlflow_fallback_{ts}.json"
        fallback_path.write_text(
            json.dumps({"operation": operation_name, "payload": fallback_payload}, indent=2),
            encoding="utf-8",
        )
        logger.warning(
            "mlflow_write.fallback_written",
            path=str(fallback_path),
        )
    return None


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Wraps MLflow so that every training run is tracked automatically.

    All runs are nested under a single experiment named ``experiment_name``.
    Parameters, metrics, model artifacts, and feature schemas are recorded
    for every run.

    Args:
        tracking_uri: MLflow tracking server URI or local directory path.
        experiment_name: Name of the MLflow experiment bucket.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "nse_algotrader",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._log = structlog.get_logger(self.__class__.__name__)

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(experiment_name)
            self._log.info(
                "experiment_tracker.created",
                experiment_name=experiment_name,
                experiment_id=self._experiment_id,
            )
        else:
            self._experiment_id = experiment.experiment_id
            self._log.info(
                "experiment_tracker.loaded",
                experiment_name=experiment_name,
                experiment_id=self._experiment_id,
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager that wraps an MLflow run.

        Usage::

            with tracker.start_run("retrain_v2.3.1") as run:
                tracker.log_params(params)
                ...

        Args:
            run_name: Human-readable label for this run (e.g. "retrain_v2.3.1").
            tags: Optional key-value tags attached to the run in MLflow.

        Yields:
            The active ``mlflow.ActiveRun`` context object.
        """
        resolved_tags = tags or {}
        resolved_tags.setdefault("project", "nse_algotrader")
        self._log.info("start_run", run_name=run_name, tags=resolved_tags)
        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=resolved_tags,
        ) as active_run:
            yield active_run
        self._log.info("end_run", run_name=run_name, run_id=active_run.info.run_id)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat hyperparameter dictionary to the active MLflow run.

        Args:
            params: Flat dict of parameter names → values. Nested dicts are
                serialised to JSON strings automatically.
        """
        flat: dict[str, Any] = {}
        for k, v in params.items():
            flat[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        _mlflow_with_retry(
            lambda: mlflow.log_params(flat),
            fallback_payload={"params": flat},
            operation_name="log_params",
        )
        self._log.debug("log_params", keys=list(flat.keys()))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log scalar metrics to the active MLflow run.

        Typical metrics: ``sharpe``, ``profit_factor``, ``win_rate``,
        ``max_drawdown``, ``n_trades`` per fold.

        Args:
            metrics: Dict of metric name → scalar value.
            step: Optional integer step (e.g. fold index) for time-series plots
                in the MLflow UI.
        """
        _mlflow_with_retry(
            lambda: mlflow.log_metrics(metrics, step=step),
            fallback_payload={"metrics": metrics, "step": step},
            operation_name="log_metrics",
        )
        self._log.debug("log_metrics", keys=list(metrics.keys()), step=step)

    def log_model(
        self,
        model: Any,
        model_name: str,
        feature_names: list[str],
    ) -> None:
        """Log a trained model artifact with its feature schema and package versions.

        Supports XGBoost, LightGBM, CatBoost, and generic sklearn-compatible
        models.  The feature schema and installed package versions are saved
        as JSON artifacts alongside the model.

        Args:
            model: Trained model object.
            model_name: Artifact path inside the MLflow run (e.g. "xgb_model").
            feature_names: Ordered list of feature column names used during
                training (required for reproducibility and SEBI audits).
        """
        import importlib
        import sys

        model_type = type(model).__module__

        if "xgboost" in model_type:
            _mlflow_with_retry(
                lambda: mlflow.xgboost.log_model(model, model_name),
                fallback_payload={"model_name": model_name, "model_type": "xgboost"},
                operation_name="log_model",
            )
        elif "lightgbm" in model_type:
            _mlflow_with_retry(
                lambda: mlflow.lightgbm.log_model(model, model_name),
                fallback_payload={"model_name": model_name, "model_type": "lightgbm"},
                operation_name="log_model",
            )
        elif "catboost" in model_type:
            _mlflow_with_retry(
                lambda: mlflow.catboost.log_model(model, model_name),
                fallback_payload={"model_name": model_name, "model_type": "catboost"},
                operation_name="log_model",
            )
        else:
            _mlflow_with_retry(
                lambda: mlflow.sklearn.log_model(model, model_name),
                fallback_payload={"model_name": model_name, "model_type": "sklearn"},
                operation_name="log_model",
            )

        # Feature schema
        schema_path = Path(tempfile.gettempdir()) / f"{model_name}_feature_schema.json"
        schema_path.write_text(
            json.dumps({"feature_names": feature_names, "n_features": len(feature_names)}, indent=2)
        )
        _mlflow_with_retry(
            lambda: mlflow.log_artifact(str(schema_path), artifact_path="schemas"),
            fallback_payload={"artifact": "feature_schema", "model_name": model_name},
            operation_name="log_artifact_schema",
        )
        schema_path.unlink(missing_ok=True)

        # Package versions
        packages_of_interest = ["xgboost", "lightgbm", "catboost", "numpy", "pandas", "optuna", "mlflow"]
        versions: dict[str, str] = {}
        for pkg in packages_of_interest:
            try:
                mod = importlib.import_module(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[pkg] = "not_installed"
        versions["python"] = sys.version

        versions_path = Path(tempfile.gettempdir()) / f"{model_name}_pkg_versions.json"
        versions_path.write_text(json.dumps(versions, indent=2))
        _mlflow_with_retry(
            lambda: mlflow.log_artifact(str(versions_path), artifact_path="metadata"),
            fallback_payload={"artifact": "pkg_versions", "model_name": model_name},
            operation_name="log_artifact_versions",
        )
        versions_path.unlink(missing_ok=True)

        self._log.info("log_model", model_name=model_name, n_features=len(feature_names))

    def log_feature_importance(self, importance_dict: dict[str, float]) -> None:
        """Log feature importance scores as a JSON artifact.

        Args:
            importance_dict: Mapping of feature name → importance score.
                Typically the output of ``EnsembleExplainer.compute_average_importance``.
        """
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        tmp_path = Path(tempfile.gettempdir()) / "feature_importance.json"
        tmp_path.write_text(json.dumps(sorted_importance, indent=2))
        _mlflow_with_retry(
            lambda: mlflow.log_artifact(str(tmp_path), artifact_path="explainability"),
            fallback_payload={"artifact": "feature_importance"},
            operation_name="log_artifact_feature_importance",
        )
        tmp_path.unlink(missing_ok=True)
        self._log.debug("log_feature_importance", n_features=len(importance_dict))

    def log_walk_forward_results(self, fold_metrics: list[dict[str, Any]]) -> None:
        """Log per-fold metrics and aggregate statistics.

        Each entry in ``fold_metrics`` is logged as step-indexed metrics in
        MLflow.  Aggregate mean/std/min/max are also logged as summary metrics.

        Args:
            fold_metrics: List of metric dicts, one per walk-forward fold.
                Each dict should contain keys like ``sharpe``, ``profit_factor``,
                ``win_rate``, ``max_drawdown``, ``n_trades``.
        """
        import statistics

        if not fold_metrics:
            self._log.warning("log_walk_forward_results: empty fold_metrics list")
            return

        # Log per-fold metrics indexed by fold number
        for fold_idx, fold_dict in enumerate(fold_metrics):
            self.log_metrics(fold_dict, step=fold_idx)

        # Aggregate stats across folds
        all_keys = {k for d in fold_metrics for k in d}
        agg_metrics: dict[str, float] = {}
        for key in all_keys:
            values = [d[key] for d in fold_metrics if key in d]
            if not values:
                continue
            agg_metrics[f"{key}_mean"] = statistics.mean(values)
            agg_metrics[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            agg_metrics[f"{key}_min"] = min(values)
            agg_metrics[f"{key}_max"] = max(values)

        _mlflow_with_retry(
            lambda: mlflow.log_metrics(agg_metrics),
            fallback_payload={"metrics": agg_metrics},
            operation_name="log_metrics_walk_forward_agg",
        )

        # Save full fold results as JSON artifact
        tmp_path = Path(tempfile.gettempdir()) / "walk_forward_results.json"
        tmp_path.write_text(
            json.dumps({"fold_metrics": fold_metrics, "aggregates": agg_metrics}, indent=2)
        )
        _mlflow_with_retry(
            lambda: mlflow.log_artifact(str(tmp_path), artifact_path="walk_forward"),
            fallback_payload={"artifact": "walk_forward_results"},
            operation_name="log_artifact_walk_forward",
        )
        tmp_path.unlink(missing_ok=True)
        self._log.info("log_walk_forward_results", n_folds=len(fold_metrics), agg_keys=list(agg_metrics.keys()))

    # ------------------------------------------------------------------
    # Run retrieval
    # ------------------------------------------------------------------

    def get_best_run(self, metric: str = "sharpe") -> dict[str, Any]:
        """Fetch metadata for the run with the highest value of ``metric``.

        Args:
            metric: MLflow metric name to rank runs by (default: ``"sharpe"``).

        Returns:
            Dict with keys: ``run_id``, ``run_name``, ``metrics``, ``params``,
            ``artifact_uri``.

        Raises:
            ValueError: If the experiment has no completed runs.
        """
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        runs = client.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"No completed runs found in experiment '{self.experiment_name}'."
            )
        best = runs[0]
        result = {
            "run_id": best.info.run_id,
            "run_name": best.info.run_name,
            "metrics": dict(best.data.metrics),
            "params": dict(best.data.params),
            "artifact_uri": best.info.artifact_uri,
        }
        self._log.info("get_best_run", metric=metric, run_id=result["run_id"])
        return result

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_production_model(self, run_id: str, model_name: str) -> None:
        """Promote a logged model to the 'Production' stage in the MLflow Model Registry.

        Args:
            run_id: MLflow run ID that contains the logged model artifact.
            model_name: Registered model name (created if it does not exist).
        """
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

        # Retrieve the artifact URI for the model logged under this run
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        model_uri = f"{artifact_uri}/{model_name}"
        self._log.info(
            "register_production_model.registering",
            run_id=run_id,
            model_name=model_name,
            model_uri=model_uri,
        )

        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        self._log.info(
            "register_production_model.done",
            model_name=model_name,
            version=mv.version,
            stage="Production",
        )


# ---------------------------------------------------------------------------
# DVC Manager
# ---------------------------------------------------------------------------


class DVCManager:
    """DVC data and model versioning.

    Thin wrapper around DVC CLI commands.  Assumes ``dvc`` is installed and
    the repository has already been initialised with ``dvc init``.

    Args:
        repo_root: Path to the root of the git/DVC repository.
    """

    def __init__(self, repo_root: Path = Path(".")) -> None:
        self.repo_root = repo_root.resolve()
        self._log = structlog.get_logger(self.__class__.__name__)

    def _run(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a shell command in the repo root.

        Args:
            cmd: Command tokens list (no shell interpolation).
            check: If True, raise on non-zero exit code.

        Returns:
            CompletedProcess result object.

        Raises:
            subprocess.CalledProcessError: If check=True and the command fails.
        """
        self._log.debug("dvc_run", cmd=" ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=check,
        )
        if result.returncode != 0:
            self._log.error(
                "dvc_run.failed",
                cmd=" ".join(cmd),
                stderr=result.stderr.strip(),
            )
        return result

    def add_data(self, data_path: Path) -> None:
        """Track a data file or directory with DVC.

        Equivalent to ``dvc add <data_path>``.  The resulting ``.dvc`` file
        should be committed to git by the caller.

        Args:
            data_path: Absolute or relative path to the data file/directory.
        """
        self._log.info("add_data", path=str(data_path))
        self._run(["dvc", "add", str(data_path)])
        self._log.info("add_data.done", path=str(data_path))

    def push_model(self, model_path: Path) -> None:
        """Add a model artifact to DVC and commit the .dvc pointer to git.

        Steps performed:
            1. ``dvc add <model_path>``
            2. ``git add <model_path>.dvc .gitignore``
            3. ``git commit -m "chore: update model artifact <model_path.name>"``

        Args:
            model_path: Path to the model file (e.g. ``.pkl`` or ``.cbm``).
        """
        self._log.info("push_model.start", path=str(model_path))
        self._run(["dvc", "add", str(model_path)])

        dvc_file = str(model_path) + ".dvc"
        self._run(
            [
                "git",
                "add",
                dvc_file,
                ".gitignore",
            ]
        )
        self._run(
            [
                "git",
                "commit",
                "-m",
                f"chore: update model artifact {model_path.name}",
            ]
        )
        self._run(["dvc", "push"])
        self._log.info("push_model.done", path=str(model_path))

    def pull_model(self, version_tag: str) -> None:
        """Restore a specific model version by checking out its git tag and pulling from DVC.

        Steps performed:
            1. ``git checkout <version_tag>``
            2. ``dvc pull``

        Args:
            version_tag: Git tag or commit SHA identifying the desired version
                (e.g. ``"v2.3.1"`` or a full commit hash).
        """
        self._log.info("pull_model.start", version_tag=version_tag)
        self._run(["git", "checkout", version_tag])
        self._run(["dvc", "pull"])
        self._log.info("pull_model.done", version_tag=version_tag)
