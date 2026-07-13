"""Production model registry for NSE AlgoTrader.

Spec reference: SURV-5.
A deployed model is IMMUTABLE. Inference refuses to run on schema mismatches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class ModelMismatchError(Exception):
    """Raised when live feature schema does not match the trained schema."""


class RegistryError(Exception):
    """Raised for registry-level failures (duplicate version, no approved model, etc.)."""


class InsufficientDataError(Exception):
    """Raised when training data does not meet minimum bar or trade requirements."""


@dataclass(frozen=True)
class FrozenModelSpec:
    """Production model specification.

    Once deployed, no field changes are allowed until the Sunday retrain window.
    The feature_schema_hash guards against silent schema drift — if feature
    order or names change the hash changes and inference is blocked.

    Attributes:
        version: Semantic version string, e.g. ``"v2.3.1"``.
        feature_names: Ordered tuple of feature names; hash changes if order changes.
        feature_schema_hash: Full 64-character SHA-256 hex digest of ``str(feature_names)``.
        bar_type: One of ``"volume"``, ``"tick"``, or ``"time_15min"``.
        bar_spec: Bar-type parameters, e.g. ``{"volume_threshold": 50000}``.
        trained_on: Training data cutoff date.
        symbol_universe: Symbols the model was trained on.
        validation_metrics: Walk-forward validation metrics, e.g.
            ``{"sharpe": 1.2, "profit_factor": 1.4}``.
        treelite_xgb_path: Path to compiled XGBoost ``.so`` file.
        treelite_lgb_path: Path to compiled LightGBM ``.so`` file.
        treelite_cat_path: Path to compiled CatBoost ``.so`` file.
        meta_lr_path: Path to meta LogisticRegression ``.pkl`` file.
        package_versions: Library versions at training time, e.g.
            ``{"xgboost": "2.0.1"}``.
        approval_status: One of ``"approved"``, ``"pending"``, ``"rejected"``.
        holdout_sharpe: Sharpe ratio on the reserved holdout fold (fold[-1]).
            None if not computed (e.g. older spec files).
        holdout_accuracy: Accuracy on the reserved holdout fold (fold[-1]).
            None if not computed.
        unstable_features_count: Number of features flagged as unstable
            (std > 0.5 × mean importance across folds). None if not computed.
        min_train_bars_required: Minimum number of training bars required.
            None means no enforced minimum beyond the registry default.
        min_trades_per_fold_required: Minimum number of trades per fold.
            None means no enforced minimum beyond the registry default.
    """

    version: str
    feature_names: tuple[str, ...]
    feature_schema_hash: str
    bar_type: str
    bar_spec: dict[str, Any]
    trained_on: date
    symbol_universe: tuple[str, ...]
    validation_metrics: dict[str, float]
    treelite_xgb_path: str
    treelite_lgb_path: str
    treelite_cat_path: str
    meta_lr_path: str
    package_versions: dict[str, str]
    approval_status: str
    # Optional fields added in v2 — default None for backward compatibility
    holdout_sharpe: float | None = None
    holdout_accuracy: float | None = None
    unstable_features_count: int | None = None
    min_train_bars_required: int | None = None
    min_trades_per_fold_required: int | None = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_input(self, X_columns: list[str]) -> None:
        """Raise ModelMismatchError if the live feature schema has drifted.

        Args:
            X_columns: Feature column names from the live feature vector,
                in the order they appear.

        Raises:
            ModelMismatchError: If ``X_columns`` do not exactly match
                ``self.feature_names`` (order-sensitive).
        """
        live_tuple = tuple(X_columns)
        if live_tuple != self.feature_names:
            live_hash = self.compute_feature_hash(live_tuple)
            raise ModelMismatchError(
                f"Feature schema mismatch. "
                f"Expected hash={self.feature_schema_hash} "
                f"(trained on {len(self.feature_names)} features), "
                f"got hash={live_hash} "
                f"({len(X_columns)} features). "
                f"First differing index: "
                f"{next((i for i, (a, b) in enumerate(zip(self.feature_names, live_tuple)) if a != b), 'length differs')}."
            )

    def validate_bar_type(self, live_bar_type: str, live_bar_spec: dict[str, Any]) -> None:
        """Raise ModelMismatchError if the live bar type or spec differs from training.

        The train-bar-type must equal the live-bar-type (spec constraint
        TRAIN_BAR_TYPE == LIVE_BAR_TYPE).

        Args:
            live_bar_type: Bar type string used by the live feed.
            live_bar_spec: Bar spec dict used by the live feed.

        Raises:
            ModelMismatchError: If bar type or bar spec does not match the
                trained configuration.
        """
        if live_bar_type != self.bar_type:
            raise ModelMismatchError(
                f"Bar type mismatch: model trained on '{self.bar_type}', "
                f"live feed produces '{live_bar_type}'."
            )
        if live_bar_spec != self.bar_spec:
            raise ModelMismatchError(
                f"Bar spec mismatch: model bar_spec={self.bar_spec!r}, "
                f"live bar_spec={live_bar_spec!r}."
            )

    # ------------------------------------------------------------------
    # Hash utility
    # ------------------------------------------------------------------

    @classmethod
    def compute_feature_hash(cls, feature_names: tuple[str, ...]) -> str:
        """Compute the full 64-character SHA-256 hex digest over the feature name tuple.

        Args:
            feature_names: Ordered tuple of feature names.

        Returns:
            Full 64-character SHA-256 hex digest of ``str(feature_names)``.
        """
        raw = str(feature_names).encode("utf-8")
        return sha256(raw).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the spec to a JSON file.

        Includes all optional v2 fields so that ``load()`` can round-trip them
        correctly.  Fields with None values are written explicitly so that
        older readers can distinguish "not computed" from missing.

        Also persists the feature schema artifact alongside the spec JSON so
        that future loads can detect schema drift via
        :func:`~algo_trading.features.engine.load_and_validate_feature_schema`
        (SURV-5).

        Args:
            path: Destination file path. Parent directories must exist.
        """
        data: dict[str, Any] = {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "feature_schema_hash": self.feature_schema_hash,
            "bar_type": self.bar_type,
            "bar_spec": self.bar_spec,
            "trained_on": self.trained_on.isoformat(),
            "symbol_universe": list(self.symbol_universe),
            "validation_metrics": self.validation_metrics,
            "treelite_xgb_path": self.treelite_xgb_path,
            "treelite_lgb_path": self.treelite_lgb_path,
            "treelite_cat_path": self.treelite_cat_path,
            "meta_lr_path": self.meta_lr_path,
            "package_versions": self.package_versions,
            "approval_status": self.approval_status,
            # v2 optional fields
            "holdout_sharpe": self.holdout_sharpe,
            "holdout_accuracy": self.holdout_accuracy,
            "unstable_features_count": self.unstable_features_count,
            "min_train_bars_required": self.min_train_bars_required,
            "min_trades_per_fold_required": self.min_trades_per_fold_required,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("model_spec_saved", version=self.version, path=str(path))

        # Persist a feature schema artifact alongside the spec so that the
        # inference engine can detect drift without re-loading the full spec.
        schema_path = path.with_suffix(".feature_schema.json")
        try:
            from algo_trading.features.engine import write_feature_schema  # noqa: PLC0415
            write_feature_schema(schema_path)
            logger.info(
                "model_spec_feature_schema_written",
                version=self.version,
                schema_path=str(schema_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model_spec_feature_schema_write_failed",
                version=self.version,
                schema_path=str(schema_path),
                error=str(exc),
            )

    @classmethod
    def load(cls, path: Path) -> "FrozenModelSpec":
        """Deserialise a spec from a JSON file and validate the feature hash.

        Handles both v1 spec files (without optional fields) and v2 spec files
        (with holdout_sharpe, holdout_accuracy, etc.) via ``data.get()`` with
        None defaults.

        Args:
            path: Path to a JSON file previously written by :meth:`save`.

        Returns:
            A fully validated :class:`FrozenModelSpec` instance.

        Raises:
            ModelMismatchError: If the stored ``feature_schema_hash`` does not
                match the hash re-computed from the stored ``feature_names``.
            FileNotFoundError: If ``path`` does not exist.
        """
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        feature_names: tuple[str, ...] = tuple(data["feature_names"])
        expected_hash = cls.compute_feature_hash(feature_names)
        stored_hash: str = data["feature_schema_hash"]
        if stored_hash != expected_hash:
            raise ModelMismatchError(
                f"Spec file corrupted: stored hash '{stored_hash}' does not match "
                f"recomputed hash '{expected_hash}' for feature_names."
            )
        spec = cls(
            version=data["version"],
            feature_names=feature_names,
            feature_schema_hash=stored_hash,
            bar_type=data["bar_type"],
            bar_spec=data["bar_spec"],
            trained_on=date.fromisoformat(data["trained_on"]),
            symbol_universe=tuple(data["symbol_universe"]),
            validation_metrics=data["validation_metrics"],
            treelite_xgb_path=data["treelite_xgb_path"],
            treelite_lgb_path=data["treelite_lgb_path"],
            treelite_cat_path=data["treelite_cat_path"],
            meta_lr_path=data["meta_lr_path"],
            package_versions=data["package_versions"],
            approval_status=data["approval_status"],
            # v2 optional fields — use .get() with None defaults for backward compat
            holdout_sharpe=data.get("holdout_sharpe", None),
            holdout_accuracy=data.get("holdout_accuracy", None),
            unstable_features_count=data.get("unstable_features_count", None),
            min_train_bars_required=data.get("min_train_bars_required", None),
            min_trades_per_fold_required=data.get("min_trades_per_fold_required", None),
        )
        logger.info("model_spec_loaded", version=spec.version, path=str(path))

        # Validate the feature schema artifact alongside the spec if it exists.
        # This catches silent drift between the spec's recorded feature names and
        # the feature engine currently running in production (SURV-5).
        schema_path = path.with_suffix(".feature_schema.json")
        if schema_path.exists():
            try:
                from algo_trading.features.engine import (  # noqa: PLC0415
                    FeatureSchemaMismatchError,
                    load_and_validate_feature_schema,
                )
                load_and_validate_feature_schema(schema_path)
                logger.info(
                    "model_spec_feature_schema_validated",
                    version=spec.version,
                    schema_path=str(schema_path),
                )
            except FeatureSchemaMismatchError:
                # Re-raise without wrapping — caller must handle drift explicitly.
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "model_spec_feature_schema_validation_failed",
                    version=spec.version,
                    schema_path=str(schema_path),
                    error=str(exc),
                )
        else:
            logger.debug(
                "model_spec_feature_schema_not_found",
                version=spec.version,
                schema_path=str(schema_path),
                note="Schema artifact absent — skipping drift check (SURV-5)",
            )

        return spec


class ModelRegistry:
    """Manages versioned model specs and enforces the retrain schedule.

    The registry stores one JSON file per spec under ``registry_dir``.
    Only one model may hold ``approval_status == "approved"`` at any time;
    registering a new approved model automatically demotes the previous one
    to ``"pending"`` on disk.

    Attributes:
        RETRAIN_DAY: Weekday index on which retraining is permitted (6 = Sunday).
        RETRAIN_HOUR: Earliest IST hour at which retraining is permitted (20 = 8 PM).
        DEFAULT_MIN_BARS: Default minimum training bars required.
        DEFAULT_MIN_TRADES: Default minimum trades per fold required.
    """

    RETRAIN_DAY: int = 6   # Sunday
    RETRAIN_HOUR: int = 20  # 20:00 IST

    # NSE market session in IST (inclusive)
    _MARKET_OPEN_HOUR: int = 9
    _MARKET_OPEN_MINUTE: int = 15
    _MARKET_CLOSE_HOUR: int = 15
    _MARKET_CLOSE_MINUTE: int = 30

    DEFAULT_MIN_BARS: int = 5000
    DEFAULT_MIN_TRADES: int = 200

    def __init__(self, registry_dir: Path = Path("models/registry")) -> None:
        """Initialise the registry.

        Args:
            registry_dir: Directory in which spec JSON files are stored.
                Created if it does not exist.
        """
        self._dir = registry_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("model_registry_initialised", registry_dir=str(self._dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Required artifact file extensions that must exist on disk before registration.
    # .so files are platform-specific (Treelite compiled libraries).
    _REQUIRED_ARTIFACT_PATHS = ("treelite_xgb_path", "treelite_lgb_path",
                                 "treelite_cat_path", "meta_lr_path")
    # Minimum required validation metrics (must be present and > 0).
    _REQUIRED_METRICS = ("sharpe", "profit_factor", "win_rate")
    # Edge gate thresholds — must match backtest runner constants.
    _MIN_SHARPE = 1.0
    _MIN_PROFIT_FACTOR = 1.2
    _MIN_WIN_RATE = 0.40

    def validate_artifacts(self, spec: FrozenModelSpec) -> None:
        """Validate that all required artifact files exist and metrics pass edge gate.

        Called by :meth:`register` for approved specs before committing to disk.
        Prevents stub/incomplete models from being registered as production-ready.

        Args:
            spec: The :class:`FrozenModelSpec` to validate.

        Raises:
            RegistryError: If any artifact file is missing or metrics fail gate. [P0-FIX-10]
        """
        errors: list[str] = []

        # 1. Check every required artifact file exists on disk.
        for attr in self._REQUIRED_ARTIFACT_PATHS:
            path_str: str = getattr(spec, attr, "")
            if not path_str:
                errors.append(f"Missing artifact field: {attr}")
                continue
            artifact_path = Path(path_str)
            if not artifact_path.exists():
                errors.append(
                    f"Artifact file missing on disk: {attr}={path_str!r}"
                )
            elif artifact_path.stat().st_size == 0:
                errors.append(
                    f"Artifact file is zero bytes (stub?): {attr}={path_str!r}"
                )

        # 2. Check required validation metrics are present and pass edge gate.
        metrics = spec.validation_metrics
        for key in self._REQUIRED_METRICS:
            if key not in metrics:
                errors.append(f"Missing validation metric: {key!r}")
            elif metrics[key] <= 0:
                errors.append(
                    f"Validation metric {key!r} = {metrics[key]:.4f} is <= 0 (stub?)"
                )

        # 3. Enforce edge gate thresholds on net-of-cost metrics.
        sharpe = metrics.get("sharpe", 0.0)
        pf = metrics.get("profit_factor", 0.0)
        wr = metrics.get("win_rate", 0.0)
        if sharpe < self._MIN_SHARPE:
            errors.append(
                f"Sharpe {sharpe:.2f} < {self._MIN_SHARPE} — edge gate not passed"
            )
        if pf < self._MIN_PROFIT_FACTOR:
            errors.append(
                f"Profit factor {pf:.2f} < {self._MIN_PROFIT_FACTOR} — edge gate not passed"
            )
        if wr < self._MIN_WIN_RATE:
            errors.append(
                f"Win rate {wr:.2%} < {self._MIN_WIN_RATE:.0%} — edge gate not passed"
            )

        # 4. Verify feature schema hash matches stored feature names.
        recomputed = FrozenModelSpec.compute_feature_hash(spec.feature_names)
        if recomputed != spec.feature_schema_hash:
            errors.append(
                f"feature_schema_hash mismatch: stored={spec.feature_schema_hash!r}, "
                f"recomputed={recomputed!r}"
            )

        if errors:
            error_list = "\n  - ".join(errors)
            logger.error(
                "model_artifact_validation_failed",
                version=spec.version,
                error_count=len(errors),
            )
            raise RegistryError(
                f"Model '{spec.version}' failed artifact validation "
                f"({len(errors)} error(s)):\n  - {error_list}\n"
                "Do NOT register stub or incomplete models."
            )

        logger.info(
            "model_artifact_validation_passed",
            version=spec.version,
            sharpe=sharpe,
            profit_factor=pf,
            win_rate=wr,
        )

    def register(self, spec: FrozenModelSpec) -> None:
        """Persist a spec to the registry.

        If ``spec.approval_status == "approved"`` any previously approved spec
        is demoted to ``"pending"`` so only one approved model exists at any
        time. For approved specs, artifact validation is enforced. [P0-FIX-10]

        Args:
            spec: The :class:`FrozenModelSpec` to register.

        Raises:
            RegistryError: If ``spec.version`` already exists, or if an approved
                spec fails artifact validation.
        """
        dest = self._spec_path(spec.version)
        if dest.exists():
            raise RegistryError(
                f"Model version '{spec.version}' already exists in registry. "
                "Production models are immutable — bump the version."
            )

        # Enforce version uniqueness across all registered specs (belt-and-suspenders
        # guard against races or renamed files where the path check is insufficient).
        existing_versions = {s.version for s in self.list_versions()}
        if spec.version in existing_versions:
            raise ValueError(f"Model version '{spec.version}' already registered")

        # P0-FIX-10: Validate artifacts for approved models — refuse to register
        # incomplete/stub models as production-ready.
        if spec.approval_status == "approved":
            self.validate_artifacts(spec)
            self._demote_current_approved()

        spec.save(dest)
        logger.info(
            "model_registered",
            version=spec.version,
            approval_status=spec.approval_status,
        )

    def get_production_model(self) -> FrozenModelSpec:
        """Return the single approved production model.

        Returns:
            The :class:`FrozenModelSpec` whose ``approval_status`` is
            ``"approved"``.

        Raises:
            RegistryError: If zero or more than one approved spec is found.
        """
        approved = [s for s in self.list_versions() if s.approval_status == "approved"]
        if not approved:
            raise RegistryError("No approved production model found in registry.")
        if len(approved) > 1:
            versions = [s.version for s in approved]
            raise RegistryError(
                f"Registry integrity violation: multiple approved models found: {versions}. "
                "Run registry repair."
            )
        return approved[0]

    def is_safe_to_retrain(self) -> bool:
        """Return True only when it is safe to trigger a retrain.

        Safe conditions (all must hold):
        - Current IST weekday is Sunday (``RETRAIN_DAY == 6``).
        - Current IST hour is ``>= RETRAIN_HOUR`` (20:00).
        - Market is not currently open (09:15–15:30 IST, though Sunday is
          always outside market hours — kept for defensive correctness).

        Returns:
            ``True`` if retraining is permitted right now, ``False`` otherwise.
        """
        now_ist = datetime.now(tz=IST)
        is_sunday = now_ist.weekday() == self.RETRAIN_DAY
        is_after_retrain_hour = now_ist.hour >= self.RETRAIN_HOUR
        is_market_open = self._is_market_open(now_ist)

        safe = is_sunday and is_after_retrain_hour and not is_market_open
        logger.debug(
            "retrain_safety_check",
            is_sunday=is_sunday,
            is_after_retrain_hour=is_after_retrain_hour,
            is_market_open=is_market_open,
            safe=safe,
            now_ist=now_ist.isoformat(),
        )
        return safe

    def list_versions(self) -> list[FrozenModelSpec]:
        """Return all registered specs sorted by version string.

        Returns:
            List of :class:`FrozenModelSpec` objects, newest-first by
            lexicographic version sort.
        """
        specs: list[FrozenModelSpec] = []
        for p in sorted(self._dir.glob("*.json"), reverse=True):
            try:
                specs.append(FrozenModelSpec.load(p))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "registry_spec_load_failed", path=str(p), error=str(exc)
                )
        return specs

    def validate_training_data(
        self,
        n_bars: int,
        n_trades: int,
        min_bars: int = DEFAULT_MIN_BARS,
        min_trades: int = DEFAULT_MIN_TRADES,
    ) -> None:
        """Raise InsufficientDataError if training data does not meet minimums.

        This check must be called before :meth:`MLEnsemble.train` to avoid
        training on sparse data that would produce meaningless walk-forward
        metrics.

        Args:
            n_bars: Number of OHLCV bars available for training.
            n_trades: Number of labeled trade samples available for training.
            min_bars: Minimum required bars (default: ``DEFAULT_MIN_BARS = 5000``).
            min_trades: Minimum required trades (default:
                ``DEFAULT_MIN_TRADES = 200``).

        Raises:
            InsufficientDataError: If ``n_bars < min_bars`` or
                ``n_trades < min_trades``.
        """
        errors: list[str] = []
        if n_bars < min_bars:
            errors.append(
                f"Insufficient bars: {n_bars} available, {min_bars} required."
            )
        if n_trades < min_trades:
            errors.append(
                f"Insufficient trades: {n_trades} available, {min_trades} required."
            )
        if errors:
            msg = " | ".join(errors)
            logger.error(
                "insufficient_training_data",
                n_bars=n_bars,
                n_trades=n_trades,
                min_bars=min_bars,
                min_trades=min_trades,
                errors=errors,
            )
            raise InsufficientDataError(msg)
        logger.info(
            "training_data_validated",
            n_bars=n_bars,
            n_trades=n_trades,
            min_bars=min_bars,
            min_trades=min_trades,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spec_path(self, version: str) -> Path:
        """Return the file path for a given version string.

        Args:
            version: Version string, e.g. ``"v2.3.1"``.

        Returns:
            Absolute path under ``registry_dir``.
        """
        safe_version = version.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_version}.json"

    def _demote_current_approved(self) -> None:
        """Overwrite any approved spec on disk with ``approval_status="pending"``.

        Uses object reconstruction to respect the frozen dataclass; writes the
        mutated dict directly to avoid creating a new dataclass instance (which
        would fail because the dataclass is frozen and ``approval_status`` is
        part of its identity hash).
        """
        for p in self._dir.glob("*.json"):
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("approval_status") == "approved":
                raw["approval_status"] = "pending"
                p.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                logger.info(
                    "model_demoted_to_pending",
                    version=raw.get("version"),
                    path=str(p),
                )

    @staticmethod
    def _is_market_open(now_ist: datetime) -> bool:
        """Return True if ``now_ist`` falls within the NSE trading session.

        Args:
            now_ist: Current datetime in IST.

        Returns:
            ``True`` if 09:15 <= time <= 15:30 on a weekday, else ``False``.
        """
        if now_ist.weekday() >= 5:  # Saturday or Sunday
            return False
        open_minutes = 9 * 60 + 15
        close_minutes = 15 * 60 + 30
        current_minutes = now_ist.hour * 60 + now_ist.minute
        return open_minutes <= current_minutes < close_minutes
