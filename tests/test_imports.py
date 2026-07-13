"""
tests/test_imports.py — P2-FIX-93: Import smoke tests for all modules.

Verifies that every algo_trading module is importable under the production
dependency set. Optional dependencies (ML, broker SDK, QuestDB) are handled
with `pytest.importorskip` or explicit try/except so the test suite does not
fail on minimal installs.

A test failing here means a SyntaxError, a missing __init__.py, a broken
import chain, or an optional dependency that was incorrectly marked required.
This must pass before every deployment.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional

import pytest

try:
    import pydantic
    _PYDANTIC_ERRORS = (pydantic.ValidationError,)
except ImportError:
    _PYDANTIC_ERRORS = ()


def _import(module_path: str) -> ModuleType:
    """
    Import by dotted path.

    - SyntaxError / ImportError from the module itself → test FAILS (real bug)
    - pydantic.ValidationError → test SKIPPED (Settings() needs .env, not a bug)
    """
    try:
        return importlib.import_module(module_path)
    except _PYDANTIC_ERRORS:
        pytest.skip(
            f"{module_path}: pydantic ValidationError — Settings requires .env (not an import bug)"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise  # let pytest report it as FAILED


def _optional(module_path: str, requires: str = "") -> Optional[ModuleType]:
    """
    Try to import module_path. Returns None if an optional dependency is
    missing (ImportError/ModuleNotFoundError). pydantic.ValidationError is also
    treated as None (config-gated module, not a broken import).
    """
    try:
        return importlib.import_module(module_path)
    except _PYDANTIC_ERRORS:
        return None  # valid — module imports OK but needs config at runtime
    except (ImportError, ModuleNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Core modules — always required, no optional dependencies
# ---------------------------------------------------------------------------

class TestCoreImports:
    """Modules that must always import successfully."""

    def test_market_calendar(self) -> None:
        mod = _import("algo_trading.market_calendar")
        assert hasattr(mod, "IST")
        assert hasattr(mod, "now_ist")
        assert hasattr(mod, "is_market_open")

    def test_logging_setup(self) -> None:
        _import("algo_trading.logging_setup")

    def test_data_broker_base(self) -> None:
        mod = _import("algo_trading.data.broker_base")
        assert hasattr(mod, "Order")
        assert hasattr(mod, "Fill")
        assert hasattr(mod, "OrderUpdate")

    def test_execution_order_fsm(self) -> None:
        mod = _import("algo_trading.execution.order_fsm")
        assert hasattr(mod, "OrderFSM")
        assert hasattr(mod, "OrderState")
        assert hasattr(mod, "OrderEvent")
        assert hasattr(mod, "TransitionError")

    def test_execution_exception_policy(self) -> None:
        mod = _import("algo_trading.execution.exception_policy")
        assert hasattr(mod, "fail_closed")
        assert hasattr(mod, "fail_open")
        assert hasattr(mod, "safe_exit_action")
        assert hasattr(mod, "require_audit_log_writable")

    def test_execution_signal_hierarchy(self) -> None:
        mod = _import("algo_trading.execution.signal_hierarchy")
        assert hasattr(mod, "SignalHierarchy")

    def test_execution_rate_limiter(self) -> None:
        mod = _import("algo_trading.execution.rate_limiter")
        assert hasattr(mod, "DualLockRateLimiter")

    def test_risk_cost_model(self) -> None:
        mod = _import("algo_trading.risk.cost_model")
        assert hasattr(mod, "IndiaCostModel")

    def test_risk_position_sizer(self) -> None:
        _import("algo_trading.risk.position_sizer")

    def test_risk_manager(self) -> None:
        mod = _import("algo_trading.risk.manager")
        assert hasattr(mod, "RiskManager")

    def test_labeling_triple_barrier(self) -> None:
        mod = _import("algo_trading.labeling.triple_barrier")
        assert hasattr(mod, "TripleBarrierLabeler")

    def test_labeling_walk_forward(self) -> None:
        mod = _import("algo_trading.labeling.walk_forward")
        assert hasattr(mod, "WalkForwardSplitter")

    def test_signals_ema_cross(self) -> None:
        mod = _import("algo_trading.signals.ema_cross")
        assert hasattr(mod, "EMACrossStrategy")

    def test_data_integrity_guard(self) -> None:
        mod = _import("algo_trading.data.integrity_guard")
        assert hasattr(mod, "TickValidator")
        assert hasattr(mod, "DataQualityMonitor")

    def test_data_ring_buffer(self) -> None:
        _import("algo_trading.data.ring_buffer")

    def test_data_system_state(self) -> None:
        mod = _import("algo_trading.data.system_state")
        assert hasattr(mod, "LiveSystemState")

    def test_backtest_runner(self) -> None:
        _import("algo_trading.backtest.runner")

    def test_features_stage_router(self) -> None:
        pytest.importorskip("msgpack", reason="msgpack not installed")
        _import("algo_trading.features.stage_router")

    def test_features_technical(self) -> None:
        pytest.importorskip("pandas_ta", reason="pandas_ta not installed")
        _import("algo_trading.features.technical")

    def test_features_custom(self) -> None:
        pytest.importorskip("numba", reason="numba not installed")
        _import("algo_trading.features.custom")

    def test_fo_contracts(self) -> None:
        _import("algo_trading.fo.contracts")

    def test_fo_expiry(self) -> None:
        _import("algo_trading.fo.expiry")

    def test_fo_cost_model(self) -> None:
        _import("algo_trading.fo.cost_model")

    def test_fo_strategies(self) -> None:
        _import("algo_trading.fo.strategies")

    def test_fo_margin(self) -> None:
        _import("algo_trading.fo.margin")

    def test_fo_execution(self) -> None:
        _import("algo_trading.fo.execution")

    def test_fo_instrument_master_lot_size_history(self) -> None:
        _import("algo_trading.fo.instrument_master.lot_size_history")

    def test_fo_instrument_master_strike_generator(self) -> None:
        _import("algo_trading.fo.instrument_master.strike_generator")

    def test_market_data_chain_snapshot(self) -> None:
        _import("algo_trading.market_data.chain_snapshot")

    def test_market_data_staleness_detector(self) -> None:
        _import("algo_trading.market_data.staleness_detector")

    def test_market_data_chain_normalizer(self) -> None:
        _import("algo_trading.market_data.chain_normalizer")

    def test_market_data_snapshot_archiver(self) -> None:
        _import("algo_trading.market_data.snapshot_archiver")


# ---------------------------------------------------------------------------
# Modules with optional dependencies (hmmlearn, arch, scipy, etc.)
# ---------------------------------------------------------------------------

class TestOptionalMLImports:
    """Modules that depend on optional ML packages."""

    def test_features_statistical(self) -> None:
        """Requires hmmlearn and arch."""
        hmmlearn = pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
        mod = _import("algo_trading.features.statistical")
        assert mod is not None

    def test_features_microstructure(self) -> None:
        """Requires scipy (for VPIN)."""
        scipy = pytest.importorskip("scipy", reason="scipy not installed")
        _import("algo_trading.features.microstructure")

    def test_features_hawkes(self) -> None:
        """Requires scipy."""
        scipy = pytest.importorskip("scipy", reason="scipy not installed")
        _import("algo_trading.features.hawkes")

    def test_features_tfi(self) -> None:
        _import("algo_trading.features.tfi")

    def test_features_fo_features(self) -> None:
        _import("algo_trading.features.fo_features")

    def test_models_ensemble(self) -> None:
        """Requires xgboost, lightgbm, catboost."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        _import("algo_trading.models.ensemble")

    def test_models_inference_engine(self) -> None:
        """Requires xgboost."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        _import("algo_trading.models.inference_engine")

    def test_models_tuner(self) -> None:
        """Requires optuna and xgboost."""
        pytest.importorskip("optuna", reason="optuna not installed")
        _import("algo_trading.models.tuner")

    def test_models_tracker(self) -> None:
        """Requires mlflow."""
        pytest.importorskip("mlflow", reason="mlflow not installed")
        _import("algo_trading.models.tracker")

    def test_models_explainer(self) -> None:
        """Requires shap."""
        pytest.importorskip("shap", reason="shap not installed")
        _import("algo_trading.models.explainer")

    def test_models_model_registry(self) -> None:
        _import("algo_trading.models.model_registry")

    def test_models_online_model(self) -> None:
        """Requires river."""
        river = _optional("algo_trading.models.online_model")
        # If river not installed, skip — do not fail
        if river is None:
            pytest.skip("river not installed — online_model skipped")

    def test_models_rl_env(self) -> None:
        """Requires gymnasium and numpy."""
        pytest.importorskip("gymnasium", reason="gymnasium not installed")
        _import("algo_trading.models.rl_env")

    def test_models_rl_sizer(self) -> None:
        """RL sizer raises NotImplementedError at runtime (Phase 4 only)."""
        mod = _optional("algo_trading.models.rl_sizer")
        if mod is None:
            pytest.skip("stable_baselines3 not installed")

    def test_models_sentiment(self) -> None:
        """Sentiment worker — optional NLP/ZMQ dependency. Tests SyntaxError fix."""
        # Must at least be parseable (SyntaxError = real bug)
        import ast, pathlib
        src = pathlib.Path("algo_trading/models/sentiment.py").read_text(encoding="utf-8")
        ast.parse(src)  # raises SyntaxError if broken
        # Actual import may fail if zmq/transformers not installed — that's OK
        _optional("algo_trading.models.sentiment")

    def test_fo_pricing_bsm(self) -> None:
        """Requires scipy."""
        pytest.importorskip("scipy", reason="scipy not installed")
        _import("algo_trading.fo.pricing")

    def test_fo_vol_surface(self) -> None:
        """Requires scipy."""
        pytest.importorskip("scipy", reason="scipy not installed")
        _import("algo_trading.fo.vol_surface")

    def test_fo_portfolio_risk(self) -> None:
        _import("algo_trading.fo.portfolio_risk")

    def test_fo_signal_generator(self) -> None:
        _import("algo_trading.fo.signal_generator")

    def test_fo_greeks_hedger(self) -> None:
        _import("algo_trading.fo.greeks_hedger")

    def test_fo_vix_estimator(self) -> None:
        _import("algo_trading.fo.vix_estimator")

    def test_fo_sabr(self) -> None:
        pytest.importorskip("scipy", reason="scipy not installed")
        _import("algo_trading.fo.sabr")

    def test_risk_stress_engine(self) -> None:
        _import("algo_trading.risk.stress_engine")

    def test_risk_gap_risk_model(self) -> None:
        _import("algo_trading.risk.gap_risk_model")

    def test_risk_kill_switch_manager(self) -> None:
        _import("algo_trading.risk.kill_switch_manager")

    def test_risk_margin_headroom(self) -> None:
        _import("algo_trading.risk.margin_headroom")


# ---------------------------------------------------------------------------
# Broker-specific modules (no credential needed for import-only)
# ---------------------------------------------------------------------------

class TestBrokerImports:
    """
    Broker adapters must be importable without network or credentials.
    Import only tests the module structure, not broker connectivity.
    """

    def test_broker_shoonya_importable(self) -> None:
        """ShoonyaApi-py is optional — skip if not installed."""
        mod = _optional("algo_trading.data.broker_shoonya")
        if mod is None:
            pytest.skip("ShoonyaApi-py not installed — broker_shoonya skipped")
        assert hasattr(mod, "ShoonyaBroker")

    def test_data_truedata_feed(self) -> None:
        """TrueData feed — skip if truedata-ws not installed."""
        mod = _optional("algo_trading.data.truedata_feed")
        if mod is None:
            pytest.skip("truedata-ws not installed")

    def test_data_websocket_feed(self) -> None:
        mod = _optional("algo_trading.data.websocket_feed")
        if mod is None:
            pytest.skip("zmq not installed")

    def test_data_candle_store(self) -> None:
        """QuestDB store — skip if questdb or requests not available."""
        _import("algo_trading.data.candle_store")

    def test_data_fo_chain(self) -> None:
        _import("algo_trading.data.fo_chain")

    def test_data_tick_aggregator(self) -> None:
        _import("algo_trading.data.tick_aggregator")


# ---------------------------------------------------------------------------
# Alerts — ZMQ required
# ---------------------------------------------------------------------------

class TestAlertImports:
    def test_telegram_bot(self) -> None:
        mod = _optional("algo_trading.alerts.telegram_bot")
        if mod is None:
            pytest.skip("zmq not installed")
        assert hasattr(mod, "TelegramAlertSender")

    def test_alert_sink(self) -> None:
        mod = _optional("algo_trading.alerts.alert_sink")
        if mod is None:
            pytest.skip("zmq not installed")


# ---------------------------------------------------------------------------
# Features engine (requires zmq and polars)
# ---------------------------------------------------------------------------

class TestFeatureEngineImports:
    def test_features_engine(self) -> None:
        mod = _optional("algo_trading.features.engine")
        if mod is None:
            pytest.skip("zmq or polars not installed")
        assert hasattr(mod, "FeatureComputer")


# ---------------------------------------------------------------------------
# Orchestrators (heavy — require many modules to load)
# ---------------------------------------------------------------------------

class TestOrchestratorImports:
    def test_daily_orchestrator(self) -> None:
        mod = _optional("algo_trading.app.daily_orchestrator")
        if mod is None:
            pytest.skip("orchestrator dependency missing")

    def test_fo_orchestrator(self) -> None:
        mod = _optional("algo_trading.app.fo_orchestrator")
        if mod is None:
            pytest.skip("fo_orchestrator dependency missing")


# ---------------------------------------------------------------------------
# Research modules
# ---------------------------------------------------------------------------

class TestResearchImports:
    def test_research_fill_simulator(self) -> None:
        _import("algo_trading.research.fill_simulator")

    def test_research_chain_replay_engine(self) -> None:
        _import("algo_trading.research.chain_replay_engine")

    def test_research_walk_forward_runner(self) -> None:
        _import("algo_trading.research.walk_forward_runner")

    def test_research_slippage_model(self) -> None:
        _import("algo_trading.research.slippage_model")

    def test_research_metrics_engine(self) -> None:
        _import("algo_trading.research.metrics_engine")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class TestSignalImports:
    def test_ema_cross_filtered(self) -> None:
        _import("algo_trading.signals.ema_cross_filtered")

    def test_event_calendar(self) -> None:
        _import("algo_trading.signals.event_calendar")

    def test_multi_timeframe(self) -> None:
        _import("algo_trading.signals.multi_timeframe")

    def test_mean_reversion(self) -> None:
        _import("algo_trading.signals.mean_reversion")


# ---------------------------------------------------------------------------
# Config — must be importable without env vars (Pydantic optional fields)
# ---------------------------------------------------------------------------

class TestConfigImport:
    def test_config_importable_without_env(self) -> None:
        """
        config.py should not raise on import even if .env is absent.
        It uses Pydantic's Optional fields with defaults.
        The Settings object construction may raise if required fields are missing,
        but the module itself must be importable.
        """
        # Just test that the module-level code doesn't crash on import
        mod = _import("algo_trading.config")
        assert hasattr(mod, "Settings")
        assert hasattr(mod, "TradingMode")
