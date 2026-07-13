from __future__ import annotations

import json
from pathlib import Path

from algo_trading.readiness_audit import run_readiness_audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_readiness_audit_blocks_current_failure_modes(tmp_path: Path) -> None:
    results = {
        "data_source": "Brownian Bridge disaggregation of 5-year daily data",
        "bar_type": "synthetic 15-min",
        "per_symbol_backtest": {
            "TCS": {"passes_gate": True},
            "INFY": {"passes_gate": False},
        },
    }
    probabilities = {
        "average_probabilities": {"SELL": 0.4, "HOLD": 0.0001, "BUY": 0.5999},
        "all_rows": [
            {"symbol": "TCS", "sell_prob": 0.2, "hold_prob": 0.1, "buy_prob": 0.7},
            {"symbol": "INFY", "sell_prob": 0.3, "hold_prob": 0.0, "buy_prob": 0.7},
        ],
    }
    model_spec = {
        "training_symbols": ["TCS"],
        "feature_columns": ["feat_rsi"],
    }
    feature_columns = ["feat_rsi", "feat_atr"]

    results_path = tmp_path / "results.json"
    probability_path = tmp_path / "probabilities.json"
    model_spec_path = tmp_path / "model_spec.json"
    features_path = tmp_path / "features.json"

    _write_json(results_path, results)
    _write_json(probability_path, probabilities)
    _write_json(model_spec_path, model_spec)
    _write_json(features_path, feature_columns)

    audit = run_readiness_audit(
        results_path=results_path,
        probability_path=probability_path,
        model_spec_path=model_spec_path,
        feature_columns_path=features_path,
    )

    finding_ids = {finding.check_id for finding in audit.findings}
    assert audit.status == "BLOCKED"
    assert "SYNTHETIC_INTRADAY_DATA" in finding_ids
    assert "FEATURE_SCHEMA_MISMATCH" in finding_ids
    assert "OUT_OF_UNIVERSE_SCORING" in finding_ids
    assert "HOLD_PROBABILITY_COLLAPSE" in finding_ids
    assert "SYMBOL_GATES_FAILED" in finding_ids


def test_readiness_audit_passes_clean_artifacts(tmp_path: Path) -> None:
    results = {
        "data_source": "Real Shoonya 15-minute historical bars",
        "bar_type": "real 15-min",
        "per_symbol_backtest": {
            "TCS": {"passes_gate": True},
        },
    }
    probabilities = {
        "average_probabilities": {"SELL": 0.3, "HOLD": 0.2, "BUY": 0.5},
        "all_rows": [
            {"symbol": "TCS", "sell_prob": 0.3, "hold_prob": 0.2, "buy_prob": 0.5},
        ],
    }
    model_spec = {
        "training_symbols": ["TCS"],
        "feature_columns": ["feat_rsi", "feat_atr"],
    }
    feature_columns = ["feat_rsi", "feat_atr"]

    results_path = tmp_path / "results.json"
    probability_path = tmp_path / "probabilities.json"
    model_spec_path = tmp_path / "model_spec.json"
    features_path = tmp_path / "features.json"

    _write_json(results_path, results)
    _write_json(probability_path, probabilities)
    _write_json(model_spec_path, model_spec)
    _write_json(features_path, feature_columns)

    audit = run_readiness_audit(
        results_path=results_path,
        probability_path=probability_path,
        model_spec_path=model_spec_path,
        feature_columns_path=features_path,
    )

    assert audit.status == "PASS"
    assert audit.summary["blockers"] == 0
    assert audit.summary["warnings"] == 0
