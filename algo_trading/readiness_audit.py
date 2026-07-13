"""Research-to-production readiness audit.

This module turns the most important trading-system caveats into deterministic
checks: data provenance, model metadata consistency, out-of-universe scoring,
probability sanity, and strategy gate status.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BLOCKER = "BLOCKER"
WARN = "WARN"
PASS = "PASS"


@dataclass(frozen=True)
class AuditFinding:
    """Single readiness finding."""

    check_id: str
    severity: str
    title: str
    detail: str
    action: str


@dataclass(frozen=True)
class ReadinessAudit:
    """Complete readiness audit result."""

    status: str
    findings: list[AuditFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": [asdict(finding) for finding in self.findings],
        }


def run_readiness_audit(
    *,
    results_path: str | Path = "results/results_15min.json",
    probability_path: str | Path = "results/probability_report_latest.json",
    model_spec_path: str | Path = "models/trained/frozen_model_spec.json",
    feature_columns_path: str | Path = "models/trained/15min/feat_cols.json",
) -> ReadinessAudit:
    """Run all available readiness checks.

    Missing optional artifacts are reported as warnings or blockers depending on
    whether the artifact is required to make a go-live decision.
    """
    results = _load_json(Path(results_path))
    probabilities = _load_json(Path(probability_path))
    model_spec = _load_json(Path(model_spec_path))
    feature_columns = _load_json(Path(feature_columns_path))

    findings: list[AuditFinding] = []
    findings.extend(_check_data_provenance(results))
    findings.extend(_check_model_metadata(model_spec, feature_columns))
    findings.extend(_check_scored_universe(model_spec, probabilities))
    findings.extend(_check_probability_distribution(probabilities))
    findings.extend(_check_strategy_gates(results))
    findings.extend(
        _check_required_artifacts(results, probabilities, model_spec, feature_columns)
    )

    status = _rollup_status(findings)
    summary = _build_summary(results, probabilities, model_spec, feature_columns, findings)
    return ReadinessAudit(status=status, findings=findings, summary=summary)


def write_audit_report(audit: ReadinessAudit, output_path: str | Path) -> Path:
    """Write an audit report as pretty JSON and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_read_error": f"Invalid JSON: {path}"}


def _check_data_provenance(results: Any | None) -> list[AuditFinding]:
    if results is None:
        return [
            AuditFinding(
                "DATA_ARTIFACT_MISSING",
                WARN,
                "Backtest result artifact missing",
                "No results artifact was available for provenance checks.",
                "Generate a fresh validation report before trusting strategy metrics.",
            )
        ]

    source_text = " ".join(
        str(results.get(key, ""))
        for key in ("data_source", "bar_type")
        if isinstance(results, dict)
    ).lower()
    if any(token in source_text for token in ("synthetic", "brownian", "disaggregation")):
        return [
            AuditFinding(
                "SYNTHETIC_INTRADAY_DATA",
                BLOCKER,
                "Backtest uses synthetic intraday data",
                "The current 15-minute result describes Brownian Bridge disaggregation or synthetic bars.",
                "Do not use these metrics for live trading; rerun on real intraday bars.",
            )
        ]
    if not source_text.strip():
        return [
            AuditFinding(
                "DATA_PROVENANCE_UNKNOWN",
                BLOCKER,
                "Data provenance is unknown",
                "The result artifact does not identify whether bars are real, synthetic, or derived.",
                "Add explicit provenance metadata and block production reports when provenance is unknown.",
            )
        ]
    return [
        AuditFinding(
            "DATA_PROVENANCE_PRESENT",
            PASS,
            "Data provenance is declared",
            f"Result artifact declares: {source_text}",
            "Keep provenance attached to all model and backtest artifacts.",
        )
    ]


def _check_model_metadata(model_spec: Any | None, feature_columns: Any | None) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if not isinstance(model_spec, dict):
        return [
            AuditFinding(
                "MODEL_SPEC_MISSING",
                BLOCKER,
                "Model spec missing or unreadable",
                "The audit could not read the frozen model specification.",
                "Freeze model metadata with training symbols, features, label method, and training period.",
            )
        ]

    training_symbols = model_spec.get("training_symbols") or []
    if not training_symbols:
        findings.append(
            AuditFinding(
                "TRAINING_UNIVERSE_MISSING",
                BLOCKER,
                "Training universe missing",
                "The model spec does not list the symbols used for training.",
                "Persist training symbols and block scoring when the universe is unknown.",
            )
        )

    spec_features = model_spec.get("feature_columns") or []
    actual_features = feature_columns if isinstance(feature_columns, list) else []
    if actual_features and spec_features and list(spec_features) != list(actual_features):
        findings.append(
            AuditFinding(
                "FEATURE_SCHEMA_MISMATCH",
                BLOCKER,
                "Model spec features differ from inference features",
                f"Spec has {len(spec_features)} features, while the 15-minute feature list has {len(actual_features)}.",
                "Regenerate the frozen model spec from the exact trained pipeline before scoring.",
            )
        )
    elif actual_features and spec_features:
        findings.append(
            AuditFinding(
                "FEATURE_SCHEMA_MATCH",
                PASS,
                "Model feature schema matches",
                f"Both artifacts list {len(actual_features)} features in the same order.",
                "Keep feature order immutable between training and inference.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "FEATURE_SCHEMA_INCOMPLETE",
                BLOCKER,
                "Feature schema metadata incomplete",
                "The audit could not compare model spec features with inference features.",
                "Persist the exact feature column list in both the model spec and model directory.",
            )
        )
    return findings


def _check_scored_universe(model_spec: Any | None, probabilities: Any | None) -> list[AuditFinding]:
    if not isinstance(model_spec, dict) or not isinstance(probabilities, dict):
        return []

    training_symbols = {str(symbol).upper() for symbol in model_spec.get("training_symbols", [])}
    scored_symbols = {
        str(row.get("symbol", "")).upper()
        for row in probabilities.get("all_rows", [])
        if isinstance(row, dict) and row.get("symbol")
    }
    if not scored_symbols:
        return [
            AuditFinding(
                "SCORED_UNIVERSE_MISSING",
                WARN,
                "No scored symbols found",
                "The probability artifact does not include scored rows.",
                "Include all scored symbols in probability reports.",
            )
        ]

    extra = sorted(scored_symbols - training_symbols)
    if extra:
        preview = ", ".join(extra[:12])
        suffix = "..." if len(extra) > 12 else ""
        return [
            AuditFinding(
                "OUT_OF_UNIVERSE_SCORING",
                BLOCKER,
                "Probability report scores symbols outside the training universe",
                f"{len(extra)} scored symbols were not listed in model training metadata: {preview}{suffix}",
                "Restrict scoring to validated symbols or retrain/validate the model for the wider universe.",
            )
        ]
    return [
        AuditFinding(
            "SCORED_UNIVERSE_MATCH",
            PASS,
            "Scored universe matches training metadata",
            f"All {len(scored_symbols)} scored symbols appear in the model spec.",
            "Keep this check in the inference pipeline.",
        )
    ]


def _check_probability_distribution(probabilities: Any | None) -> list[AuditFinding]:
    if not isinstance(probabilities, dict):
        return []

    findings: list[AuditFinding] = []
    averages = probabilities.get("average_probabilities") or {}
    hold_prob = float(averages.get("HOLD", 0.0) or 0.0)
    if hold_prob < 0.01:
        findings.append(
            AuditFinding(
                "HOLD_PROBABILITY_COLLAPSE",
                WARN,
                "Average HOLD probability is near zero",
                f"Average HOLD probability is {hold_prob:.6f}, which suggests calibration or label-distribution issues.",
                "Review labels, class weights, probability calibration, and no-trade thresholds.",
            )
        )

    rows = [row for row in probabilities.get("all_rows", []) if isinstance(row, dict)]
    invalid_rows = [
        row.get("symbol", "?")
        for row in rows
        if abs(
            float(row.get("sell_prob", 0.0) or 0.0)
            + float(row.get("hold_prob", 0.0) or 0.0)
            + float(row.get("buy_prob", 0.0) or 0.0)
            - 1.0
        )
        > 0.01
    ]
    if invalid_rows:
        findings.append(
            AuditFinding(
                "PROBABILITIES_NOT_NORMALIZED",
                BLOCKER,
                "Probability rows do not sum to one",
                f"{len(invalid_rows)} rows have class probabilities outside a 1% tolerance.",
                "Fix inference output before using confidence thresholds.",
            )
        )
    elif rows:
        findings.append(
            AuditFinding(
                "PROBABILITIES_NORMALIZED",
                PASS,
                "Probability rows are normalized",
                f"Checked {len(rows)} probability rows.",
                "Continue validating every probability artifact.",
            )
        )
    return findings


def _check_strategy_gates(results: Any | None) -> list[AuditFinding]:
    if not isinstance(results, dict):
        return []

    per_symbol = results.get("per_symbol_backtest") or {}
    if not isinstance(per_symbol, dict) or not per_symbol:
        return [
            AuditFinding(
                "STRATEGY_GATES_MISSING",
                WARN,
                "Per-symbol strategy gates missing",
                "The result artifact does not include symbol-level pass/fail gates.",
                "Record every validation gate in the backtest output.",
            )
        ]

    passed = sorted(
        symbol
        for symbol, metrics in per_symbol.items()
        if isinstance(metrics, dict) and metrics.get("passes_gate") is True
    )
    total = len(per_symbol)
    if len(passed) < total:
        return [
            AuditFinding(
                "SYMBOL_GATES_FAILED",
                BLOCKER,
                "Most symbols fail production gates",
                f"{len(passed)}/{total} symbols passed current strategy gates.",
                "Paper trade only passing symbols after real-data validation; reject failed symbols.",
            )
        ]
    return [
        AuditFinding(
            "SYMBOL_GATES_PASS",
            PASS,
            "All symbols pass strategy gates",
            f"All {total} symbols passed the recorded validation gates.",
            "Proceed only after independent benchmark and paper-trading checks.",
        )
    ]


def _check_required_artifacts(
    results: Any | None,
    probabilities: Any | None,
    model_spec: Any | None,
    feature_columns: Any | None,
) -> list[AuditFinding]:
    missing = []
    if results is None:
        missing.append("results/results_15min.json")
    if probabilities is None:
        missing.append("results/probability_report_latest.json")
    if model_spec is None:
        missing.append("models/trained/frozen_model_spec.json")
    if feature_columns is None:
        missing.append("models/trained/15min/feat_cols.json")
    if not missing:
        return []
    return [
        AuditFinding(
            "READINESS_ARTIFACTS_MISSING",
            BLOCKER,
            "Required readiness artifacts are missing",
            "Missing: " + ", ".join(missing),
            "Generate the required artifacts before making a trading decision.",
        )
    ]


def _rollup_status(findings: list[AuditFinding]) -> str:
    if any(finding.severity == BLOCKER for finding in findings):
        return "BLOCKED"
    if any(finding.severity == WARN for finding in findings):
        return "WARN"
    return "PASS"


def _build_summary(
    results: Any | None,
    probabilities: Any | None,
    model_spec: Any | None,
    feature_columns: Any | None,
    findings: list[AuditFinding],
) -> dict[str, Any]:
    per_symbol = results.get("per_symbol_backtest", {}) if isinstance(results, dict) else {}
    passed_symbols = [
        symbol
        for symbol, metrics in per_symbol.items()
        if isinstance(metrics, dict) and metrics.get("passes_gate") is True
    ]
    scored_rows = probabilities.get("all_rows", []) if isinstance(probabilities, dict) else []
    training_symbols = model_spec.get("training_symbols", []) if isinstance(model_spec, dict) else []
    return {
        "status_reason": _status_reason(findings),
        "blockers": sum(1 for finding in findings if finding.severity == BLOCKER),
        "warnings": sum(1 for finding in findings if finding.severity == WARN),
        "passes": sum(1 for finding in findings if finding.severity == PASS),
        "training_symbol_count": len(training_symbols),
        "scored_symbol_count": len(scored_rows) if isinstance(scored_rows, list) else 0,
        "feature_count_from_model_spec": len(model_spec.get("feature_columns", []))
        if isinstance(model_spec, dict)
        else 0,
        "feature_count_from_inference": len(feature_columns) if isinstance(feature_columns, list) else 0,
        "strategy_symbols_passed": len(passed_symbols),
        "strategy_symbols_total": len(per_symbol) if isinstance(per_symbol, dict) else 0,
    }


def _status_reason(findings: list[AuditFinding]) -> str:
    blockers = [finding.title for finding in findings if finding.severity == BLOCKER]
    if blockers:
        return "; ".join(blockers[:3])
    warnings = [finding.title for finding in findings if finding.severity == WARN]
    if warnings:
        return "; ".join(warnings[:3])
    return "All readiness checks passed."
