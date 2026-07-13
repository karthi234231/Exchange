"""Run the research-to-production readiness audit.

Usage:
    python scripts/audit_readiness.py
    python scripts/audit_readiness.py --output results/readiness_audit_latest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo_trading.readiness_audit import (
    BLOCKER,
    WARN,
    run_readiness_audit,
    write_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether current model/backtest artifacts are safe to trust."
    )
    parser.add_argument("--results", default="results/results_15min.json")
    parser.add_argument("--probabilities", default="results/probability_report_latest.json")
    parser.add_argument("--model-spec", default="models/trained/frozen_model_spec.json")
    parser.add_argument("--features", default="models/trained/15min/feat_cols.json")
    parser.add_argument("--output", default="results/readiness_audit_latest.json")
    args = parser.parse_args()

    audit = run_readiness_audit(
        results_path=args.results,
        probability_path=args.probabilities,
        model_spec_path=args.model_spec,
        feature_columns_path=args.features,
    )
    output_path = write_audit_report(audit, Path(args.output))

    print(f"Readiness status: {audit.status}")
    print(f"Report written: {output_path}")
    print(f"Blockers: {audit.summary.get('blockers', 0)}")
    print(f"Warnings: {audit.summary.get('warnings', 0)}")
    print()
    for finding in audit.findings:
        if finding.severity in {BLOCKER, WARN}:
            print(f"[{finding.severity}] {finding.check_id}: {finding.title}")
            print(f"  {finding.detail}")
            print(f"  Action: {finding.action}")

    return 1 if audit.status == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
