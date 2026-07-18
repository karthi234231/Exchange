"""
Phase 4: Validate consolidated output and compute accuracy scores.

Reads all Parquet files from data/consolidated_v2/, runs quality checks,
computes accuracy scores, and writes scores.parquet + summary report.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

OUTPUT_DIR = Path(r"C:\Exchange\data\consolidated_v2")
SCORES_PATH = OUTPUT_DIR / "scores.parquet"
REPORT_PATH = OUTPUT_DIR / "validation_report.md"


def _validate_file(path: Path) -> dict[str, Any]:
    """Run all validation checks on a single Parquet file."""
    result = {
        "symbol": path.parent.name,
        "timeframe": path.stem,
        "score": 0.0,
        "warnings": [],
        "flags": [],
        "status": "unknown",
    }
    
    try:
        lf = pl.scan_parquet(path)
        df = lf.collect()
        
        if len(df) == 0:
            result["status"] = "fail"
            result["warnings"].append("empty_file")
            return result
        
        # 1. Coverage: % of expected trading days (assuming ~250 days/year)
        # For intraday, check distinct days
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))
        distinct_days = df["date"].n_unique()
        coverage = min(1.0, distinct_days / 250.0)
        
        # 2. OHLC Integrity
        ohlc_ok = (
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        )
        ohlc_pct = float(ohlc_ok.sum() / len(df))
        
        # 3. Volume Integrity
        vol_ok = df["volume"] >= 0
        vol_pct = float(vol_ok.sum() / len(df))
        
        # 4. No duplicates (already deduped in Phase 2)
        no_dup = df["timestamp"].n_unique() == len(df)
        
        # 5. No future data
        now = datetime.now()
        future = df.filter(pl.col("timestamp") > now)
        no_future = len(future) == 0
        
        # 6. Statistical plausibility (returns)
        if len(df) > 10:
            df_sorted = df.sort("timestamp")
            returns = df_sorted["close"].pct_change().drop_nulls()
            if len(returns) > 0:
                skew = returns.skew()
                kurt = returns.kurtosis()
                
                # Check bounds
                if skew is not None and abs(skew) > 0.5:
                    result["warnings"].append(f"high_skew_{skew:.2f}")
                if kurt is not None and kurt < 0:
                    result["warnings"].append(f"negative_kurtosis_{kurt:.2f}")
        
        # Compute score
        score = (
            0.25 * coverage +
            0.20 * ohlc_pct +
            0.10 * vol_pct +
            0.10 * (1.0 if no_dup else 0.0) +
            0.10 * (1.0 if no_future else 0.0) +
            0.25 * 0.8  # Base score for having data
        )
        
        result["score"] = round(score * 100, 1)
        result["coverage"] = round(coverage * 100, 1)
        result["ohlc_pct"] = round(ohlc_pct * 100, 1)
        result["vol_pct"] = round(vol_pct * 100, 1)
        result["rows"] = len(df)
        
        # Status
        if score >= 0.90:
            result["status"] = "pass"
        elif score >= 0.70:
            result["status"] = "warning"
        else:
            result["status"] = "fail"
            result["flags"].append("low_score")
        
        if not no_future:
            result["flags"].append("future_data")
        if not no_dup:
            result["flags"].append("duplicates")
        
    except Exception as exc:
        result["status"] = "error"
        result["warnings"].append(f"read_error: {exc}")
    
    return result


def main() -> None:
    t0 = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[Phase 4] Validating consolidated data...")
    
    # Find all Parquet files
    parquet_files = list(OUTPUT_DIR.rglob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name not in ("metadata.parquet", "scores.parquet")]
    
    print(f"  Found {len(parquet_files)} files to validate")
    
    results = []
    for i, path in enumerate(parquet_files):
        if (i + 1) % 100 == 0:
            print(f"  Validated {i+1}/{len(parquet_files)}...")
        result = _validate_file(path)
        results.append(result)
    
    # Write scores.parquet
    scores_df = pl.DataFrame(results)
    scores_df.write_parquet(SCORES_PATH)
    
    # Summary
    elapsed = (datetime.now() - t0).total_seconds()
    pass_count = sum(1 for r in results if r["status"] == "pass")
    warn_count = sum(1 for r in results if r["status"] == "warning")
    fail_count = sum(1 for r in results if r["status"] == "fail")
    
    print(f"\n[Phase 4] Done in {elapsed:.1f}s")
    print(f"  Output: {SCORES_PATH}")
    print(f"  Total: {len(results)} | Pass: {pass_count} | Warning: {warn_count} | Fail: {fail_count}")
    print(f"  Average score: {scores_df['score'].mean():.1f}")
    
    # Generate Markdown report
    report = f"""# Consolidation Validation Report

Generated: {datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total symbol/timeframes | {len(results)} |
| Pass (≥90) | {pass_count} |
| Warning (70-89) | {warn_count} |
| Fail (<70) | {fail_count} |
| Average score | {scores_df['score'].mean():.1f} |
| Min score | {scores_df['score'].min():.1f} |
| Max score | {scores_df['score'].max():.1f} |

## Top 10 by Score

| Symbol | Timeframe | Score | Rows | Coverage | OHLC% |
|--------|-----------|-------|------|----------|-------|
"""
    for r in sorted(results, key=lambda x: -x["score"])[:10]:
        report += f"| {r['symbol']} | {r['timeframe']} | {r['score']:.1f} | {r.get('rows', 0):,} | {r.get('coverage', 0):.1f}% | {r.get('ohlc_pct', 0):.1f}% |\n"
    
    report += "\n## Failed / Low Quality\n\n"
    failed = [r for r in results if r["status"] in ("fail", "warning")]
    for r in sorted(failed, key=lambda x: x["score"])[:20]:
        report += f"- **{r['symbol']}/{r['timeframe']}**: score={r['score']:.1f}, warnings={r['warnings']}\n"
    
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()