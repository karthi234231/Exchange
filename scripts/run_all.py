"""
Orchestrator: run the full consolidation pipeline (Phase 1 → 2 → 4).

Usage:
    python scripts/run_all.py              # Full run
    python scripts/run_all.py --limit 10   # Test with 10 groups only
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from build_inventory import main as build_inventory
from consolidate_groups import main as consolidate_groups
from validate_output import main as validate_output


def main(limit_groups: int | None = None) -> None:
    print("=" * 80)
    print("  NSE DATA CONSOLIDATION V2 — FULL PIPELINE")
    print("=" * 80)
    
    # Phase 1: Build inventory (skip if already exists)
    print("\n" + "=" * 80)
    if (Path(r"C:\Exchange\data\consolidated_v2") / "metadata.parquet").exists():
        print("[Phase 1/3] SKIP – metadata already exists")
    else:
        print("[Phase 1/3] BUILD INVENTORY")
        print("=" * 80)
        build_inventory()
    
    # Phase 2: Consolidate groups
    print("\n" + "=" * 80)
    print("[Phase 2/3] CONSOLIDATE GROUPS")
    print("=" * 80)
    consolidate_groups(limit_groups=limit_groups)
    
    # Phase 4: Validate output
    print("\n" + "=" * 80)
    print("[Phase 4/3] VALIDATE OUTPUT")
    print("=" * 80)
    validate_output()
    
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Output directory: C:\\Exchange\\data\\consolidated_v2")
    print(f"  Metadata:        metadata.parquet")
    print(f"  Scores:          scores.parquet")
    print(f"  Report:          validation_report.md")


if __name__ == "__main__":
    limit = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
    main(limit_groups=limit)