"""
scripts/kill_switch.py — Operator CLI for manual kill-switch control. [P1-FIX-29]

Usage:
    python -m algo_trading.scripts.kill_switch trigger "Daily loss exceeded manually"
    python -m algo_trading.scripts.kill_switch clear "Reviewed positions, safe to resume"
    python -m algo_trading.scripts.kill_switch status

How it works:
    trigger: Writes the reason to state/kill_switch.trigger and logs it.
             The live RiskManager will read this via can_trade() → trigger_kill_switch().

    clear:   Writes the reason to state/kill_switch_clear.sentinel.
             The live RiskManager reads this via check_kill_switch_clear_sentinel()
             on every can_trade() call and clears the kill switch.

    status:  Reads state/live_state.json to report current kill-switch state.

Safety:
    - Sentinel files are written atomically (write-then-rename).
    - The trigger file is a one-shot: once read by the live system it is deleted.
    - This script never connects to the broker or modifies trade state directly.
    - Only the live RiskManager process is authoritative; this script is advisory.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_STATE_DIR = Path("state")
_KILL_TRIGGER_PATH = _STATE_DIR / "kill_switch.trigger"
_KILL_CLEAR_SENTINEL = _STATE_DIR / "kill_switch_clear.sentinel"
_LIVE_STATE_PATH = _STATE_DIR / "live_state.json"


def _write_atomic(path: Path, content: str) -> None:
    """Write atomically: write to .tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def cmd_trigger(reason: str) -> None:
    """Trigger the kill switch by writing a trigger sentinel file."""
    payload = json.dumps({
        "reason": reason,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "operator": "manual CLI",
    }, indent=2)
    _write_atomic(_KILL_TRIGGER_PATH, payload)
    print(f"[KILL SWITCH TRIGGERED] Reason: {reason}")
    print(f"Written to: {_KILL_TRIGGER_PATH}")
    print("The live system will detect this on the next can_trade() call.")
    print("Use 'status' to confirm it was picked up.")


def cmd_clear(reason: str) -> None:
    """Clear the kill switch by writing the clear sentinel file."""
    payload = json.dumps({
        "reason": reason,
        "cleared_at": datetime.now(timezone.utc).isoformat(),
        "operator": "manual CLI",
    }, indent=2)
    _write_atomic(_KILL_CLEAR_SENTINEL, payload)
    print(f"[KILL SWITCH CLEAR REQUESTED] Reason: {reason}")
    print(f"Written to: {_KILL_CLEAR_SENTINEL}")
    print("The live system will detect this on the next can_trade() call.")
    print("IMPORTANT: Verify open positions and P&L before resuming trading.")
    print("Use 'status' to confirm it was picked up.")


def cmd_status() -> None:
    """Print current kill-switch state from the live state file."""
    if not _LIVE_STATE_PATH.exists():
        print(f"[STATUS] No live state file found at {_LIVE_STATE_PATH}")
        print("         Is the trading system running?")
        return

    try:
        data = json.loads(_LIVE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] Cannot read live state: {exc}")
        return

    ks = data.get("kill_switch_active", False)
    saved_at = data.get("saved_at", "unknown")
    session = data.get("session_date", "unknown")
    pnl = data.get("daily_pnl", "unknown")

    print("=" * 50)
    print("KILL SWITCH STATUS")
    print("=" * 50)
    print(f"  Active:       {'YES — TRADING HALTED' if ks else 'NO — trading allowed'}")
    print(f"  Session date: {session}")
    print(f"  Daily P&L:    ₹{pnl}")
    print(f"  State saved:  {saved_at}")
    print()

    if _KILL_TRIGGER_PATH.exists():
        print(f"  [!] Pending trigger file exists: {_KILL_TRIGGER_PATH}")
        print(f"      Content: {_KILL_TRIGGER_PATH.read_text(encoding='utf-8')[:200]}")
    if _KILL_CLEAR_SENTINEL.exists():
        print(f"  [!] Pending clear sentinel exists: {_KILL_CLEAR_SENTINEL}")
        print(f"      Content: {_KILL_CLEAR_SENTINEL.read_text(encoding='utf-8')[:200]}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "trigger":
        reason = " ".join(args[1:]) or "Manual operator kill switch"
        cmd_trigger(reason)
    elif cmd == "clear":
        reason = " ".join(args[1:]) or "Manual operator clear — positions reviewed"
        cmd_clear(reason)
    elif cmd == "status":
        cmd_status()
    else:
        print(f"Unknown command: {cmd!r}")
        print("Valid commands: trigger, clear, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
