"""
tests/test_kill_switch_manager.py — Comprehensive test suite for KillSwitchManager.

Covers all kill levels, reset paths, audit log persistence, state replay,
alert callbacks, and the KillEvent JSON roundtrip.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from algo_trading.risk.kill_switch_manager import (
    KillEvent,
    KillLevel,
    KillReason,
    KillSwitchManager,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ksm(tmp_path: Path, alert_cb=None) -> KillSwitchManager:
    """Construct a fresh KillSwitchManager backed by a temp audit log."""
    log_path = tmp_path / "kill_switch.jsonl"
    return KillSwitchManager(audit_log_path=log_path, alert_callback=alert_cb)


# ── Test 1: Fresh state ───────────────────────────────────────────────────────

class TestFreshState:
    def test_trading_allowed_no_strategy(self, tmp_path):
        ksm = _ksm(tmp_path)
        allowed, reason = ksm.is_trading_allowed()
        assert allowed is True
        assert reason == "OK"

    def test_trading_allowed_with_strategy_id(self, tmp_path):
        ksm = _ksm(tmp_path)
        allowed, reason = ksm.is_trading_allowed("iron_condor_v1")
        assert allowed is True
        assert reason == "OK"

    def test_status_all_clear(self, tmp_path):
        ksm = _ksm(tmp_path)
        s = ksm.status()
        assert s["portfolio_killed"] is False
        assert s["portfolio_kill_reason"] is None
        assert s["human_override"] is False
        assert s["strategy_kills"] == {}

    def test_audit_log_empty_on_init(self, tmp_path):
        ksm = _ksm(tmp_path)
        assert ksm.load_audit_log() == []


# ── Test 2: STRATEGY-level kill ───────────────────────────────────────────────

class TestStrategyKill:
    def test_strategy_kill_blocks_only_that_strategy(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="strat_a")

        allowed_a, reason_a = ksm.is_trading_allowed("strat_a")
        assert allowed_a is False
        assert "strat_a" in reason_a

    def test_strategy_kill_does_not_block_other_strategies(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="strat_a")

        allowed_b, reason_b = ksm.is_trading_allowed("strat_b")
        assert allowed_b is True
        assert reason_b == "OK"

    def test_strategy_kill_does_not_block_portfolio_check(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="strat_a")

        allowed, reason = ksm.is_trading_allowed()
        assert allowed is True

    def test_strategy_kill_appears_in_status(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.DATA_QUALITY, strategy_id="strat_x")

        s = ksm.status()
        assert "strat_x" in s["strategy_kills"]
        assert s["strategy_kills"]["strat_x"] == KillReason.DATA_QUALITY.value
        assert s["portfolio_killed"] is False

    def test_multiple_strategy_kills_are_independent(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="s1")
        ksm.trigger(KillLevel.STRATEGY, KillReason.BROKER_ERROR, strategy_id="s2")

        assert ksm.is_trading_allowed("s1")[0] is False
        assert ksm.is_trading_allowed("s2")[0] is False
        assert ksm.is_trading_allowed("s3")[0] is True


# ── Test 3: PORTFOLIO-level kill ──────────────────────────────────────────────

class TestPortfolioKill:
    def test_portfolio_kill_blocks_all_strategies(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT,
                    details={"daily_loss_pct": -2.5})

        allowed, reason = ksm.is_trading_allowed("any_strategy")
        assert allowed is False
        assert "Portfolio kill" in reason or "portfolio" in reason.lower()

    def test_portfolio_kill_blocks_none_check(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.MAX_DRAWDOWN)

        allowed, _ = ksm.is_trading_allowed()
        assert allowed is False

    def test_portfolio_kill_appears_in_status(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.VIX_EMERGENCY)

        s = ksm.status()
        assert s["portfolio_killed"] is True
        assert s["portfolio_kill_reason"] == KillReason.VIX_EMERGENCY.value
        assert s["human_override"] is False


# ── Test 4: HUMAN-level kill ──────────────────────────────────────────────────

class TestHumanKill:
    def test_human_kill_sets_human_override(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL, triggered_by="ops_team")

        s = ksm.status()
        assert s["human_override"] is True
        assert s["portfolio_killed"] is True

    def test_human_kill_blocks_all_trading(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL, triggered_by="ops_team")

        for sid in (None, "strat_a", "strat_b"):
            allowed, reason = ksm.is_trading_allowed(sid)
            assert allowed is False
            assert "Human override" in reason or "human" in reason.lower()


# ── Test 5: Reset STRATEGY ────────────────────────────────────────────────────

class TestResetStrategy:
    def test_reset_strategy_re_allows_trading(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="strat_a")
        result = ksm.reset(KillLevel.STRATEGY, strategy_id="strat_a")

        assert result is True
        allowed, reason = ksm.is_trading_allowed("strat_a")
        assert allowed is True
        assert reason == "OK"

    def test_reset_strategy_does_not_affect_other_kills(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="s1")
        ksm.trigger(KillLevel.STRATEGY, KillReason.BROKER_ERROR, strategy_id="s2")
        ksm.reset(KillLevel.STRATEGY, strategy_id="s1")

        assert ksm.is_trading_allowed("s1")[0] is True
        assert ksm.is_trading_allowed("s2")[0] is False

    def test_reset_nonexistent_strategy_returns_false(self, tmp_path):
        ksm = _ksm(tmp_path)
        result = ksm.reset(KillLevel.STRATEGY, strategy_id="never_killed")
        assert result is False


# ── Test 6: Reset PORTFOLIO ───────────────────────────────────────────────────

class TestResetPortfolio:
    def test_reset_portfolio_re_allows_trading(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT)
        result = ksm.reset(KillLevel.PORTFOLIO)

        assert result is True
        allowed, reason = ksm.is_trading_allowed("any_strategy")
        assert allowed is True
        assert reason == "OK"

    def test_status_cleared_after_portfolio_reset(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.MAX_DRAWDOWN)
        ksm.reset(KillLevel.PORTFOLIO)

        s = ksm.status()
        assert s["portfolio_killed"] is False
        assert s["portfolio_kill_reason"] is None


# ── Test 7 & 8: Reset HUMAN ──────────────────────────────────────────────────

class TestResetHuman:
    def test_reset_human_without_confirm_returns_false(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL)
        result = ksm.reset(KillLevel.HUMAN, confirm_human=False)

        assert result is False

    def test_human_kill_still_active_after_failed_reset(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL)
        ksm.reset(KillLevel.HUMAN, confirm_human=False)

        allowed, _ = ksm.is_trading_allowed()
        assert allowed is False
        assert ksm.status()["human_override"] is True

    def test_reset_human_with_confirm_succeeds(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL, triggered_by="ops_team")
        result = ksm.reset(KillLevel.HUMAN, confirm_human=True, reset_by="ops_team")

        assert result is True
        allowed, reason = ksm.is_trading_allowed()
        assert allowed is True
        assert reason == "OK"

    def test_status_cleared_after_human_reset(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL)
        ksm.reset(KillLevel.HUMAN, confirm_human=True)

        s = ksm.status()
        assert s["human_override"] is False
        assert s["portfolio_killed"] is False


# ── Test 9: status() dict ─────────────────────────────────────────────────────

class TestStatus:
    def test_status_has_required_keys(self, tmp_path):
        ksm = _ksm(tmp_path)
        s = ksm.status()
        assert set(s.keys()) == {"portfolio_killed", "portfolio_kill_reason",
                                  "human_override", "strategy_kills"}

    def test_status_strategy_kills_values_are_strings(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.DATA_QUALITY, strategy_id="my_strat")
        s = ksm.status()
        for v in s["strategy_kills"].values():
            assert isinstance(v, str)


# ── Test 10: Alert callback ───────────────────────────────────────────────────

class TestAlertCallback:
    def test_alert_called_on_strategy_trigger(self, tmp_path):
        calls: list[tuple[str, str]] = []

        def spy(msg: str, level: str) -> None:
            calls.append((msg, level))

        ksm = _ksm(tmp_path, alert_cb=spy)
        ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id="strat_a")

        assert len(calls) == 1
        msg, level = calls[0]
        assert level == "CRITICAL"
        assert "strat_a" in msg or "STRATEGY" in msg

    def test_alert_called_on_portfolio_trigger(self, tmp_path):
        calls: list[tuple[str, str]] = []
        ksm = _ksm(tmp_path, alert_cb=lambda m, l: calls.append((m, l)))
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.VIX_EMERGENCY)

        assert len(calls) == 1
        assert calls[0][1] == "CRITICAL"

    def test_alert_called_on_reset(self, tmp_path):
        calls: list[tuple[str, str]] = []
        ksm = _ksm(tmp_path, alert_cb=lambda m, l: calls.append((m, l)))
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT)
        ksm.reset(KillLevel.PORTFOLIO)

        assert len(calls) == 2  # trigger + reset
        assert calls[1][1] == "WARNING"

    def test_failing_alert_callback_does_not_raise(self, tmp_path):
        def broken_cb(msg, level):
            raise RuntimeError("alert sink down")

        ksm = _ksm(tmp_path, alert_cb=broken_cb)
        # Should not propagate the exception
        ksm.trigger(KillLevel.STRATEGY, KillReason.BROKER_ERROR, strategy_id="s1")
        assert ksm.is_trading_allowed("s1")[0] is False


# ── Test 11: Audit log ────────────────────────────────────────────────────────

class TestAuditLog:
    def test_trigger_and_reset_produce_two_events(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT)
        ksm.reset(KillLevel.PORTFOLIO)

        events = ksm.load_audit_log()
        assert len(events) == 2
        assert events[0].event_type == "TRIGGERED"
        assert events[1].event_type == "RESET"

    def test_audit_event_fields_are_correct(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.STRATEGY, KillReason.DATA_QUALITY,
                    strategy_id="strat_z", triggered_by="SYSTEM",
                    details={"note": "bad tick"})

        events = ksm.load_audit_log()
        assert len(events) == 1
        ev = events[0]
        assert ev.level == KillLevel.STRATEGY
        assert ev.reason == KillReason.DATA_QUALITY
        assert ev.strategy_id == "strat_z"
        assert ev.triggered_by == "SYSTEM"
        assert ev.details == {"note": "bad tick"}
        assert ev.event_id  # non-empty
        assert ev.schema_version == 1


# ── Test 12: KillEvent JSON roundtrip ─────────────────────────────────────────

class TestKillEventRoundtrip:
    def test_to_json_and_from_json_line(self, tmp_path):
        ksm = _ksm(tmp_path)
        ksm.trigger(KillLevel.HUMAN, KillReason.MANUAL, triggered_by="ops")

        events = ksm.load_audit_log()
        original = events[0]
        roundtripped = KillEvent.from_json_line(original.to_json_line())

        assert roundtripped.event_id == original.event_id
        assert roundtripped.timestamp == original.timestamp
        assert roundtripped.event_type == original.event_type
        assert roundtripped.level == original.level
        assert roundtripped.reason == original.reason
        assert roundtripped.strategy_id == original.strategy_id
        assert roundtripped.triggered_by == original.triggered_by
        assert roundtripped.details == original.details
        assert roundtripped.schema_version == original.schema_version

    def test_roundtrip_with_strategy_id_none(self):
        from datetime import datetime
        ev = KillEvent(
            event_id="abc12345",
            timestamp=datetime(2026, 6, 19, 9, 15, 0),
            event_type="TRIGGERED",
            level=KillLevel.PORTFOLIO,
            reason=KillReason.VIX_EMERGENCY,
            strategy_id=None,
            triggered_by="SYSTEM",
            details={"vix": 35.2},
        )
        rt = KillEvent.from_json_line(ev.to_json_line())
        assert rt.strategy_id is None
        assert rt.details == {"vix": 35.2}


# ── Test 13: State persistence and replay ────────────────────────────────────

class TestStatePersistenceReplay:
    def test_reconstructed_ksm_respects_portfolio_kill(self, tmp_path):
        log_path = tmp_path / "ks.jsonl"
        ksm1 = KillSwitchManager(audit_log_path=log_path)
        ksm1.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT)

        # Reconstruct from the same log — no in-memory state shared
        ksm2 = KillSwitchManager(audit_log_path=log_path)
        allowed, reason = ksm2.is_trading_allowed("any_strat")
        assert allowed is False
        assert ksm2.status()["portfolio_killed"] is True

    def test_reconstructed_ksm_respects_strategy_kill(self, tmp_path):
        log_path = tmp_path / "ks.jsonl"
        ksm1 = KillSwitchManager(audit_log_path=log_path)
        ksm1.trigger(KillLevel.STRATEGY, KillReason.BROKER_ERROR, strategy_id="s1")

        ksm2 = KillSwitchManager(audit_log_path=log_path)
        assert ksm2.is_trading_allowed("s1")[0] is False
        assert ksm2.is_trading_allowed("s2")[0] is True

    def test_reconstructed_ksm_respects_human_override(self, tmp_path):
        log_path = tmp_path / "ks.jsonl"
        ksm1 = KillSwitchManager(audit_log_path=log_path)
        ksm1.trigger(KillLevel.HUMAN, KillReason.MANUAL, triggered_by="cto")

        ksm2 = KillSwitchManager(audit_log_path=log_path)
        assert ksm2.status()["human_override"] is True
        assert ksm2.is_trading_allowed()[0] is False

    def test_reset_persisted_and_replayed(self, tmp_path):
        log_path = tmp_path / "ks.jsonl"
        ksm1 = KillSwitchManager(audit_log_path=log_path)
        ksm1.trigger(KillLevel.PORTFOLIO, KillReason.MAX_DRAWDOWN)
        ksm1.reset(KillLevel.PORTFOLIO)

        ksm2 = KillSwitchManager(audit_log_path=log_path)
        allowed, reason = ksm2.is_trading_allowed("strat_a")
        assert allowed is True
        assert reason == "OK"


# ── Test 14: ValueError on STRATEGY trigger without strategy_id ───────────────

class TestValueError:
    def test_strategy_trigger_without_id_raises(self, tmp_path):
        ksm = _ksm(tmp_path)
        with pytest.raises(ValueError, match="strategy_id"):
            ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG)

    def test_strategy_trigger_with_none_id_raises(self, tmp_path):
        ksm = _ksm(tmp_path)
        with pytest.raises(ValueError):
            ksm.trigger(KillLevel.STRATEGY, KillReason.STUCK_LEG, strategy_id=None)


# ── Test 15: Thread safety ────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_triggers_and_checks(self, tmp_path):
        ksm = _ksm(tmp_path)
        errors: list[Exception] = []

        def trigger_loop():
            try:
                for i in range(10):
                    ksm.trigger(KillLevel.STRATEGY, KillReason.UNKNOWN,
                                strategy_id=f"strat_{i}")
            except Exception as exc:
                errors.append(exc)

        def check_loop():
            try:
                for i in range(10):
                    ksm.is_trading_allowed(f"strat_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=trigger_loop),
            threading.Thread(target=check_loop),
            threading.Thread(target=check_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread safety errors: {errors}"
