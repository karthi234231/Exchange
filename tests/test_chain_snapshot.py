"""
tests/test_chain_snapshot.py — Unit tests for market_data/chain_snapshot.py

Verifies:
  - OptionQuote.is_valid(): correct validation rules
  - OptionQuote.mid_price(): bid/ask midpoint and LTP fallback
  - ChainSnapshot.get_quote(): O(1) lookup
  - ChainSnapshot.strikes_by_expiry(): sorted, filtered
  - ChainSnapshot.all_expiries(): sorted distinct
  - ChainSnapshot.atm_quote(): nearest strike to underlying
  - ChainSnapshot.completeness_pct(): % of expected strikes present
  - ChainSnapshot.iv_smile(): full smile with NaN for missing IVs
"""
from __future__ import annotations

import math
from datetime import date, datetime

import pytest

from algo_trading.market_data.chain_snapshot import (
    OptionQuote,
    ChainSnapshot,
    CURRENT_SCHEMA_VERSION,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_quote(
    symbol: str = "NIFTY",
    expiry: date = date(2026, 6, 10),
    strike: float = 24000.0,
    option_type: str = "CE",
    bid: float = 99.0,
    ask: float = 101.0,
    ltp: float = 100.0,
    iv: float = 0.15,
    oi: int = 5000,
    volume: int = 1000,
    delta: float | None = 0.5,
    quote_ts: datetime | None = None,
) -> OptionQuote:
    if quote_ts is None:
        quote_ts = datetime(2026, 6, 10, 10, 30)
    return OptionQuote(
        symbol=symbol, expiry=expiry, strike=strike,
        option_type=option_type, bid=bid, ask=ask, ltp=ltp,
        iv=iv, oi=oi, volume=volume,
        delta=delta, gamma=None, vega=None, theta=None,
        quote_timestamp=quote_ts,
    )


def make_snapshot(
    underlying: float = 24000.0,
    quotes: dict | None = None,
) -> ChainSnapshot:
    if quotes is None:
        quotes = {}
    return ChainSnapshot(
        symbol="NIFTY",
        snapshot_timestamp=datetime(2026, 6, 10, 10, 30),
        event_timestamp=datetime(2026, 6, 10, 10, 30),
        underlying_price=underlying,
        underlying_bid=None,
        underlying_ask=None,
        schema_version=CURRENT_SCHEMA_VERSION,
        sequence_number=1,
        data_quality="COMPLETE",
        quotes=quotes,
    )


EXP1 = date(2026, 6, 10)
EXP2 = date(2026, 6, 17)

STRIKES = [23800.0, 23900.0, 24000.0, 24100.0, 24200.0]


def make_populated_snapshot(underlying: float = 24000.0) -> ChainSnapshot:
    """Snapshot with 5 strikes × 2 option_types × 2 expiries = 20 quotes."""
    quotes = {}
    for exp in [EXP1, EXP2]:
        for strike in STRIKES:
            for ot in ("CE", "PE"):
                iv = 0.15 + (abs(strike - underlying) / underlying) * 0.5  # smile
                q = make_quote(
                    expiry=exp, strike=strike, option_type=ot,
                    bid=max(0.5, 100.0 - abs(strike - underlying) * 0.01),
                    ask=max(1.0, 101.0 - abs(strike - underlying) * 0.01),
                    iv=iv,
                )
                quotes[(exp, strike, ot)] = q
    return make_snapshot(underlying=underlying, quotes=quotes)


# ── OptionQuote.is_valid ──────────────────────────────────────────────────────

class TestOptionQuoteIsValid:
    def test_valid_quote(self):
        q = make_quote()
        assert q.is_valid() is True

    def test_negative_bid_invalid(self):
        q = make_quote(bid=-1.0)
        assert q.is_valid() is False

    def test_negative_ask_invalid(self):
        q = make_quote(ask=-1.0)
        assert q.is_valid() is False

    def test_negative_ltp_invalid(self):
        q = make_quote(ltp=-5.0)
        assert q.is_valid() is False

    def test_crossed_market_invalid(self):
        """bid > ask (crossed market) → invalid."""
        q = make_quote(bid=105.0, ask=100.0)
        assert q.is_valid() is False

    def test_locked_market_ok(self):
        """bid == ask (locked market) is valid."""
        q = make_quote(bid=100.0, ask=100.0)
        assert q.is_valid() is True

    def test_zero_bid_ask_ok(self):
        """bid=0 and ask=0 means no market depth — but valid."""
        q = make_quote(bid=0.0, ask=0.0)
        assert q.is_valid() is True

    def test_negative_iv_invalid(self):
        """IV < 0 (and not the sentinel -1.0) → invalid."""
        q = make_quote(iv=-0.05)
        assert q.is_valid() is False

    def test_sentinel_iv_valid(self):
        """IV = -1.0 is the sentinel for 'unavailable' → valid."""
        q = make_quote(iv=-1.0)
        assert q.is_valid() is True

    def test_invalid_option_type(self):
        q = make_quote(option_type="XX")
        assert q.is_valid() is False

    def test_valid_pe(self):
        q = make_quote(option_type="PE")
        assert q.is_valid() is True


# ── OptionQuote.mid_price ─────────────────────────────────────────────────────

class TestOptionQuoteMidPrice:
    def test_normal_mid(self):
        q = make_quote(bid=99.0, ask=101.0)
        assert q.mid_price() == pytest.approx(100.0)

    def test_zero_bid_ask_falls_back_to_ltp(self):
        q = make_quote(bid=0.0, ask=0.0, ltp=95.0)
        assert q.mid_price() == pytest.approx(95.0)

    def test_wide_spread_mid(self):
        q = make_quote(bid=90.0, ask=110.0)
        assert q.mid_price() == pytest.approx(100.0)


# ── ChainSnapshot.get_quote ───────────────────────────────────────────────────

class TestGetQuote:
    def test_found(self):
        snap = make_populated_snapshot()
        q = snap.get_quote(EXP1, 24000.0, "CE")
        assert q is not None
        assert q.strike == 24000.0
        assert q.option_type == "CE"

    def test_missing_strike_returns_none(self):
        snap = make_populated_snapshot()
        q = snap.get_quote(EXP1, 99999.0, "CE")
        assert q is None

    def test_missing_expiry_returns_none(self):
        snap = make_populated_snapshot()
        q = snap.get_quote(date(2099, 1, 1), 24000.0, "CE")
        assert q is None


# ── ChainSnapshot.strikes_by_expiry ──────────────────────────────────────────

class TestStrikesByExpiry:
    def test_correct_strikes(self):
        snap = make_populated_snapshot()
        strikes = snap.strikes_by_expiry(EXP1)
        assert set(strikes) == set(STRIKES)

    def test_sorted(self):
        snap = make_populated_snapshot()
        strikes = snap.strikes_by_expiry(EXP1)
        assert strikes == sorted(strikes)

    def test_empty_for_unknown_expiry(self):
        snap = make_populated_snapshot()
        assert snap.strikes_by_expiry(date(2099, 1, 1)) == []


# ── ChainSnapshot.all_expiries ────────────────────────────────────────────────

class TestAllExpiries:
    def test_returns_both_expiries(self):
        snap = make_populated_snapshot()
        exps = snap.all_expiries()
        assert EXP1 in exps
        assert EXP2 in exps

    def test_sorted_ascending(self):
        snap = make_populated_snapshot()
        exps = snap.all_expiries()
        assert exps == sorted(exps)

    def test_no_duplicates(self):
        snap = make_populated_snapshot()
        exps = snap.all_expiries()
        assert len(exps) == len(set(exps))


# ── ChainSnapshot.atm_quote ───────────────────────────────────────────────────

class TestAtmQuote:
    def test_atm_strike_at_24000(self):
        """Underlying=24000; ATM=24000 (exact match)."""
        snap = make_populated_snapshot(underlying=24000.0)
        q = snap.atm_quote(EXP1, "CE")
        assert q is not None
        assert q.strike == 24000.0

    def test_atm_picks_nearest_strike(self):
        """Underlying=24050; nearest strike from [23800..24200] step 100 = 24000."""
        snap = make_populated_snapshot(underlying=24050.0)
        q = snap.atm_quote(EXP1, "CE")
        # Nearest to 24050 among [23800, 23900, 24000, 24100, 24200]
        assert q is not None
        assert q.strike in (24000.0, 24100.0)

    def test_missing_expiry_returns_none(self):
        snap = make_populated_snapshot()
        q = snap.atm_quote(date(2099, 1, 1), "CE")
        assert q is None

    def test_pe_atm(self):
        snap = make_populated_snapshot(underlying=24000.0)
        q = snap.atm_quote(EXP1, "PE")
        assert q is not None
        assert q.option_type == "PE"


# ── ChainSnapshot.completeness_pct ───────────────────────────────────────────

class TestCompletenessPct:
    def test_all_present_100_pct(self):
        snap = make_populated_snapshot()
        pct = snap.completeness_pct(expected_strikes=STRIKES, expiry=EXP1)
        assert pct == 100.0

    def test_one_extra_strike_reduces_pct(self):
        """Adding an extra expected strike not in the snapshot reduces %."""
        snap = make_populated_snapshot()
        extra = STRIKES + [99999.0]
        pct = snap.completeness_pct(expected_strikes=extra, expiry=EXP1)
        expected = len(STRIKES) / len(extra) * 100.0
        assert abs(pct - expected) < 0.01

    def test_empty_expected_returns_100(self):
        snap = make_populated_snapshot()
        pct = snap.completeness_pct(expected_strikes=[], expiry=EXP1)
        assert pct == 100.0

    def test_no_strikes_in_snapshot_zero_pct(self):
        snap = make_snapshot()  # empty snapshot
        pct = snap.completeness_pct(expected_strikes=STRIKES, expiry=EXP1)
        assert pct == 0.0


# ── ChainSnapshot.iv_smile ────────────────────────────────────────────────────

class TestIVSmile:
    def test_returns_dict(self):
        snap = make_populated_snapshot()
        smile = snap.iv_smile(EXP1)
        assert isinstance(smile, dict)

    def test_keys_are_strikes(self):
        snap = make_populated_snapshot()
        smile = snap.iv_smile(EXP1)
        for k in smile:
            assert k in STRIKES

    def test_values_are_tuples(self):
        snap = make_populated_snapshot()
        smile = snap.iv_smile(EXP1)
        for k, v in smile.items():
            assert isinstance(v, tuple) and len(v) == 2

    def test_iv_values_positive(self):
        snap = make_populated_snapshot()
        smile = snap.iv_smile(EXP1)
        for strike, (call_iv, put_iv) in smile.items():
            # iv should be positive (or NaN if unavailable)
            if not math.isnan(call_iv):
                assert call_iv > 0
            if not math.isnan(put_iv):
                assert put_iv > 0

    def test_missing_iv_is_nan(self):
        """Quote with iv=-1.0 (unavailable sentinel) → NaN in smile."""
        quotes = {
            (EXP1, 24000.0, "CE"): make_quote(strike=24000.0, option_type="CE", iv=-1.0),
            (EXP1, 24000.0, "PE"): make_quote(strike=24000.0, option_type="PE", iv=0.18),
        }
        snap = make_snapshot(quotes=quotes)
        smile = snap.iv_smile(EXP1)
        call_iv, put_iv = smile[24000.0]
        assert math.isnan(call_iv), "Unavailable CE IV should be NaN"
        assert not math.isnan(put_iv), "Available PE IV should not be NaN"

    def test_unknown_expiry_empty_dict(self):
        snap = make_populated_snapshot()
        smile = snap.iv_smile(date(2099, 1, 1))
        assert smile == {}
