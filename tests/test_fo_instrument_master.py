"""
tests/test_fo_instrument_master.py — Unit tests for instrument_master sub-package.

Covers:
  - LotSizeHistory: date-aware lot size resolution for NIFTY and BANKNIFTY
  - LotSizeNotFoundError: raised for unknown symbols / dates before first epoch
  - get_lot_size(): convenience wrapper
  - generate_strikes(): correct count, alignment, sorted order
  - atm_strike(): nearest valid strike
  - get_moneyness_category(): correct classification
  - otm_strikes(): correct OTM enumeration for CE and PE

Reference lot sizes (NSE circulars FAOP67267, FAOP70616):
  NIFTY    : 75 until 2025-12-31; 65 from 2026-01-01
  BANKNIFTY: 35 until 2025-12-31; 30 from 2026-01-01
"""
from __future__ import annotations

import pytest
from datetime import date

from algo_trading.fo.instrument_master import (
    LotSizeHistory,
    LotSizeNotFoundError,
    get_lot_size,
    generate_strikes,
    atm_strike,
    get_moneyness_category,
    otm_strikes,
)


# ── LotSizeHistory ────────────────────────────────────────────────────────────

class TestLotSizeHistory:
    """Verify date-aware lot size resolution."""

    def test_nifty_pre_2026(self):
        """NIFTY lot size was 75 in 2025 (FAOP67267)."""
        assert get_lot_size("NIFTY", date(2025, 6, 1)) == 75

    def test_nifty_2026(self):
        """NIFTY lot size is 65 from 2026-01-01 (FAOP70616)."""
        assert get_lot_size("NIFTY", date(2026, 1, 1)) == 65

    def test_nifty_today(self):
        """NIFTY lot size on 2026-06-19 (today) is 65."""
        assert get_lot_size("NIFTY", date(2026, 6, 19)) == 65

    def test_nifty_last_day_of_old_epoch(self):
        """NIFTY lot size on 2025-12-31 is still 75 (epoch boundary)."""
        assert get_lot_size("NIFTY", date(2025, 12, 31)) == 75

    def test_banknifty_pre_2026(self):
        """BANKNIFTY lot size was 35 in 2025 (FAOP70616 preamble)."""
        assert get_lot_size("BANKNIFTY", date(2025, 6, 1)) == 35

    def test_banknifty_2026(self):
        """BANKNIFTY lot size is 30 from 2026-01-01 (FAOP70616)."""
        assert get_lot_size("BANKNIFTY", date(2026, 1, 1)) == 30

    def test_case_insensitive(self):
        """Symbol lookup must be case-insensitive."""
        assert get_lot_size("nifty", date(2026, 1, 15)) == 65
        assert get_lot_size("Nifty", date(2026, 1, 15)) == 65

    def test_unknown_symbol_raises(self):
        """Unknown symbol raises LotSizeNotFoundError."""
        with pytest.raises(LotSizeNotFoundError):
            get_lot_size("FINNIFTY", date(2026, 1, 1))

    def test_lot_size_positive(self):
        """Lot size must be a positive integer."""
        lot = get_lot_size("NIFTY", date(2026, 3, 15))
        assert isinstance(lot, int)
        assert lot > 0


# ── generate_strikes ──────────────────────────────────────────────────────────

class TestGenerateStrikes:
    """Verify strike generation around a given ATM price."""

    def test_nifty_strike_interval_50(self):
        """All Nifty strikes must be multiples of 50."""
        strikes = generate_strikes(
            underlying_price=24000.0,
            strike_interval=50.0,
            n_strikes_each_side=5,
        )
        for s in strikes:
            assert s % 50 == 0, f"Strike {s} is not a multiple of 50"

    def test_count_correct(self):
        """n_strikes_each_side=5 gives 11 strikes (ATM ± 5)."""
        strikes = generate_strikes(24000.0, 50.0, n_strikes_each_side=5)
        assert len(strikes) == 11

    def test_sorted_ascending(self):
        strikes = generate_strikes(24000.0, 50.0, n_strikes_each_side=5)
        assert strikes == sorted(strikes)

    def test_contains_atm(self):
        """ATM strike must be in the generated list."""
        strikes = generate_strikes(24000.0, 50.0, n_strikes_each_side=5)
        atm = atm_strike(24000.0, 50.0)
        assert atm in strikes

    def test_banknifty_interval_100(self):
        """BankNifty strikes in multiples of 100."""
        strikes = generate_strikes(52000.0, 100.0, n_strikes_each_side=5)
        for s in strikes:
            assert s % 100 == 0

    def test_all_positive(self):
        strikes = generate_strikes(24000.0, 50.0, n_strikes_each_side=10)
        assert all(s > 0 for s in strikes)


# ── atm_strike ────────────────────────────────────────────────────────────────

class TestAtmStrike:
    """Verify ATM strike rounding."""

    def test_exact_atm(self):
        """Price exactly on a strike → returns that strike."""
        assert atm_strike(24000.0, 50.0) == 24000.0

    def test_rounds_down(self):
        """24010 → nearest 50-interval strike = 24000."""
        assert atm_strike(24010.0, 50.0) == 24000.0

    def test_rounds_to_nearest(self):
        """24025 → 24000 (lower) or 24050 (upper) — both acceptable if consistent."""
        result = atm_strike(24025.0, 50.0)
        # Result must be one of the two nearest strikes
        assert result in (24000.0, 24050.0)

    def test_rounds_up(self):
        """24040 → nearest is 24050."""
        assert atm_strike(24040.0, 50.0) == 24050.0

    def test_banknifty_rounding(self):
        """52050 → 52100 for BankNifty (interval=100)."""
        assert atm_strike(52050.0, 100.0) in (52000.0, 52100.0)


# ── get_moneyness_category ────────────────────────────────────────────────────

class TestMoneynessCategory:
    """
    Verify moneyness classification for a CE option (call).

    ATM = 24000, interval = 50
    DEEP_ITM : |distance| > 5 intervals = > 250 pts
    ITM      : 1–5 intervals = 50–250 pts ITM side
    ATM      : within 0.5 intervals = within 25 pts
    OTM      : 1–5 intervals OTM
    DEEP_OTM : > 5 intervals OTM
    """

    def test_call_atm(self):
        """Strike == underlying → ATM."""
        cat = get_moneyness_category(
            underlying=24000.0, strike=24000.0,
            option_type="CE", strike_interval=50.0,
        )
        assert cat == "ATM"

    def test_call_itm(self):
        """Strike < underlying by 2 intervals → ITM."""
        cat = get_moneyness_category(24000.0, 23900.0, "CE", 50.0)
        assert cat == "ITM"

    def test_call_otm(self):
        """Strike > underlying by 2 intervals → OTM."""
        cat = get_moneyness_category(24000.0, 24100.0, "CE", 50.0)
        assert cat == "OTM"

    def test_call_deep_itm(self):
        """Strike 300+ below underlying → DEEP_ITM."""
        cat = get_moneyness_category(24000.0, 23600.0, "CE", 50.0)
        assert cat == "DEEP_ITM"

    def test_call_deep_otm(self):
        """Strike 300+ above underlying → DEEP_OTM."""
        cat = get_moneyness_category(24000.0, 24350.0, "CE", 50.0)
        assert cat == "DEEP_OTM"

    def test_put_itm(self):
        """Put: strike > underlying → ITM."""
        cat = get_moneyness_category(24000.0, 24100.0, "PE", 50.0)
        assert cat == "ITM"

    def test_put_otm(self):
        """Put: strike < underlying → OTM."""
        cat = get_moneyness_category(24000.0, 23900.0, "PE", 50.0)
        assert cat == "OTM"


# ── otm_strikes ───────────────────────────────────────────────────────────────

class TestOtmStrikes:
    """Verify OTM strike enumeration for CE and PE."""

    def test_call_otm_strikes_above_atm(self):
        """CE OTM strikes must all be above ATM."""
        atm = 24000.0
        strikes = otm_strikes(atm, 50.0, n=5, option_type="CE")
        assert all(s > atm for s in strikes), "CE OTM strikes must be > ATM"

    def test_put_otm_strikes_below_atm(self):
        """PE OTM strikes must all be below ATM."""
        atm = 24000.0
        strikes = otm_strikes(atm, 50.0, n=5, option_type="PE")
        assert all(s < atm for s in strikes), "PE OTM strikes must be < ATM"

    def test_count(self):
        """Returns exactly n strikes."""
        strikes = otm_strikes(24000.0, 50.0, n=3, option_type="CE")
        assert len(strikes) == 3

    def test_call_sorted_ascending(self):
        strikes = otm_strikes(24000.0, 50.0, n=5, option_type="CE")
        assert strikes == sorted(strikes)

    def test_put_sorted_descending_or_ascending(self):
        """PE OTM strikes should be sorted (either order is acceptable)."""
        strikes = otm_strikes(24000.0, 50.0, n=5, option_type="PE")
        assert strikes == sorted(strikes) or strikes == sorted(strikes, reverse=True)

    def test_all_multiples_of_interval(self):
        """All OTM strikes must be multiples of the interval."""
        for opt in ("CE", "PE"):
            strikes = otm_strikes(24000.0, 50.0, n=5, option_type=opt)
            for s in strikes:
                assert s % 50 == 0, f"OTM strike {s} not a multiple of 50"
