# backward compatibility shim — module moved to algo_trading.core.calendar
from algo_trading.core.calendar import *  # noqa: F401, F403
from algo_trading.core.calendar import (  # noqa: F401
    MarketCalendar, SessionState, SpecialSession,
    IST, SESSION_START, SESSION_END, NO_NEW_TRADES_AFTER, FORCE_CLOSE_TIME,
    is_trading_day, is_market_open, is_pre_open, can_enter_new_trade,
    next_trading_day, previous_trading_day, bar_start_time,
    now_ist, today_ist, get_calendar, seconds_to_force_close,
    first_trading_day_of_month, is_sip_day, is_expiry_day,
    get_expiry_risk_multiplier, check_instrument_master_freshness,
    validate_fo_prerequisites,
)
