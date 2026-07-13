"""
config.py — Central configuration for NSE AlgoTrader.

Single source of truth for all runtime settings. Loaded from .env at import time.
Fails fast if required fields are missing or invalid.

Constraints enforced here:
  - TRADING_MODE=live requires ENABLE_LIVE_TRADING=true + STRATEGY_ID + BROKER_APPROVAL_DATE
  - use_microstructure=true requires data_feed=truedata (ARCH-FIX-1)
  - TRAIN_BAR_TYPE == LIVE_BAR_TYPE invariant checked in model loader (not here)
  - All monetary values are Decimal — never float
"""
from __future__ import annotations

import base64
import logging
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).parent.parent / ".env"


class TradingMode(str, Enum):
    MOCK = "mock"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """
    All configuration loaded from .env. Sensitive fields are SecretStr.
    Import the module-level `settings` singleton — never instantiate directly.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Shoonya (Primary Broker) ────────────────────────────────────────────
    shoonya_user: Optional[str] = Field(None, description="Shoonya login user ID")
    shoonya_password: Optional[SecretStr] = Field(None, description="Shoonya password")
    shoonya_totp_secret: Optional[SecretStr] = Field(None, description="TOTP base32 seed for 2FA")
    shoonya_vendor_code: Optional[str] = Field(None, description="Vendor/app code from Shoonya")
    shoonya_api_secret: Optional[SecretStr] = Field(None, description="API secret from Shoonya dashboard")
    shoonya_imei: Optional[str] = Field(None, description="Device IMEI registered with Shoonya")

    # ── TrueData (Stage 2B+ only) ───────────────────────────────────────────
    truedata_user: Optional[str] = Field(None)
    truedata_password: Optional[SecretStr] = Field(None)

    # ── Trading Mode ────────────────────────────────────────────────────────
    trading_mode: TradingMode = Field(TradingMode.PAPER)
    enable_live_trading: bool = Field(False)

    # ── Data Feed (Stage 1 = shoonya, Stage 2B+ = truedata) ────────────────
    data_feed: Literal["shoonya", "truedata"] = Field("shoonya")
    use_microstructure: bool = Field(False)  # Guard: ARCH-FIX-1

    # ── Bar Type (Stage 1 = time, Stage 2A+ = volume) ──────────────────────
    bar_type: Literal["time", "volume"] = Field("time")
    bar_interval_minutes: int = Field(15, ge=1, le=60)

    # ── NSE Trading Session (IST) ───────────────────────────────────────────
    session_start_hhmm: str = Field("09:15")
    session_end_hhmm: str = Field("15:30")
    no_new_trades_after_hhmm: str = Field("15:15")  # [SURV-8]
    force_close_hhmm: str = Field("15:15")          # [SURV-8]
    skip_first_bar: bool = Field(True)               # Skip 09:15-09:30 bar [SURV-8]

    # ── Strategy / Compliance ───────────────────────────────────────────────
    strategy_id: Optional[str] = Field(None)
    broker_approval_date: Optional[str] = Field(None)

    # P0-FIX-14: Additional live-gate compliance fields.
    # All are optional in mock/paper mode; ALL are required in live mode.
    #
    # APPROVED_STATIC_IP: The outbound IP address registered with SEBI/broker.
    #   Set to your VPS/colocation egress IP. The gate checks the format only
    #   (socket.gethostbyname) — runtime IP verification is the operator's
    #   responsibility via VPN/firewall policy.
    approved_static_ip: Optional[str] = Field(
        None,
        description="Outbound IP registered with broker/SEBI (required in live mode)",
    )
    # OTR_TRACKING_ENABLED: Must be true — SEBI mandates order-to-trade ratio
    #   monitoring. The system emits otr_* structured log lines when this is set.
    otr_tracking_enabled: bool = Field(
        False,
        description="SEBI OTR (Order-to-Trade Ratio) tracking enabled",
    )
    # AUDIT_LOG_PATH: Absolute path where the structured trade audit log is written.
    #   Must be a separate, non-rotating append-only file (SEBI PFUTP §22).
    audit_log_path: Optional[str] = Field(
        None,
        description="Absolute path for the SEBI-compliant trade audit log file",
    )
    # KILL_SWITCH_TESTED: Set to true ONLY after manually verifying the kill
    #   switch triggers a full position flatten and halts new orders.
    kill_switch_tested: bool = Field(
        False,
        description="Kill switch manually tested this cycle (required before live)",
    )
    # APPROVED_MODEL_VERSION: Semver string (e.g. '1.4.2') of the ML model
    #   that was approved for live trading. Must match FrozenModelSpec.version.
    approved_model_version: Optional[str] = Field(
        None,
        description="Approved ML model version for live trading (must match FrozenModelSpec)",
    )

    # ── Capital & Risk Limits ───────────────────────────────────────────────
    capital_inr: Decimal = Field(Decimal("100000"), description="Total capital in INR")
    max_daily_loss_pct: float = Field(0.02, ge=0.005, le=0.05)   # 2% default
    max_position_pct: float = Field(0.05, ge=0.01, le=0.20)      # 5% of capital per position
    max_gross_exposure_pct: float = Field(0.20, ge=0.05, le=1.0) # 20% gross
    max_drawdown_pct: float = Field(0.10, ge=0.02, le=0.30)      # 10% from peak

    # ── Order Execution ─────────────────────────────────────────────────────
    max_orders_per_second: int = Field(8, ge=1, le=10)   # SEBI limit is 10, keep 2-order buffer
    stale_order_timeout_sec: int = Field(20, ge=5)       # Cancel LIMIT if not filled [SURV-11]
    max_price_drift_bps: int = Field(30, ge=5)           # Skip entry if price moved >30bps
    max_order_modifications: int = Field(5, ge=1)        # Per-order limit [ARCH-FIX-5]
    product_type: Literal["MIS", "CNC"] = Field("MIS")  # MIS = intraday mandatory
    exchange_segment: Literal["NSE", "BSE"] = Field("NSE")

    # ── EMA Cross Signal Parameters ─────────────────────────────────────────
    ema_fast_period: int = Field(9, ge=2, le=50)
    ema_slow_period: int = Field(21, ge=5, le=200)
    atr_period: int = Field(14, ge=5, le=50)
    atr_stop_multiplier: float = Field(1.5, ge=0.5, le=5.0)   # Stop = entry ± 1.5*ATR
    min_bars_before_signal: int = Field(21)  # Need at least ema_slow bars

    # ── ZeroMQ IPC Ports ───────────────────────────────────────────────────
    zmq_tick_pub_port: int = Field(5555)       # DataIngestor PUB → FeatureComputer SUB
    zmq_bar_pub_port: int = Field(5558)        # BarBuilder → FeatureEngine / SignalRouter
    zmq_feature_pub_port: int = Field(5556)    # FeatureComputer PUB → MLInference SUB
    zmq_ml_signal_push_port: int = Field(5557) # MLInference PUSH → OrderRouter PULL
    zmq_alert_push_port: int = Field(5560)     # Execution → AlertSink [ARCH-FIX-3]

    # ── QuestDB ─────────────────────────────────────────────────────────────
    # questdb_ilp_port: HTTP ILP endpoint (QuestDB port 9000 — same as REST API).
    # TCP ILP (port 9009) is legacy; questdb.ingress Python client uses HTTP by default.
    questdb_host: str = Field("localhost")
    questdb_ilp_port: int = Field(9000)   # HTTP ILP — matches questdb.ingress http:: scheme
    questdb_pg_port: int = Field(8812)
    questdb_pg_user: str = Field("admin")
    questdb_pg_password: SecretStr = Field("quest")  # P1-FIX-47: live gate; P1-FIX-48: SecretStr
    questdb_pg_database: str = Field("qdb")

    # ── Alerts ──────────────────────────────────────────────────────────────
    telegram_bot_token: Optional[SecretStr] = Field(None)
    telegram_chat_id: Optional[str] = Field(None)

    # ── Research / Backtest Mode ────────────────────────────────────────────
    research_mode: bool = Field(
        default=False,
        description=(
            "When True, no live broker is instantiated. Safe for research/backtesting. "
            "Automatically forces enable_live_trading=False and trading_mode=paper."
        ),
    )

    # ── System ──────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")
    log_levels: dict[str, str] = Field(
        default_factory=lambda: {"apscheduler": "WARNING", "zmq": "WARNING"},
        description="Per-module log level overrides (e.g. {\"apscheduler\": \"WARNING\"})",
    )
    symbols: str = Field("RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK")
    log_dir: str = Field("logs")

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("shoonya_totp_secret", mode="before")
    @classmethod
    def _validate_totp(cls, v: object) -> str | None:
        if v is None:
            return None
        raw = str(v).strip().upper().replace(" ", "")
        if not raw:
            return None
        try:
            base64.b32decode(raw + "=" * (-len(raw) % 8))
        except Exception as exc:
            raise ValueError(
                "SHOONYA_TOTP_SECRET is not valid base32. "
                "Re-export from your authenticator app and try again."
            ) from exc
        return raw

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _validate_mode(cls, v: object) -> str:
        v = str(v).strip().lower()
        if v not in ("mock", "paper", "live"):
            raise ValueError(f"TRADING_MODE must be mock|paper|live, got: {v!r}")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR/CRITICAL, got: {v!r}")
        return v

    @field_validator("ema_slow_period")
    @classmethod
    def _validate_ema_periods(cls, slow: int, info: object) -> int:
        # info.data has already-validated fields (Pydantic v2)
        fast = getattr(info, "data", {}).get("ema_fast_period", 9)
        if fast >= slow:
            raise ValueError(f"ema_fast_period ({fast}) must be < ema_slow_period ({slow})")
        return slow

    @model_validator(mode="after")
    def _research_mode_gate(self) -> "Settings":
        """#96: When research_mode=True, force paper trading and disable live broker.

        This is a safety invariant: research/backtesting code paths must never
        accidentally instantiate a live broker connection.  The validator runs
        after all individual field validators so it can override already-parsed
        trading_mode and enable_live_trading values.
        """
        if self.research_mode:
            object.__setattr__(self, "enable_live_trading", False)
            object.__setattr__(self, "trading_mode", TradingMode.PAPER)
            log.info(
                "config.research_mode: enable_live_trading forced False, "
                "trading_mode forced paper"
            )
        return self

    @model_validator(mode="after")
    def _broker_credentials_gate(self) -> "Settings":
        """Require broker credentials only when actually connecting to broker."""
        if self.research_mode or self.trading_mode == TradingMode.MOCK:
            return self
        missing = [
            name for name, val in [
                ("SHOONYA_USER", self.shoonya_user),
                ("SHOONYA_PASSWORD", self.shoonya_password),
                ("SHOONYA_TOTP_SECRET", self.shoonya_totp_secret),
                ("SHOONYA_VENDOR_CODE", self.shoonya_vendor_code),
                ("SHOONYA_API_SECRET", self.shoonya_api_secret),
                ("SHOONYA_IMEI", self.shoonya_imei),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Broker credentials required for {self.trading_mode.value} mode: "
                + ", ".join(missing)
                + ". Set RESEARCH_MODE=true or TRADING_MODE=mock to bypass (tests/research only)."
            )
        return self

    @model_validator(mode="after")
    def _live_gate(self) -> "Settings":
        """Hard block on live trading until all compliance gates are satisfied. [P0-FIX-14]

        Every check here corresponds to a SEBI/broker requirement.  Removing any
        check is a compliance violation — add a justification comment if a gate
        must be bypassed for testing (use TRADING_MODE=paper instead).
        """
        if self.trading_mode == TradingMode.LIVE:
            errors: list[str] = []

            # Gate 1: Explicit opt-in flag (prevents accidental live mode)
            if not self.enable_live_trading:
                errors.append("ENABLE_LIVE_TRADING must be true")

            # Gate 2: SEBI-registered strategy ID
            if not self.strategy_id:
                errors.append("STRATEGY_ID must be set (SEBI-registered strategy ID)")

            # Gate 3: Broker algo approval date
            if not self.broker_approval_date:
                errors.append("BROKER_APPROVAL_DATE must be set (YYYY-MM-DD)")

            # Gate 4 [P0-FIX-14]: Static IP registered with broker/SEBI
            if not self.approved_static_ip:
                errors.append(
                    "APPROVED_STATIC_IP must be set — the outbound IP address "
                    "registered with your broker/SEBI (typically your VPS egress IP)"
                )
            else:
                # Basic format check — not a full connectivity test
                import socket as _sock
                try:
                    _sock.inet_aton(self.approved_static_ip)
                except OSError:
                    errors.append(
                        f"APPROVED_STATIC_IP={self.approved_static_ip!r} is not a valid IPv4 address"
                    )

            # Gate 5 [P0-FIX-14]: SEBI OTR monitoring must be active
            if not self.otr_tracking_enabled:
                errors.append(
                    "OTR_TRACKING_ENABLED must be true — SEBI mandates order-to-trade "
                    "ratio monitoring for algorithmic strategies"
                )

            # Gate 6 [P0-FIX-14]: Audit log path must be configured and writable
            if not self.audit_log_path:
                errors.append(
                    "AUDIT_LOG_PATH must be set — absolute path for the SEBI-compliant "
                    "trade audit log (SEBI PFUTP §22 requires 5-year retention)"
                )
            else:
                audit_path = Path(self.audit_log_path)
                if not audit_path.is_absolute():
                    errors.append(
                        f"AUDIT_LOG_PATH={self.audit_log_path!r} must be an absolute path"
                    )
                else:
                    try:
                        audit_path.parent.mkdir(parents=True, exist_ok=True)
                        # Touch the file to verify write access
                        audit_path.touch(exist_ok=True)
                    except Exception as exc:
                        errors.append(
                            f"AUDIT_LOG_PATH={self.audit_log_path!r} is not writable: {exc}"
                        )

            # Gate 7 [P0-FIX-14]: Kill switch must have been manually tested this cycle
            if not self.kill_switch_tested:
                errors.append(
                    "KILL_SWITCH_TESTED must be true — manually verify the kill switch "
                    "flattens all positions and halts new orders before each live cycle"
                )

            # Gate 8 [P0-FIX-14]: Approved model version must be pinned
            if not self.approved_model_version:
                errors.append(
                    "APPROVED_MODEL_VERSION must be set — semver of the ML model "
                    "approved for live trading (must match FrozenModelSpec.version)"
                )

            # Gate 9 [P1-FIX-47]: Block default QuestDB password in live mode.
            # "quest" is QuestDB's factory default — using it in production
            # exposes the database to anyone on the network who knows the default.
            if self.questdb_pg_password.get_secret_value() == "quest":
                errors.append(
                    "QUESTDB_PG_PASSWORD must be changed from the default 'quest' — "
                    "set a strong, unique password in .env before live trading"
                )

            if errors:
                raise ValueError(
                    "LIVE TRADING BLOCKED — satisfy ALL compliance gates before going live:\n"
                    + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
                )
            log.warning(
                "LIVE TRADING ACTIVE — strategy=%s approved=%s model_version=%s ip=%s",
                self.strategy_id,
                self.broker_approval_date,
                self.approved_model_version,
                self.approved_static_ip,
            )
        return self

    @model_validator(mode="after")
    def _paper_gate(self) -> "Settings":
        """P2-FIX-95: Paper mode startup validation.

        Paper mode has lighter requirements than live, but still needs:
        - A plausible capital amount (otherwise position sizing is nonsensical)
        - At least one symbol to trade
        - Warning if Telegram is not configured (operator loses alerts)

        Mock mode intentionally skips all validations (used for unit tests).
        """
        if self.trading_mode not in (TradingMode.PAPER,):
            return self

        warnings_: list[str] = []

        if self.capital_inr < Decimal("10000"):
            warnings_.append(
                f"CAPITAL_INR={self.capital_inr} is very low for paper trading. "
                f"Position sizing math may produce zero-share orders. "
                f"Recommend >= 10,000 INR."
            )

        if not self.symbol_list:
            raise ValueError(
                "SYMBOLS must not be empty — provide at least one NSE symbol to trade "
                "(e.g. SYMBOLS=RELIANCE,INFY)"
            )

        if not self.telegram_bot_token:
            warnings_.append(
                "TELEGRAM_BOT_TOKEN is not set — paper trading alerts will be silent. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to receive trade alerts."
            )

        for w in warnings_:
            log.warning("config.paper_gate.WARNING: %s", w)

        return self

    @model_validator(mode="after")
    def _microstructure_gate(self) -> "Settings":
        """ARCH-FIX-1: Microstructure requires TrueData unthrottled feed."""
        if self.use_microstructure and self.data_feed != "truedata":
            raise ValueError(
                "use_microstructure=true requires data_feed=truedata. "
                "Shoonya WebSocket is throttled (1-3 snaps/sec) — "
                "Hawkes/VPIN/TFI produce ghost liquidity readings on snapped data."
            )
        if self.use_microstructure and self.bar_type == "time":
            raise ValueError(
                "use_microstructure=true is incompatible with bar_type=time. "
                "OBI/Hawkes require tick-by-tick volume bars."
            )
        return self

    # ── Convenience Properties ──────────────────────────────────────────────

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    @property
    def max_daily_loss_inr(self) -> Decimal:
        return self.capital_inr * Decimal(str(self.max_daily_loss_pct))

    @property
    def max_position_inr(self) -> Decimal:
        return self.capital_inr * Decimal(str(self.max_position_pct))

    @property
    def max_gross_exposure_inr(self) -> Decimal:
        return self.capital_inr * Decimal(str(self.max_gross_exposure_pct))

    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE and self.enable_live_trading

    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    def safe_repr(self) -> str:
        """Log-safe summary — no secret values."""
        return (
            f"Settings(mode={self.trading_mode.value!r}, "
            f"user={self.shoonya_user or 'not-set'!r}, "
            f"feed={self.data_feed!r}, "
            f"bar_type={self.bar_type!r}, "
            f"symbols={self.symbol_list}, "
            f"capital=₹{self.capital_inr:,.0f}, "
            f"live={self.enable_live_trading})"
        )

    def install_log_secret_filter(self) -> None:
        """
        Attach secret-redaction filter to root logger.
        Call once after logging.basicConfig(), before any other logging.
        """
        secrets = []
        if self.shoonya_password:
            secrets.append(self.shoonya_password.get_secret_value())
        if self.shoonya_totp_secret:
            secrets.append(self.shoonya_totp_secret.get_secret_value())
        if self.shoonya_api_secret:
            secrets.append(self.shoonya_api_secret.get_secret_value())
        secrets.append(self.questdb_pg_password.get_secret_value())  # P1-FIX-48
        if self.truedata_password:
            secrets.append(self.truedata_password.get_secret_value())
        if self.telegram_bot_token:
            secrets.append(self.telegram_bot_token.get_secret_value())

        filt = _SecretRedactFilter(secrets)
        root = logging.getLogger()
        for handler in root.handlers:
            handler.addFilter(filt)
        log.debug("Secret redaction log filter installed (%d secrets)", len(secrets))


class _SecretRedactFilter(logging.Filter):
    """Replaces literal secret values with [REDACTED] in all log records."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        # Sort longest-first to prevent partial-match shadowing
        self._secrets = sorted(
            (s for s in secrets if s and len(s) >= 8), key=len, reverse=True
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for secret in self._secrets:
            if secret in msg:
                record.msg = str(record.msg).replace(secret, "[REDACTED]")
                record.args = ()
        return True


# ── Module-level singleton ──────────────────────────────────────────────────
# Validated at import time. Any misconfiguration surfaces immediately at startup.
# All modules must: from algo_trading.config import settings
settings: Settings = Settings()  # type: ignore[call-arg]
