"""
NSE News Pipeline Dashboard

Panels:
1. Sidebar       — symbol selector, time range, auto-refresh toggle
2. Top Signals   — table of current STRONG_BUY / STRONG_SELL signals with alpha scores
3. All Signals   — full signal table with color coding
4. News Feed     — recent news events, color-coded by sentiment
5. Bulk Deals    — today's institutional moves leaderboard
6. Pipeline Health — last run time, success rate per flow
7. Backtest Stats — rolling win rate, Sharpe ratio, return distribution

Auto-refresh: every 60 seconds using st.rerun() if auto_refresh is enabled.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8765"   # FastAPI service
REFRESH_INTERVAL_SEC = 60

# Signal type colors
SIGNAL_COLORS = {
    "STRONG_BUY":  "#00C853",   # bright green
    "BUY":         "#69F0AE",   # light green
    "NEUTRAL":     "#90A4AE",   # grey
    "SELL":        "#FF8A65",   # light red
    "STRONG_SELL": "#D50000",   # bright red
}

SENTIMENT_COLORS = {
    "bullish": "#00C853",
    "neutral": "#90A4AE",
    "bearish": "#D50000",
}

# Signal type ordering for consistent display
SIGNAL_ORDER = ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]

# ── Utility helpers ───────────────────────────────────────────────────────────

def _time_ago(ts_str: str | None) -> str:
    """Convert an ISO-8601 timestamp string to a human-readable 'X ago' string."""
    if not ts_str:
        return "unknown"
    try:
        # Handle both naive and tz-aware ISO strings
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta_sec = (now - ts).total_seconds()
        if delta_sec < 0:
            return "just now"
        if delta_sec < 60:
            return f"{int(delta_sec)}s ago"
        if delta_sec < 3600:
            mins = int(delta_sec // 60)
            return f"{mins} min ago"
        if delta_sec < 86400:
            hours = int(delta_sec // 3600)
            mins = int((delta_sec % 3600) // 60)
            if mins:
                return f"{hours}h {mins}m ago"
            return f"{hours}h ago"
        days = int(delta_sec // 86400)
        return f"{days}d ago"
    except Exception:
        return ts_str or "unknown"


def _alpha_color(score: float) -> str:
    """Return a hex color for an alpha score in [-1, 1]."""
    if score >= 0.60:
        return "#00C853"
    if score >= 0.30:
        return "#69F0AE"
    if score > -0.30:
        return "#90A4AE"
    if score > -0.60:
        return "#FF8A65"
    return "#D50000"


def _signal_color(signal_type: str) -> str:
    return SIGNAL_COLORS.get(signal_type, "#90A4AE")


def _fmt_cr(val: Any) -> str:
    """Format a crore value with ₹ prefix and 2 decimal places."""
    try:
        return f"₹{float(val):.2f} Cr"
    except (TypeError, ValueError):
        return "—"


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_signals(symbol: str | None = None, limit: int = 50) -> list[dict]:
    """Fetch signals from FastAPI /signals endpoint."""
    try:
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}/signals/", params=params)
            r.raise_for_status()
            return r.json().get("signals", [])
    except Exception as e:
        st.warning(f"Could not fetch signals: {e}")
        return []


@st.cache_data(ttl=30)
def fetch_news(symbol: str | None = None, hours: int = 4, limit: int = 100) -> list[dict]:
    """Fetch news from FastAPI /news endpoint."""
    try:
        params: dict[str, Any] = {"hours": hours, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}/news/", params=params)
            r.raise_for_status()
            return r.json().get("events", [])
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_bulk_deals(symbol: str | None = None) -> list[dict]:
    """Fetch bulk deals from FastAPI /bulk-deals endpoint."""
    try:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}/bulk-deals/", params=params)
            r.raise_for_status()
            return r.json().get("deals", [])
    except Exception as e:
        st.warning(f"Could not fetch bulk deals: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    """Fetch pipeline health from FastAPI /health endpoint."""
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}/health")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.warning(f"Could not fetch pipeline health: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_backtest_summary(days: int = 30) -> dict:
    """Fetch backtest summary from FastAPI /backtest/summary endpoint."""
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}/backtest/summary", params={"days": days})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.warning(f"Could not fetch backtest summary: {e}")
        return {}


# ── UI Components ──────────────────────────────────────────────────────────────

def render_signal_badge(signal_type: str) -> str:
    """Return colored HTML badge for signal type."""
    color = SIGNAL_COLORS.get(signal_type, "#90A4AE")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:12px">{signal_type}</span>'
    )


def render_top_signals_panel(signals: list[dict]) -> None:
    """
    Panel 1: Top Signals
    Show only STRONG_BUY and STRONG_SELL signals.
    Table columns: Symbol | Signal | Alpha Score | News | Bulk Deal | Delivery | VIX | Age
    Color the Alpha Score column with a gradient.
    If no strong signals: show info message.
    """
    st.subheader("Top Signals — STRONG_BUY / STRONG_SELL")

    strong = [
        s for s in signals
        if s.get("signal_type") in ("STRONG_BUY", "STRONG_SELL")
    ]

    if not strong:
        st.info("No strong signals at this time.")
        return

    rows = []
    for s in strong:
        rows.append({
            "Symbol":       s.get("symbol", "—"),
            "Signal":       s.get("signal_type", "—"),
            "Alpha Score":  round(s.get("alpha_score", 0.0), 4),
            "News":         round(s.get("news_score", 0.0), 4),
            "Bulk Deal":    round(s.get("bulk_deal_score", 0.0), 4),
            "Delivery":     round(s.get("delivery_score", 0.0), 4),
            "VIX":          round(s.get("vix_score", 0.0), 4),
            "Age":          _time_ago(s.get("generated_at")),
        })

    df = pd.DataFrame(rows)

    # Color-map alpha score column using a background gradient
    styled = df.style.background_gradient(
        subset=["Alpha Score"],
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
    ).applymap(
        lambda v: (
            f"color: white; font-weight: bold; background: {_signal_color(v)};"
            if isinstance(v, str) and v in SIGNAL_COLORS else ""
        ),
        subset=["Signal"],
    )

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "Symbol":      st.column_config.TextColumn("Symbol", width="small"),
            "Signal":      st.column_config.TextColumn("Signal", width="medium"),
            "Alpha Score": st.column_config.NumberColumn("Alpha", format="%.4f"),
            "News":        st.column_config.NumberColumn("News", format="%.4f"),
            "Bulk Deal":   st.column_config.NumberColumn("Bulk", format="%.4f"),
            "Delivery":    st.column_config.NumberColumn("Deliv.", format="%.4f"),
            "VIX":         st.column_config.NumberColumn("VIX", format="%.4f"),
            "Age":         st.column_config.TextColumn("Age", width="small"),
        },
        hide_index=True,
    )

    # Breakdown metric cards
    buys  = [s for s in strong if s.get("signal_type") == "STRONG_BUY"]
    sells = [s for s in strong if s.get("signal_type") == "STRONG_SELL"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Strong Buy Count",  len(buys))
    col2.metric("Strong Sell Count", len(sells))
    avg_alpha = sum(s.get("alpha_score", 0) for s in strong) / len(strong)
    col3.metric("Avg Alpha Score",   f"{avg_alpha:.4f}")


def render_all_signals_panel(signals: list[dict]) -> None:
    """
    Panel 2: All Signals
    Full table of all signals with color-coded signal_type column.
    Horizontal bar chart: alpha scores per symbol, colored by signal type.
    """
    st.subheader("All Signals")

    if not signals:
        st.info("No signals available.")
        return

    rows = []
    for s in signals:
        rows.append({
            "Symbol":      s.get("symbol", "—"),
            "Signal":      s.get("signal_type", "—"),
            "Alpha Score": round(s.get("alpha_score", 0.0), 4),
            "News":        round(s.get("news_score", 0.0), 4),
            "Bulk Deal":   round(s.get("bulk_deal_score", 0.0), 4),
            "Delivery":    round(s.get("delivery_score", 0.0), 4),
            "VIX":         round(s.get("vix_score", 0.0), 4),
            "Age":         _time_ago(s.get("generated_at")),
            "Events":      len(s.get("contributing_events", [])),
        })

    df = pd.DataFrame(rows)

    # Sort by alpha score descending
    df = df.sort_values("Alpha Score", ascending=False).reset_index(drop=True)

    styled = df.style.background_gradient(
        subset=["Alpha Score"],
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
    ).applymap(
        lambda v: (
            f"color: white; font-weight: bold; background: {_signal_color(v)};"
            if isinstance(v, str) and v in SIGNAL_COLORS else ""
        ),
        subset=["Signal"],
    )

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "Symbol":      st.column_config.TextColumn("Symbol", width="small"),
            "Signal":      st.column_config.TextColumn("Signal Type", width="medium"),
            "Alpha Score": st.column_config.NumberColumn("Alpha", format="%.4f"),
            "News":        st.column_config.NumberColumn("News", format="%.4f"),
            "Bulk Deal":   st.column_config.NumberColumn("Bulk", format="%.4f"),
            "Delivery":    st.column_config.NumberColumn("Deliv.", format="%.4f"),
            "VIX":         st.column_config.NumberColumn("VIX", format="%.4f"),
            "Age":         st.column_config.TextColumn("Age", width="small"),
            "Events":      st.column_config.NumberColumn("Events", format="%d"),
        },
        hide_index=True,
    )

    # Horizontal bar chart: alpha score per symbol
    st.markdown("##### Alpha Score by Symbol")
    chart_df = df[["Symbol", "Alpha Score", "Signal"]].copy()
    chart_df["Color"] = chart_df["Signal"].map(SIGNAL_COLORS).fillna("#90A4AE")

    fig = px.bar(
        chart_df,
        x="Alpha Score",
        y="Symbol",
        orientation="h",
        color="Signal",
        color_discrete_map=SIGNAL_COLORS,
        text="Alpha Score",
        title="Alpha Score per Symbol",
        height=max(300, len(chart_df) * 35),
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinewidth=2, zerolinecolor="#555"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="Signal",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_news_feed_panel(news_items: list[dict]) -> None:
    """
    Panel 3: News Feed
    Colored left-border cards per news item: headline + source + symbols + time ago.
    Max 20 items shown.
    """
    st.subheader("News Feed")

    if not news_items:
        st.info("No news events in the selected window.")
        return

    display_items = news_items[:20]
    st.caption(f"Showing {len(display_items)} of {len(news_items)} events")

    for item in display_items:
        headline    = item.get("headline", "No headline")
        source      = item.get("source", "unknown")
        sentiment   = item.get("sentiment", "neutral")
        event_type  = item.get("event_type", "OTHER")
        symbols     = item.get("symbols", [])
        published   = _time_ago(item.get("published_at"))
        sent_score  = item.get("sentiment_score", 0.0)
        url         = item.get("url", "#")

        border_color = SENTIMENT_COLORS.get(sentiment, "#90A4AE")
        badge_bg     = SENTIMENT_COLORS.get(sentiment, "#90A4AE")

        # Symbol tags
        symbol_tags = "".join(
            f'<span style="background:#37474F;color:#fff;padding:1px 6px;'
            f'border-radius:3px;font-size:11px;margin-right:4px">{sym}</span>'
            for sym in symbols[:5]
        )

        # Event type badge
        event_badge = (
            f'<span style="background:#1565C0;color:#fff;padding:1px 6px;'
            f'border-radius:3px;font-size:11px">{event_type}</span>'
        )

        # Sentiment badge
        sent_badge = (
            f'<span style="background:{badge_bg};color:#fff;padding:1px 6px;'
            f'border-radius:3px;font-size:11px">{sentiment.upper()} ({sent_score:+.2f})</span>'
        )

        card_html = f"""
<div style="
    border-left: 5px solid {border_color};
    background: rgba(255,255,255,0.04);
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 4px;
">
  <div style="font-weight:bold;font-size:14px;margin-bottom:4px">
    <a href="{url}" target="_blank" style="text-decoration:none;color:inherit">{headline}</a>
  </div>
  <div style="font-size:12px;color:#9E9E9E;margin-bottom:6px">
    {source} &bull; {published}
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center">
    {symbol_tags}
    {event_badge}
    {sent_badge}
  </div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)


def render_bulk_deals_panel(deals: list[dict]) -> None:
    """
    Panel 4: Bulk Deals Leaderboard
    Sorted by deal_value_cr descending.
    Color direction column: green=BUY, red=SELL.
    Summary metrics + pie chart: buy value vs sell value.
    """
    st.subheader("Bulk / Block Deals Leaderboard")

    if not deals:
        st.info("No bulk or block deals available for today.")
        return

    rows = []
    for d in deals:
        rows.append({
            "Symbol":     d.get("symbol", "—"),
            "Client":     d.get("client_name", "—"),
            "Type":       d.get("deal_type", "—").upper(),
            "Direction":  d.get("direction", "—"),
            "Qty":        d.get("quantity", 0),
            "Price":      round(d.get("price", 0.0), 2),
            "Value (Cr)": round(d.get("deal_value_cr", 0.0), 2),
            "Date":       (d.get("deal_date") or "")[:10],
        })

    df = pd.DataFrame(rows).sort_values("Value (Cr)", ascending=False).reset_index(drop=True)

    # Styled table
    def _dir_color(val: str) -> str:
        if val == "BUY":
            return "color: #00C853; font-weight: bold"
        if val == "SELL":
            return "color: #D50000; font-weight: bold"
        return ""

    styled = df.style.applymap(_dir_color, subset=["Direction"]).background_gradient(
        subset=["Value (Cr)"], cmap="Blues"
    )

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "Symbol":     st.column_config.TextColumn("Symbol", width="small"),
            "Client":     st.column_config.TextColumn("Client"),
            "Type":       st.column_config.TextColumn("Type", width="small"),
            "Direction":  st.column_config.TextColumn("Direction", width="small"),
            "Qty":        st.column_config.NumberColumn("Quantity", format="%d"),
            "Price":      st.column_config.NumberColumn("Price (₹)", format="%.2f"),
            "Value (Cr)": st.column_config.NumberColumn("Value (Cr)", format="%.2f"),
            "Date":       st.column_config.TextColumn("Date", width="small"),
        },
        hide_index=True,
    )

    # Summary metrics
    buy_val  = df.loc[df["Direction"] == "BUY",  "Value (Cr)"].sum()
    sell_val = df.loc[df["Direction"] == "SELL", "Value (Cr)"].sum()
    net_val  = buy_val - sell_val

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buy Value",  _fmt_cr(buy_val))
    col2.metric("Total Sell Value", _fmt_cr(sell_val))
    col3.metric("Net Flow (B-S)",   _fmt_cr(net_val), delta=f"{'BUY' if net_val >= 0 else 'SELL'} skew")

    # Pie chart
    if buy_val + sell_val > 0:
        pie_df = pd.DataFrame({
            "Direction": ["BUY", "SELL"],
            "Value (Cr)": [buy_val, sell_val],
        })
        fig = px.pie(
            pie_df,
            names="Direction",
            values="Value (Cr)",
            color="Direction",
            color_discrete_map={"BUY": "#00C853", "SELL": "#D50000"},
            title="Buy vs Sell Value (Cr)",
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_pipeline_health_panel(health: dict) -> None:
    """
    Panel 5: Pipeline Health
    Per-flow: name, last_run (time ago), status (colored dot), success_rate_pct,
    avg_latency_ms, consecutive_failures.
    Overall healthy: big green/red indicator.
    """
    st.subheader("Pipeline Health")

    if not health:
        st.info("Pipeline health data unavailable.")
        return

    # Overall status — API returns "overall_healthy" (bool) and "status" (str)
    overall_healthy = health.get("overall_healthy", False)
    overall_color   = "#00C853" if overall_healthy else "#D50000"
    overall_label   = "HEALTHY" if overall_healthy else "DEGRADED"
    overall_dot     = "🟢" if overall_healthy else "🔴"

    st.markdown(
        f'<div style="background:{overall_color}22;border:2px solid {overall_color};'
        f'border-radius:8px;padding:12px 20px;margin-bottom:16px;font-size:18px;font-weight:bold">'
        f'{overall_dot} Pipeline Status: {overall_label}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-flow details — API returns "flows" as a dict keyed by flow_name
    flows_dict: dict = health.get("flows", {})
    if not flows_dict:
        st.caption("No per-flow data returned.")
        return

    # Derived top-level metrics from flows dict
    total_flows   = len(flows_dict)
    healthy_flows = sum(1 for v in flows_dict.values() if v.get("last_status") == "completed")
    failed_flows  = sum(1 for v in flows_dict.values() if v.get("last_status") == "failed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows",   total_flows)
    col2.metric("Healthy Flows", healthy_flows)
    col3.metric("Failed Flows",  failed_flows)

    st.markdown("---")
    st.markdown("##### Flow Details")

    for flow_name, flow_data in flows_dict.items():
        status       = flow_data.get("last_status", "unknown")
        last_run_str = flow_data.get("last_run")
        success_rate = flow_data.get("success_rate_pct", None)
        avg_lat      = flow_data.get("avg_latency_ms", None)
        consec_fail  = flow_data.get("consecutive_failures", 0)

        status_lower = (status or "").lower()
        if status_lower == "completed":
            dot   = "🟢"
            color = "#00C853"
        elif status_lower == "failed":
            dot   = "🔴"
            color = "#D50000"
        elif status_lower == "running":
            dot   = "🟡"
            color = "#FFD600"
        else:
            dot   = "⚪"
            color = "#90A4AE"

        age_str = _time_ago(last_run_str) if last_run_str else "never"

        with st.expander(f"{dot} {flow_name}  —  {status.upper()}  —  last run: {age_str}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status",          status.upper())
            c2.metric(
                "Success Rate",
                f"{success_rate:.1f}%" if success_rate is not None else "—",
            )
            c3.metric(
                "Avg Latency",
                f"{avg_lat:.0f} ms" if avg_lat is not None else "—",
            )
            c4.metric(
                "Consec. Failures",
                str(consec_fail),
                delta=None if consec_fail == 0 else f"{consec_fail} fail(s)",
                delta_color="inverse",
            )


def render_backtest_panel(summary: dict) -> None:
    """
    Panel 6: Backtest Performance
    Key metrics row: win_rate, avg_return_pct, sharpe_ratio, max_drawdown
    By-signal-type table, by-symbol bar chart, recent 10 trades table.
    """
    st.subheader("Backtest Performance")

    if not summary:
        st.info("Backtest summary data unavailable.")
        return

    # ── Key metrics ──────────────────────────────────────────────────────────
    # API returns: period_days, total_evaluated, win_rate, avg_return_pct,
    #              sharpe_ratio, max_drawdown, by_signal_type, by_symbol,
    #              best_signal_type, best_symbol, recent_10
    win_rate     = summary.get("win_rate", None)
    avg_ret      = summary.get("avg_return_pct", None)
    sharpe       = summary.get("sharpe_ratio", None)
    max_dd       = summary.get("max_drawdown", None)
    total_trades = summary.get("total_evaluated", None)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Win Rate",
        f"{win_rate*100:.1f}%" if win_rate is not None else "—",
    )
    col2.metric(
        "Avg Return",
        f"{avg_ret:.2f}%" if avg_ret is not None else "—",
        delta=None,
    )
    col3.metric(
        "Sharpe Ratio",
        f"{sharpe:.3f}" if sharpe is not None else "—",
    )
    col4.metric(
        "Max Drawdown",
        f"{max_dd:.2f}%" if max_dd is not None else "—",
        delta_color="inverse",
    )
    col5.metric(
        "Total Trades",
        f"{total_trades}" if total_trades is not None else "—",
    )

    st.markdown("---")

    # ── By signal type ───────────────────────────────────────────────────────
    by_signal: dict = summary.get("by_signal_type", {})
    if by_signal:
        st.markdown("##### Performance by Signal Type")
        sig_rows = []
        for sig_type in SIGNAL_ORDER:
            data = by_signal.get(sig_type, {})
            if not data:
                continue
            sig_rows.append({
                "Signal":       sig_type,
                "Count":        data.get("count", 0),
                "Win Rate":     f"{data.get('win_rate', 0)*100:.1f}%",
                "Avg Return":   f"{data.get('avg_return_pct', 0):.2f}%",
                "Sharpe":       f"{data.get('sharpe_ratio', 0):.3f}" if data.get("sharpe_ratio") else "—",
            })

        if sig_rows:
            sig_df = pd.DataFrame(sig_rows)
            styled_sig = sig_df.style.applymap(
                lambda v: (
                    f"color: white; font-weight: bold; background: {_signal_color(v)};"
                    if isinstance(v, str) and v in SIGNAL_COLORS else ""
                ),
                subset=["Signal"],
            )
            st.dataframe(
                styled_sig,
                use_container_width=True,
                column_config={
                    "Signal":     st.column_config.TextColumn("Signal Type"),
                    "Count":      st.column_config.NumberColumn("Trades", format="%d"),
                    "Win Rate":   st.column_config.TextColumn("Win Rate"),
                    "Avg Return": st.column_config.TextColumn("Avg Return"),
                    "Sharpe":     st.column_config.TextColumn("Sharpe"),
                },
                hide_index=True,
            )

    # ── By symbol — top 10 bar chart ─────────────────────────────────────────
    # API returns by_symbol as a dict keyed by symbol name
    by_symbol_raw: dict = summary.get("by_symbol", {})
    if by_symbol_raw:
        st.markdown("##### Top 10 Symbols by Trade Count")
        # Convert dict to list of dicts for DataFrame construction
        by_symbol_list = [
            {"symbol": sym, **data}
            for sym, data in by_symbol_raw.items()
        ]
        sym_df = pd.DataFrame(by_symbol_list)
        sym_df = sym_df.sort_values("count", ascending=False).head(10)

        # win_rate as float 0-1 for color scale
        sym_df["win_rate_pct"] = sym_df["win_rate"] * 100 if "win_rate" in sym_df.columns else 0

        fig = px.bar(
            sym_df,
            x="symbol",
            y="count",
            color="win_rate_pct",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            text="count",
            labels={"count": "Trades", "win_rate_pct": "Win Rate %"},
            title="Trade Count by Symbol (color = win rate %)",
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_colorbar=dict(title="Win %"),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent 10 trades ─────────────────────────────────────────────────────
    # API returns "recent_10" (not "recent_trades")
    recent_trades: list[dict] = summary.get("recent_10", [])
    if recent_trades:
        st.markdown("##### Recent 10 Trades")
        trade_rows = []
        for t in recent_trades[:10]:
            ret = t.get("return_pct", 0.0)
            hit = t.get("hit", False)
            trade_rows.append({
                "Symbol":       t.get("symbol", "—"),
                "Signal":       t.get("signal_type", "—"),
                "Entry (₹)":   round(t.get("entry_price", 0.0), 2),
                "Exit (₹)":    round(t.get("exit_price", 0.0), 2),
                "Return (%)":  round(ret, 4),
                "Hit":         "✓" if hit else "✗",
                "Hold (h)":    round(t.get("holding_hours", 0.0), 2),
                "Evaluated":   (t.get("evaluation_date") or "")[:10],
            })

        trade_df = pd.DataFrame(trade_rows)

        def _ret_color(val: Any) -> str:
            try:
                f = float(val)
                return "color: #00C853; font-weight:bold" if f >= 0 else "color: #D50000; font-weight:bold"
            except (TypeError, ValueError):
                return ""

        def _hit_color(val: str) -> str:
            return "color: #00C853; font-weight:bold" if val == "✓" else "color: #D50000; font-weight:bold"

        styled_trades = (
            trade_df.style
            .applymap(_ret_color, subset=["Return (%)"])
            .applymap(_hit_color, subset=["Hit"])
            .applymap(
                lambda v: (
                    f"color: white; font-weight: bold; background: {_signal_color(v)};"
                    if isinstance(v, str) and v in SIGNAL_COLORS else ""
                ),
                subset=["Signal"],
            )
        )

        st.dataframe(
            styled_trades,
            use_container_width=True,
            column_config={
                "Symbol":      st.column_config.TextColumn("Symbol", width="small"),
                "Signal":      st.column_config.TextColumn("Signal", width="medium"),
                "Entry (₹)":  st.column_config.NumberColumn("Entry ₹", format="%.2f"),
                "Exit (₹)":   st.column_config.NumberColumn("Exit ₹", format="%.2f"),
                "Return (%)": st.column_config.NumberColumn("Return %", format="%.4f"),
                "Hit":         st.column_config.TextColumn("Hit", width="small"),
                "Hold (h)":    st.column_config.NumberColumn("Hold h", format="%.2f"),
                "Evaluated":   st.column_config.TextColumn("Evaluated", width="small"),
            },
            hide_index=True,
        )


# ── Main App ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NSE News Pipeline",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS — dark borders, subtle card backgrounds
    st.markdown(
        """
        <style>
        .stMetric { background: rgba(255,255,255,0.04); border-radius: 6px; padding: 8px; }
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 NSE News Pipeline")
        st.divider()

        # Symbol selector (multi-select from WATCHED_SYMBOLS + "All")
        import os as _os
        _env_syms = _os.getenv("WATCHED_SYMBOLS", "")
        watched_env = [s.strip() for s in _env_syms.split(",") if s.strip()]
        _defaults = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "WIPRO", "AXISBANK", "KOTAKBANK",
            "NIFTY50", "BANKNIFTY",
        ]
        watched = ["All"] + (watched_env if watched_env else _defaults)

        selected_symbol = st.selectbox("Symbol", watched)
        if selected_symbol == "All":
            selected_symbol = None

        hours = st.slider("News window (hours)", 1, 24, 4)

        backtest_days = st.slider("Backtest window (days)", 7, 90, 30)

        st.divider()
        auto_refresh = st.toggle("Auto-refresh (60s)", value=True)

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"API: {API_BASE}")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # ── Main Content ──────────────────────────────────────────────────────────

    # Fetch all data (cached independently)
    signals     = fetch_signals(symbol=selected_symbol)
    news_items  = fetch_news(symbol=selected_symbol, hours=hours)
    deals       = fetch_bulk_deals(symbol=selected_symbol)
    health      = fetch_health()
    backtest    = fetch_backtest_summary(days=backtest_days)

    # Header metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    strong_buys  = sum(1 for s in signals if s.get("signal_type") == "STRONG_BUY")
    strong_sells = sum(1 for s in signals if s.get("signal_type") == "STRONG_SELL")
    all_signals  = len(signals)
    win_rate_hdr = backtest.get("win_rate", 0) or 0
    pipeline_ok  = health.get("overall_healthy", None)

    col1.metric("Strong Buy",       strong_buys)
    col2.metric("Strong Sell",      strong_sells)
    col3.metric("Total Signals",    all_signals)
    col4.metric("News Events",      len(news_items))
    col5.metric(
        "Win Rate (backtest)",
        f"{win_rate_hdr * 100:.1f}%",
        delta="pipeline OK" if pipeline_ok is True else ("pipeline DEGRADED" if pipeline_ok is False else None),
        delta_color="normal" if pipeline_ok is True else "inverse",
    )

    st.divider()

    # Tabbed layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Top Signals",
        "📊 All Signals",
        "📰 News Feed",
        "💰 Bulk Deals",
        "🔧 Pipeline Health",
        "📈 Backtest",
    ])

    with tab1:
        render_top_signals_panel(signals)

    with tab2:
        render_all_signals_panel(signals)

    with tab3:
        render_news_feed_panel(news_items)

    with tab4:
        render_bulk_deals_panel(deals)

    with tab5:
        render_pipeline_health_panel(health)

    with tab6:
        render_backtest_panel(backtest)

    # Auto-refresh — sleep then rerun (clears cache first so fresh data is fetched)
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL_SEC)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
