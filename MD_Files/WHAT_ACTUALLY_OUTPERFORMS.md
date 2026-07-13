# What Actually Outperforms in Algorithmic Trading
## Exhaustive Research Synthesis — Live Evidence vs. Hype (June 2026)

> **Target system context:** NSE India, XGBoost/LightGBM/CatBoost ensemble, 15-min bars, 50 Nifty stocks, Shoonya broker.
> **Compiled from:** 7 parallel research agents, 300+ sources, arXiv/SSRN/NeurIPS/ICML/ICAIF/Journal of Finance/academic databases.

---

## CRITICAL FRAMING: The Backtest-to-Live Gap

Before anything else — the single most important empirical finding across all research:

| Metric | Number | Source |
|---|---|---|
| Correlation of backtest Sharpe to live Sharpe | R² < 0.025 (statistically zero) | 888 Quantopian strategies |
| Average backtest-to-live Sharpe degradation | 33–50% decline | QuantPedia, 355 strategies |
| Academic predictors out-of-sample decay | 26% lower returns | McLean & Pontiff (2016, JF) |
| ML experiments that survive with real capital | <10% | PMC review, 2022 |
| DSR-adjusted probability of genuinely outperforming | 2–5% for top backtesters | Bailey & López de Prado |
| Retail F&O traders who lost money FY22–FY24 | 93% | SEBI official study |

**Industry rule:** Discount every backtest Sharpe by at least 50% before trusting it. A Sharpe of 2.0 in backtest = expected live Sharpe of ~0.7–1.0.

---

## TABLE OF CONTENTS

1. [India/NSE-Specific Alpha — Free Data Sources](#1-indianse-specific-alpha--free-data-sources)
2. [ML Architecture: What Beats GBDT (and What Doesn't)](#2-ml-architecture-what-beats-gbdt-and-what-doesnt)
3. [Alternative Data: Live Alpha Evidence](#3-alternative-data-live-alpha-evidence)
4. [Reinforcement Learning: Honest Live Assessment](#4-reinforcement-learning-honest-live-assessment)
5. [LLM/NLP/Sentiment: What Has Shown Real Alpha](#5-llmnlpsentiment-what-has-shown-real-alpha)
6. [Execution Technology: Documented Improvements](#6-execution-technology-documented-improvements)
7. [Factor Investing: What Survived Post-2020](#7-factor-investing-what-survived-post-2020)
8. [The Failures: Documented Post-Mortems](#8-the-failures-documented-post-mortems)
9. [Actionable Priority List for Your NSE System](#9-actionable-priority-list-for-your-nse-system)
10. [Key Sources](#10-key-sources)

---

## 1. India/NSE-Specific Alpha — Free Data Sources

### 1.1 Post-Earnings Announcement Drift (PEAD) — Strongest Documented India Signal

**Evidence tier: Tier 1 (peer-reviewed, NSE-specific)**

- **Source:** "Post-Earnings-Announcement Drift Anomaly in India" (SCIRP, 2018)
- **Dataset:** Nifty 500, 2002 Q4 to 2017 Q3 (15 years, quarterly)
- **Finding:** Statistically significant PEAD in 35 out of 37 quarters
- **Magnitude:** Hedge portfolio return (long top surprise decile, short bottom decile): **~10% per quarter**
- **Drift duration:** Up to **64 trading days** post-announcement
- **Persistence:** Survives control for beta, market cap, P/B, illiquidity, idiosyncratic volatility

**Signal construction:**
```
SUE = (Actual EPS - Same Quarter Prior Year EPS) / Stock Price
Sort NSE 500 by SUE quarterly
Go long top decile for 40–60 days post-announcement
Data: NSE corporate filings, Screener.in, Trendlyne
```

**Why it persists in India:** Analyst coverage is sparse outside Nifty 100. Price discovery for mid/small-cap earnings is slow. Institutional processing lag is documented.

---

### 1.2 Momentum Factor — Strongest Factor in India Historically

**Evidence tier: Tier 1 (IIM Ahmedabad four-factor model, peer-reviewed)**

- **IIM Ahmedabad finding:** Momentum factor annual return **21.9%** (vs 15.3% for value, 11.5% for market premium) — strongest factor in India
- **NSE official index:** Nifty500 Momentum 50 CAGR 2005–2025: **21.73%** vs Nifty 500 ~13%
- **3-year Sharpe:** 0.75

**CRITICAL WARNING — the liquidity trap:**
- BacktestIndia 18-year study: All momentum alpha in India is concentrated in **illiquid stocks**
  - Low-turnover momentum: **19.43% CAGR**
  - High-turnover (liquid) momentum: **8.51% CAGR** — below Nifty 50 benchmark
- **Implication:** Scaling kills the alpha. For a Nifty 50 strategy, classical momentum has no edge over the index.

**Signal construction:**
```python
# Screen Nifty 500 (not just Nifty 50) for actual alpha
# Rank by: (6M return + 12M return) / 2, volatility-adjusted
# Skip last 1 month (short-term reversal effect)
# Rebalance semi-annually (June/December aligns with Nifty500M50 schedule)
# Cap position size at 5% per stock — circuit breaker risk management
```

**Lalwani (SSRN 2024) caveat:** Factor index performance falls considerably post-launch date. Live momentum alpha is likely 30–40% lower than 18-year backtests suggest.

---

### 1.3 Quality + Momentum Combined — Best Risk-Adjusted Strategy (NSE, 18 years)

**Evidence tier: Tier 2 (BacktestIndia, 18-year NSE backtest, 2006–2025)**

- **CAGR: 17.95%** vs Nifty 50 ~10.42% (83% more terminal wealth)
- Low volatility factor alone: **12.38–15.2% CAGR**, -44% max DD vs Nifty's -55%
- Multi-factor (Low Vol + Momentum): 14.61% CAGR, Sharpe 0.48

**Why quality is resilient:** Unlike momentum, quality (low debt, high ROIC, stable earnings) admits multiple interpretations and cannot be uniformly crowded like mechanical momentum signals. Quality is the **most decay-resistant factor** in both India and global markets.

---

### 1.4 Expiry-Day Options Strategy — F&O-Specific

**Evidence tier: Tier 2 (AlgoTest.in NSE historical options data)**

- **Strategy:** Short ATM straddle at 9:20 AM on expiry day, close by 3:15 PM
- **Bank Nifty 5-year backtest:** 593% cumulative return, Sharpe ~1.96
- **Mechanism:** Theta decay is highest on expiry day. ATM options approaching zero time value create predictable premium collapse in the absence of large directional moves.

**Critical caveats:**
- Large gap days (RBI surprise, global event) can produce unbounded losses on short straddle
- Weekly expiry changed (Nifty now expires Tuesday post-September 2025, BankNifty expires Wednesday)
- Backtested with historical data — live execution adds slippage on options spreads (0.5–2% of premium)
- **Never hold naked short straddle past 2:30 PM** — gamma explosion risk in final 90 minutes

**India-specific dynamic:** Max pain gravitational pull is real in final 2–5 sessions before monthly expiry (more OI concentration than weekly). Monthly contracts show stronger pin-to-strike effect than weekly.

---

### 1.5 Promoter Pledging — Free Negative Signal

**Evidence tier: Tier 2 (multiple peer-reviewed Indian journals)**

- Promoter pledge >25% → negative signal for future returns and crash risk
- High pledge predicts lower financial performance, deteriorating earnings quality
- **Live episode (COVID March 2020):** NSE 500 stocks with high promoter pledge fell 37% **more** than low-pledge stocks in the same sector during the crash week
- Pledge declining quarter-over-quarter → potential positive reversal signal

**Data:** BSE/NSE quarterly shareholding filings (free), aggregated on Screener.in/Trendlyne

---

### 1.6 FII/DII Daily Flow Signal

**Evidence tier: Tier 1 (Poshakwale & Thapa, Journal of International Money and Finance)**

- Bidirectional Granger causality: FII flows lead NSE Nifty returns by 1–3 days (peer-reviewed)
- FII net long in **index futures** (not cash market) is the cleaner signal
- **Best use:** regime-level indicator, not precise daily timer
  - FII cash + F&O pointing same direction → stronger signal
  - DII as contra-signal: when FII sells and DII buys, domestic conviction high

**Data:** NSE FII/DII daily: nseindia.com/reports/fii-dii (free), NSDL participant-wise OI (end-of-day)

**Important:** The signal is at 1–3 day horizon, not intraday. Don't trade FII flow data within the same session it's released.

---

### 1.7 Bulk/Block Deal Follow-Through

**Evidence tier: Tier 2 (NSE-IGIDR Working Paper, IIMB Management Review)**

- Karmakar and Madhusudan (2016, IIMB): block deals predict 5-day forward returns with **62–68% directional accuracy**
- Buyer-initiated block deals by institutions → positive abnormal return of 0.6–1.2% over 3 trading days
- **Most signal value** when: MF block buy + stock near 52-week breakout + positive earnings backdrop

**Data:** NSE bulk deals: nseindia.com/market-data/block-deal-watch (free daily CSV)

---

### 1.8 Calendar Anomalies (NSE-Documented)

**Evidence tier: Tier 1 (PMC/Springer Nature, 2022, month-of-year study)**

| Month | Effect | Significance |
|---|---|---|
| September | Significantly higher returns | Strong — especially mid/small-cap |
| November–December | Significantly positive | Consistent |
| March–May | Significantly negative/lower | Consistent |
| F&O expiry month-end | Volume spike, gamma effects | Well-documented |
| First trading day of month | SIP inflow demand surge | ArXiv 2507.04859 (22-year study) |

**SIP date effect:** 100M+ SIP investors create predictable first-day-of-month buying pressure. **Use:** avoid buying on the 1st trading day; use it as a planned exit window for momentum longs.

---

### 1.9 Pharma ANDA Filings — Event-Driven Signal

**Evidence tier: Tier 1 (JFQA, Hwang 2011, replicated)**

- FDA ANDA first-to-file status predicts **+2.1% 3-day CAR** for Indian pharma stocks
- Relevant stocks: Dr. Reddy's, Sun Pharma, Cipla, Aurobindo, Gland Pharma
- **Data:** FDA.gov — free, public. Automated monitoring via FDA's RSS feeds

---

## 2. ML Architecture: What Beats GBDT (and What Doesn't)

### The Verdict: GBDT Remains the Default for Tabular Equity Data

**Core finding from NeurIPS 2022 (Grinsztajn et al., most-cited paper on this topic):**
> "Tree-based models still outperform deep learning on tabular data" — specifically because:
> 1. Trees are robust to uninformative features (financial data has many)
> 2. Neural nets struggle with irregular target functions (financial returns exhibit this)
> 3. Rotation invariance matters in financial data; trees preserve it naturally

**QuantBench (arXiv 2504.18600, 2025) — industrial-grade benchmark, US+China equities 2003–2024:**
> "Low signal-to-noise environments cause overfitting in deep learning more severely than in GBDT."

---

### What Has Documented Live/OOS Evidence vs GBDT

| Architecture | Evidence | Beat GBDT? | Notes |
|---|---|---|---|
| **DeepLOB (CNN+LSTM on LOB)** | Replicated (LSE, Helsinki) | **YES** — on L2 data | Requires 10-level tick LOB. Only valid at sub-minute horizons. |
| **VSN + LSTM hybrid** | Oxford 2026 large-scale futures benchmark | Conditional | Best on futures systematic; not tested on NSE tabular |
| **TFT (Temporal Fusion Transformer)** | Oxford 2026, Sharpe 2.27 on futures | Conditional | Wins on futures/multi-input. Handles expiry dates, FII flow as covariates |
| **GBDT → neural head hybrid** | Qlib/QuantBench evidence | Marginal yes (5–10% IC) | Add FT-Transformer at 15–20% weight as 4th ensemble member |
| **PatchTST** | Oxford 2026, Sharpe 2.31 on futures | Conditional | Futures OOS evidence only; not NSE equity tabular |
| **LSTM/GRU** | No live equity evidence | No | Competitive with GBDT in Qlib IC metrics; not better |
| **Mamba/SSM** | 1–2 years old, no replication | No | Papers compare to other neural architectures, not GBDT |
| **TabNet** | Weak vs GBDT in most benchmarks | No | Outdated; TabR/FT-Transformer are better tabular DL |
| **N-BEATS / N-HiTS** | M-competition wins | No | Win on forecasting error metrics, not equity Sharpe |
| **Informer / Autoformer** | Beaten by 1-layer linear models | No | DLinear (AAAI 2023) beats them on benchmarks |

**FinTSB (arXiv 2502.18834, 2025) critical finding:** No single model class (statistical, ML, or DL) dominates across all market regimes. **Regime detection matters more than architecture choice.**

---

### What to Actually Do

**Immediate:** Do NOT replace your XGB+LGB+CatBoost ensemble. The evidence does not support it.

**Optional enhancement:** Add FT-Transformer or LSTM as a 4th ensemble member at 15–20% weight. Monitor IC contribution after 6 months of paper trading. Only keep it if it adds measurable diversity.

**The one legitimate DL opportunity:** If you get NSE Level-2 tick data (10-level LOB from Shoonya or TrueData), a DeepLOB-style CNN+LSTM provides **entry timing signals** (next 10–30 minutes) that complement your 15-min GBDT signal. This is the only area where DL clearly beats GBDT in replicated studies.

**Regime detection as meta-layer:** HMM regime classification on Nifty data (2018–2024 backtest) improved Sharpe from ~0.67 (buy-and-hold) to **1.05** — a 57% improvement. This is more valuable than any architecture upgrade.

---

## 3. Alternative Data: Live Alpha Evidence

### Evidence Tiers Used
- **Tier 1:** Live/real-money performance, peer-reviewed or credible practitioner firm
- **Tier 2:** Rigorous OOS backtest, academic or reputable buy-side (not vendor)
- **Tier 3:** Vendor white paper — treat with heavy skepticism

---

### Summary Table

| Alt Data | Best Evidence | Documented Performance | India Applicable? | Cost |
|---|---|---|---|---|
| **NSE Bulk/Block Deals** | Tier 2 (IIMB) | 62–68% hit rate, 0.6–1.2% 3-day CAR | YES (native) | Free |
| **Promoter Pledging** | Tier 2 (Emerald) | Crash predictor, -3.4%/qtr high-pledge | YES (native) | Free |
| **FII/DII Daily Flows** | Tier 1 (JIMF) | 1–3 day lead, Sharpe ~0.5–0.7 | YES (native) | Free |
| **PEAD / Earnings Surprise** | Tier 1 (SCIRP) | 10% quarterly hedge portfolio | YES (NSE) | Free |
| **Pharma ANDA Filings** | Tier 1 (JFQA) | +2.1% 3-day CAR on first-to-file | YES (pharma) | Free (FDA) |
| **Earnings Call NLP** | Tier 1 (ExtractAlpha) | Sharpe 2.57 (2006–2024, US) | Partial (English calls) | FinBERT free |
| **Credit Card Transactions** | Tier 2 (JFE 2020) | Sharpe 1.3–1.8 (2010–2018) | NO | $150k+/yr |
| **Satellite Imagery** | Tier 3 (vendor) | Claimed Sharpe 1.1–1.4, now crowded | NO | $50k+/yr |
| **Patent/R&D Innovation** | Tier 1 (JFE 2013) | 2.8–3.3% annualized alpha | Partial (pharma ANDA) | Free (USPTO) |
| **13F/Institutional Filings** | Tier 1 (Griffin 2009) | 1.1% remaining alpha (45-day lag) | NO (no India equiv) | Free (SEC) |
| **Options Flow (CBOE)** | Tier 1 (Pan-Poteshman) | Sharpe ~1.4 (2001–2004) | NO | Proprietary |
| **App Downloads** | Tier 2 (Kim 2021) | 67% hit rate for app-heavy cos | Partial (Zomato, Paytm) | $2k–10k/yr |
| **Social Sentiment** | Tier 1 (Boehmer) | Sharpe 0.3–0.5 gross, ~0 net | Partial | Free |
| **ESG Scoring** | Tier 1 (Pedersen) | NO independent alpha | YES (no value) | Various |

**Man Group ranking of alt-data live IR (from their public research blog):**
Credit card > Web traffic > Job postings > Satellite > Social sentiment (last)

---

### What's Accessible for NSE Retail

**Immediately implementable (free data):**
1. NSE Bulk/Block Deals (nseindia.com) — download daily
2. FII/DII flows (nseindia.com/reports) — download daily  
3. BSE/NSE shareholding disclosures → promoter pledge % change
4. NSE earnings filings → PEAD calculation
5. FDA ANDA filings → pharma event signal
6. IIM Ahmedabad factor data library (free) → factor construction templates

**Low-cost (Screener.in / Trendlyne, ₹5k–15k/year):**
- Aggregated bulk deal data with institution names
- Insider trading disclosures
- Promoter activity tracking
- Momentum/quality factor screens

---

## 4. Reinforcement Learning: Honest Live Assessment

### Executive Summary

RL for **directional equity trading** has near-zero documented live-money results. The two sub-domains where RL has genuine documented success:
1. **Execution optimization** (order splitting/liquidation against TWAP/VWAP)
2. **Market making** (inventory management in LOB)

Everything else — directional signal RL, position sizing RL, MARL portfolios — is backtest-only with most results failing during regime change.

---

### What Has Documented Performance

| Application | Algorithm | Best Documented Result | Live-Ready? |
|---|---|---|---|
| **Execution optimization** | PPO/DDPG on LOB | 0.7–10 bps improvement vs TWAP (arXiv 2601.04896) | Partially — needs LOB data |
| **VWAP execution** | Hierarchical RL (M3T) | **1.16 bps average cost saving** vs optimal baseline | Research-stage |
| **Market making** | RL-AS hybrid | Sharpe improvement +0.17 on crypto LOB | Crypto only, not NSE |
| **Formulaic alpha (WorldQuant-style)** | RL + MCTS (Alpha2) | OOS alpha improvement documented | Needs portfolio layer |
| **Offline RL portfolio** | CQL/Decision Transformer | Competitive with classical; no live results | No — needs constraint modeling |

**RBC Aiden (live, European markets, 2024):** Deep RL for VWAP execution demonstrated measurably improved execution quality in live deployment. This is the most credible institutional case.

**JPMorgan LOXM:** Uses RL for order placement optimization across global equity markets. No exact metrics published.

---

### FinRL Live Result (the only disclosed paper trading)

**FinRL-X (arXiv 2603.21330, Oct 2025–Mar 2026, Alpaca paper trading):**
- Return: +19.76%, Sharpe ~1.10, max DD -21.46%
- Authors' own statement: *"Given the limited horizon, results were not intended to establish statistically significant alpha."*
- This validates the execution pipeline, not the alpha.

**Honest comparison:** PPO on S&P 500 (SSRN 2024) — buy-and-hold Sharpe **1.46** beat RL Sharpe **1.23**. One of the few honest published comparisons — **buy-and-hold won**.

---

### Why Directional RL Fails (Documented)

1. **Non-stationarity:** Policies trained on 2015–2021 QE-driven data failed catastrophically in 2022 rate-hike regime
2. **Reward hacking:** Agent learns to hold no positions (zero variance = good Sharpe metric) rather than trade
3. **Simulation gap:** RL trained on simplified market models fails on real order books with partial fills, market impact
4. **Data starvation:** ~252 observations/year. A neural policy with millions of parameters memorizes noise

**Suspicious result to avoid:** Springer 2025 India paper claims Sharpe **4.5 OOS** on NIFTY. Almost certainly overfitted. Realistic NSE OOS Sharpe for any systematic equity strategy: **0.8–1.5**.

---

### For Your NSE System

**Use RL for:** Execution optimization ONLY — when single orders exceed ~0.5% of daily volume (Nifty futures, large equity blocks).

**Do not use RL for:** Directional signal generation on 15-min NSE equity data. No live evidence supports it.

**Theoretical best path:** Offline RL (CQL or Decision Transformer) trained on 5+ years of historical data, deployed conservatively. Avoids live exploration risk. No live results exist yet but theoretically sounder than online RL.

---

## 5. LLM/NLP/Sentiment: What Has Shown Real Alpha

### What Actually Works

**Earnings Call Q&A Section NLP — Strongest Documented Signal**

The Q&A section (analyst questioning phase) is a stronger predictor than prepared management statements. Hard to rehearse or legal-review.

- 1-standard-deviation increase in LLM "outlook score" → **1.22 percentage-point higher returns** over subsequent month (S&P 500, 2024 study)
- **Sharpe 2.57** (ExtractAlpha Transcripts Model, 2006–2024, factor/industry-neutralized) — the single strongest documented NLP alpha in published research
- Conference call tone changes predict next quarter earnings surprise (Journal of Accounting Research, 2023)

**Document change analysis (10-Q/SEBI filings comparison):**
- Compare consecutive filings for language changes (Legal Proceedings, Risk Factors sections)
- Avoids LLM look-ahead bias since it compares period-to-period
- Quantified legal risk changes = forward return predictor

**Overnight sentiment → opening trade:**
- Accumulated overnight news (non-trading hours) predicts first 15–30 minutes of next-day returns
- The **strongest remaining window** for news sentiment alpha
- Intraday news in large-caps has alpha decay measured in **minutes** — too fast for non-co-located traders

**FinBERT vs. Loughran-McDonald:**
- FinBERT accuracy: 72.2% vs Loughran-McDonald 50.1% on financial sentiment classification
- The improvement is real and translates to better signals. Not a standalone strategy — contributes 0.2–0.5 Sharpe as a factor.

---

### What Does NOT Work

**General-purpose LLM trading agents (LiveTradeBench, 2025 — 50 live days, 21 models):**
- Most LLM agents **underperform buy-and-hold**
- Correlation between LLM benchmark score and trading returns: **0.054** — essentially random
- "Being a better language model does not mean being a better trader"

**GPT-4 look-ahead bias (critical):**
- Glasserman & Lin (SSRN 2023): When entity masking is applied (removing company names/dates), GPT-4's apparent predictive power **decreases substantially**
- LLaMA-2 prompted on September–November 2019 earnings calls mentions **COVID-19 in 25%+ of responses** — information that didn't exist then
- A model with training cutoff after the event "remembers" outcomes rather than predicts them
- **Any backtest of LLM-based trading using pre-2024 data is almost certainly contaminated by look-ahead bias**

**Reddit/WSB:**
- Short-term positive alpha (days), **long-term negative alpha** (weeks-months) — classic pump pattern
- WSB attention **significantly reduces holding-period returns** — institutional traders fade WSB
- ACM study 2024: retail WSB flows are uninformed; go against the crowd at extremes

**News sentiment decay:**
- IR from news sentiment: **0.76 (2003–2007) → 0.25 (2008–2017)** — 66% decline
- Coincides with proliferation of commercial sentiment products (RavenPack 2007, Bloomberg 2013)
- Baseline sentiment strategies collapsed in 2023 regime rotation

---

### India-Specific NLP

**Gap = opportunity:**
- No published India-specific financial NLP model with documented live alpha
- NSE earnings calls are shorter, less detailed, but processable with FinBERT
- Hindi financial news (Zee Business, Moneycontrol Hindi) is a completely untapped signal
- SEBI/BSE announcement XML feeds are machine-readable (free) — **no academic study has systematically exploited this**

**Practical build for NSE:**
```python
# Step 1: Scrape NSE earnings call transcripts (Screener.in, Trendlyne)
# Step 2: Apply FinBERT for Q&A sentiment vs prepared statement sentiment
# Step 3: Compute sentiment delta quarter-over-quarter
# Step 4: Use as an additional feature in the GBDT ensemble (not standalone strategy)
# Expected contribution: 0.2–0.5 Sharpe incremental
```

---

## 6. Execution Technology: Documented Improvements

### Square-Root Market Impact Law (Implement Immediately)

The **most empirically robust finding in market microstructure** — validated across equities, FX, futures, crypto:

```
MI = 0.1% × σ × √(Q / ADV)
```

Where: σ = daily volatility, Q = order size, ADV = average daily volume

**NSE examples:**
- Nifty 50 stock (σ=2%, ADV=₹50Cr), ₹1Cr order → MI ≈ **28 bps one-way**
- Same stock, ₹5L order → MI ≈ **6 bps** (manageable)
- Nifty futures (ADV>₹500Cr), ₹50L position → MI < **1 bp** (negligible)

**Rule:** Never size mid-cap positions above 0.5% ADV if you have intraday signals.

---

### F&O Is Categorically Better for Execution at Retail India Scale

| Instrument | Bid-Ask Spread | Impact at ₹50L | Verdict |
|---|---|---|---|
| Nifty 50 large-cap equity | 0.003–0.05% | 5–20 bps | Acceptable for daily signals |
| NSE mid-cap equity | 0.05–0.5% | 20–100 bps | Kills intraday alpha |
| Nifty futures | 0.003–0.006% | <1 bp | Best execution at retail scale |
| BankNifty futures | 0.003–0.006% | <1 bp | Best execution at retail scale |
| Nifty ATM options | 0.5–2% of premium | Variable | Use for F&O-specific signals only |

**Recommendation:** If your signal applies to the broad market or banking sector, route through Nifty/BankNifty futures rather than equity. Same directional exposure, 10–20x lower market impact.

---

### TFI (Trade Flow Imbalance) as Execution Filter

**Evidence:** Kolm, Turiel, Westray (Mathematical Finance, 2023) — Deep Order Flow Imbalance: R² up to 2–5% at 1-min horizon using OFI decomposition

**For NSE at 15-min bars (simplified but actionable):**
```python
# Shoonya provides 5-level depth. Compute:
TFI = (bid_volume_l1 + bid_volume_l2 + bid_volume_l3) - 
      (ask_volume_l1 + ask_volume_l2 + ask_volume_l3)

# Rule: 
# - Strongly negative TFI at entry time → delay limit buy, use market more aggressively
# - Strongly positive TFI at entry time → post passive limit buy, likely fill
# - TFI filter reduces adverse selection cost by estimated 1–3 bps per trade
```

---

### Pre-Open Auction Imbalance Signal

**Evidence:** arXiv 1802.01921 (2018), Imperial College dissertation (Morand) — documented 5–10 min post-open predictive power

**NSE-specific:**
```
Pre-open indicative price vs previous close → direction of opening pressure
Pre-open order imbalance (buy qty - sell qty) → magnitude of opening move
Most predictive window: first 5 minutes of regular trading (9:15–9:20)
```

**Cost:** Zero. NSE publishes pre-open data in real-time feed. Already visible in Shoonya WebSocket.

---

### The LIMIT vs. MARKET Decision Rule

**Key finding:** Passive limit orders at mid have a **negative drift problem** — fills happen when market moves against you (adverse selection on fills).

- Nifty 50 stocks: Expected adverse drift on limit fill ≈ 1–3 bps
- Market orders: Cost 0.2–0.5 bps spread, zero adverse drift

**Updated execution rule:**
- Use LIMIT at mid when signal timing is flexible (±5 minutes) AND TFI is positive (flow in your direction)
- Use MARKET (or near-touch limit) when signal timing is tight OR TFI is adverse
- **NEVER convert stale LIMIT to MARKET** — cancel entirely (this rule from CLAUDE.md is correct)

---

### Almgren-Chriss IS for Large Orders

For any order exceeding 1% of ADV, implement IS-style scheduling:

- Slice with volume participation capped at 5–10% of displayed depth per slice
- Back-load in trending markets (reduce early execution to avoid unfavorable early fills)
- Front-load in mean-reverting/neutral markets (complete early, reduce variance)
- Build your personal impact model: after 300 trades, regress IS ~ order_size + volatility + TFI_at_entry + time_of_day

---

## 7. Factor Investing: What Survived Post-2020

### Factors Still Working

| Factor | Status | India Evidence |
|---|---|---|
| **Quality (ROIC, low debt, stable earnings)** | Most resilient — admits multiple interpretations, hard to crowd uniformly | 13.5% CAGR NSE, Quality+Mom: 17.95% |
| **Short-term reversal** | Microstructure effect, stable but small (Sharpe 0.5–1.5) | Present in NSE, underexplored |
| **Low volatility** | Outperforms in India — **not yet fully arbitraged** | 12.38–15.2% CAGR, Nifty100LowVol30 |
| **Carry (FI/FX)** | Genuine premium, consistent | Less applicable to NSE equity |
| **Earnings quality/PEAD** | Persistent in India's less-covered universe | 10% quarterly return documented |

### Factors That Have Decayed

| Factor | Status | Mechanism |
|---|---|---|
| **Momentum (global)** | ~80% of 1990s alpha gone — ~2% annually now | Factor ETF crowding ($400B AUM by 2024) |
| **Value (global)** | Multi-year underperformance 2010–2020; uncertain recovery | Growth-at-any-price era distorted valuations |
| **Trend following (CTA)** | Man AHL Diversified: -0.23% in 2024, -3.46% in 2023 | Choppy central-bank-dominated environment |
| **Quant multi-factor** | AQR drawdowns 2018–2020 from crowded positions | Simultaneous unwind when one large fund deleverages |

**"Not All Factors Crowd Equally" (arXiv 2512.11913, 2024) — key empirical finding:**
- Momentum: hyperbolic alpha decay, R²=0.65 fit, crowding accelerated post-2015 factor ETF growth
- Quality: only factor with **positive Minimum Regime Performance** — most crowding-resistant
- Value: does NOT show crowding — judgment-based factors admit too many interpretations

**India-specific advantage:** The low-volatility and quality-momentum anomalies appear **less arbitraged in India** than in global developed markets. India's retail-dominated ecosystem (93% F&O traders lose money) and thin institutional coverage of mid-cap creates persistent inefficiencies.

---

## 8. The Failures: Documented Post-Mortems

### 8.1 Quantopian — The Definitive Failure Analysis

Quantopian (2011–2020): raised significant capital, ran Quantopian Fund I (2017), shut down October 2020 after returning all investor capital.

**Why it failed (five documented root causes):**

1. **Adverse selection of strategies:** Best quants kept alpha proprietary. Community shared *second-best* ideas. Selection effect was structurally toxic.

2. **Systematic overfitting:** Zipline backtester made in-sample optimization easy. Community incentive → maximize backtest Sharpe to receive allocations → overfit to history.

3. **Crowding:** Multiple users discovering the same factors created a fund running correlated books. When any strategy failed, they all failed together.

4. **No execution realism:** Amateur researchers didn't model slippage, market impact, borrow costs. Strategies looking like 20% annual returns → live performance 2% or worse.

5. **Scale mismatch:** Small signals that work at $100K evaporate at $10M due to market impact. Fund needed institutional scale to be viable as a business; this destroyed the edge.

**Lesson:** Backtest infrastructure without realistic cost modeling produces dangerous results. Correlated strategies masquerade as diversification until a regime change exposes it.

---

### 8.2 The Deep Learning Pattern

Consistent failure mode documented across hundreds of papers:
- LSTM/Transformer/CNN: impressive in-sample (Sharpe 2–4) → collapses in live deployment (Sharpe ~0)
- "Effective sample size" problem: ~5,000 daily observations over 20 years. Neural network with millions of parameters memorizes noise.
- The 2022 regime change (aggressive Fed tightening) wiped out virtually every ML model trained on 2012–2021 QE-driven momentum data.

**McLean & Pontiff (2016, Journal of Finance):** 97 published return predictors show real out-of-sample returns **26% lower** than in-sample average.

---

### 8.3 RL Trading: Named Failures

- **Sentient Technologies** ($143M VC): 4% return in 2017, near-zero 2018. Liquidated.
- **Rogers AI Global Macro ETF:** Launched June 2018, shuttered July 2019, near-zero returns.
- Both: "impressive backtest / flat live" — the endemic pattern of ML trading strategies.

---

### 8.4 Alternative Data Crowding Lifecycle

**Parking lot satellite data:**
- Alpha when exclusive (2012–2016): documented 4–5% return in 3-day window around earnings
- After proliferation (2018+): alpha halved within 18 months per Battle of the Quants (2019–2020) practitioners
- Now: effectively zero as standalone signal; only alpha in novel multi-source combinations

**Credit card transaction data:**
- Initial edge (2015–2020): significant alpha for retail/e-commerce earnings nowcasting
- Current status: crowded. Second Measure acquired by Bloomberg for ~$150M in 2021 — data now institutionally priced
- Cost ($200k–$1M+/year) requires large alpha to justify; signal has decayed

**Pattern:** Any data set sold to 50+ hedge funds gets arbitraged within 2–3 years of widespread adoption.

---

### 8.5 The Backtesting Inflation Numbers (Bailey & López de Prado)

The Deflated Sharpe Ratio corrects for three inflation sources:
1. Multiple testing (running 100 parameter combinations, reporting the best)
2. Non-normal returns (fat tails inflate standard Sharpe)
3. Short sample length

**Empirical estimates:**
```
Expected maximum Sharpe from noise alone ≈ √(2 × log(N trials))
With N=100 trials: noise Sharpe ≈ √(2 × log(100)) = √9.2 ≈ 3.0

Meaning: a backtest Sharpe of 3.0 from 100 parameter combinations 
could be entirely statistical noise.
```

**Industry implication:** A backtest Sharpe >2.0 developed after extensive parameter optimization should be expected to deliver **0.5–0.8 live Sharpe at best** after DSR correction.

---

### 8.6 What Fails Specifically in India

| Global Strategy | India Failure Mode | Reason |
|---|---|---|
| HFT market making | Unprofitable | STT destroys economics at HFT turnover volumes; SEBI order-to-trade penalties |
| Momentum in large-cap | Below benchmark | All momentum alpha is in illiquid stocks; liquid large-cap momentum: 8.51% CAGR < Nifty |
| Pairs trading / stat arb | Constraint | Short selling requires stocks on approved list; many mid-caps cannot be shorted |
| RL trained on US gym envs | No transfer | No NSE-specific constraints (lot sizes, STT, SEBI margin, circuit breakers) modeled |
| High-leverage derivative strategies | Margin-constrained | SEBI 2020 margin reforms + April 2026 STT increases dramatically changed cost structure |
| Weekly options expiry plays (post-Sep 2025) | Broken signals | Nifty now expires Tuesday (was Thursday) — all Thursday max-pain strategies invalidated |

---

### 8.7 LLM Look-Ahead Bias — The Hidden Failure

**Critical finding (arXiv 2510.07920, "Profit Mirage"):**
- Standard LLMs tested on 2021 stock data achieve returns >44% — because they were trained on retrospective analysis of 2021
- When tested on post-training-cutoff data (mid-2024): performance **collapses by ~22 percentage points**
- LLaMA-2 prompted on Sep–Nov 2019 earnings calls mentions COVID-19 in **25%+ of responses**

**Any backtest of an LLM-based trading strategy using pre-model-training-cutoff data is contaminated.**

---

## 9. Actionable Priority List for Your NSE System

### Tier 1: High Confidence, Zero/Low Cost, Implement Now

| Priority | Action | Expected Edge | Effort |
|---|---|---|---|
| **1** | Implement PEAD signal: rank Nifty 500 by earnings surprise quarterly, long top decile 40–60 days | 10% quarterly hedge portfolio (documented NSE-specific) | Medium |
| **2** | Add regime detection (HMM): bull/bear/sideways → switch signal weights | Sharpe ~0.67 → 1.05 documented on Nifty | Low (hmmlearn already in stack) |
| **3** | Integrate NSE Bulk/Block Deal feed: flag institutional buying for momentum confirmation | 62–68% directional hit rate, 0.6–1.2% 3-day CAR | Low (free CSV) |
| **4** | Implement square-root impact model for position sizing: MI = 0.1% × σ × √(Q/ADV) | Prevents alpha destruction in mid-cap | Low |
| **5** | Add TFI execution filter from Shoonya 5-level depth | 1–3 bps per trade adverse selection reduction | Low |
| **6** | Pre-open imbalance signal: monitor indicative price vs previous close at 9:08–9:14 | 5–10 min post-open predictive power | Low (free, NSE feed) |
| **7** | Add promoter pledging filter: avoid/short stocks with >25% and rising pledge % | Crash risk reduction + short signal | Low (Screener.in free) |
| **8** | Track FII F&O net OI daily: use as 1–3 day directional regime indicator | Tier 1 peer-reviewed signal | Low (NSDL free) |

### Tier 2: Worth Building, Requires Some Infrastructure

| Priority | Action | Expected Edge | Effort |
|---|---|---|---|
| **9** | FinBERT on NSE earnings call transcripts: sentiment delta feature in GBDT ensemble | Incremental 0.2–0.5 Sharpe contribution | Medium (data pipeline) |
| **10** | Quality-Momentum screen on Nifty 500: quarterly rebalance | 17.95% CAGR documented 18 years | Medium |
| **11** | TFT as optional 4th ensemble member (alongside XGB/LGB/CatBoost): handle expiry calendar, FII flow as known-future inputs | 5–10% IC improvement if data-rich enough | Medium-High |
| **12** | Pharma ANDA filing monitor: subscribe to FDA RSS, alert on Indian pharma first-to-file | +2.1% 3-day CAR, event-driven | Low-Medium |
| **13** | SIP date execution: avoid buying on 1st trading day of month; use it as planned exit window | Small execution edge | Low |

### Tier 3: Monitor, Do Not Implement Yet

| Priority | Action | Status |
|---|---|---|
| **14** | DeepLOB (CNN+LSTM on 10-level LOB) | Wait until TrueData or NSE L2 tick data subscription is in place |
| **15** | Offline RL (CQL/Decision Transformer) for position sizing | No live evidence yet; theoretically sound but premature |
| **16** | Mamba/SSM as feature extractor | Research-stage, no replication on financial tabular data |
| **17** | GNN for Tata/Adani group contagion | Theoretically applicable; no published live results |

### What to Never Build

| Do NOT Build | Reason |
|---|---|
| Pure directional RL for signals | No live evidence; named fund failures documented |
| LLM as standalone trading agent | 0.054 correlation with returns; look-ahead bias |
| Social media sentiment as primary signal | Sharpe ~0.3–0.5 gross, ~0 after costs |
| HFT strategies | STT economics + SEBI rules make it structurally unprofitable in India |
| Deep learning replacing GBDT for 15-min tabular signals | NeurIPS 2022 consensus: GBDT wins on low-SNR tabular data |
| Satellite imagery for India | No applicable data at useful granularity |
| Strategies trained only on 2012–2021 data | That era (QE, low-vol, momentum regime) was anomalous; regime change risk is severe |

---

## 10. Key Sources

### India-Specific
- [IIM Ahmedabad Four-Factor Model India (free data library)](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/)
- [BacktestIndia: NSE Factor Backtests 2006–2025](https://backtestindia.com/)
- [NSE Nifty500 Momentum 50 Factsheet](https://www.niftyindices.com/Factsheet/FactsheetNifty500Momentum50.pdf)
- [Post-PEAD Anomaly India (SCIRP 2018)](https://www.scirp.org/journal/paperinformation?paperid=88060)
- [Lalwani: Factor Strategies Out-of-Sample NSE (SSRN 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4748791)
- [Month-of-Year Effect India (PMC/Springer 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8742668/)
- [Promoter Pledging & Crash Risk (ResearchGate 2024)](https://www.researchgate.net/publication/378335023)
- [FII Trading Behaviour PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9145119/)
- [NSE-IGIDR Insider Trading Working Paper](https://nsearchives.nseindia.com/research/content/NSE-IGIDR-WP4-14-15.pdf)
- [F&O Expiry vs SIP Timing 22-Year Study (arXiv 2507.04859)](https://arxiv.org/html/2507.04859v2)
- [SEBI Study: 93% F&O Traders Lost Money FY22-FY24](https://aibi.org.in/Sebipr/Updated_SEBI_Study_Reveals_93_percentage_of_Individual_Traders_Incurred_Losses_in_Equity_F&O_between_FY22_and_FY24.pdf)
- [NSE Bulk Deals Data](https://www.nseindia.com/market-data/block-deal-watch)
- [NSE FII/DII Daily Data](https://www.nseindia.com/reports/fii-dii)

### ML Architecture
- [Why Tree Models Outperform DL on Tabular Data (NeurIPS 2022)](https://www.researchgate.net/publication/362123616_Why_do_tree-based_models_still_outperform_deep_learning_on_tabular_data)
- [QuantBench: Benchmarking AI for Quantitative Investment (arXiv 2504.18600)](https://arxiv.org/abs/2504.18600)
- [FinTSB: Financial Time Series Benchmark (arXiv 2502.18834)](https://arxiv.org/pdf/2502.18834)
- [Large-Scale DL Benchmark for Finance (arXiv 2603.01820)](https://arxiv.org/abs/2603.01820)
- [DeepLOB: CNN for Limit Order Books (arXiv 1808.03668)](https://arxiv.org/abs/1808.03668)
- [Deep Order Flow Imbalance (Mathematical Finance 2023)](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413)
- [Virtue of Complexity in Return Prediction (JoF 2024)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13298)
- [Deep Learning in Asset Pricing (Management Science 2024)](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4695)

### Reinforcement Learning
- [FinRL-X Paper Trading (arXiv 2603.21330)](https://arxiv.org/abs/2603.21330)
- [RL Systematic Review 167 Papers (arXiv 2512.10913)](https://arxiv.org/html/2512.10913v1)
- [DRL for Order Execution (arXiv 2601.04896)](https://arxiv.org/abs/2601.04896)
- [Hierarchical DRL VWAP M3T (arXiv 2212.14670)](https://arxiv.org/abs/2212.14670)
- [RBC Aiden VWAP Live Deployment](https://www.thetradenews.com/thought-leadership/rbcs-aiden-vwap-a-new-era-of-ai-trading-in-europe-leveraging-advanced-ai-and-deep-reinforcement-learning-to-achieve-optimised-execution/)
- [Backtest Overfitting in RL Trading (arXiv 2209.05559)](https://arxiv.org/abs/2209.05559)
- [Wharton Experimental AI Trading Study (2025)](https://wifpr.wharton.upenn.edu/wp-content/uploads/2025/09/Sangiorgi_Deep__Learning_to_Trade.pdf)

### Alternative Data
- [ExtractAlpha Transcripts Model: Sharpe 2.57](https://extractalpha.com/fact-sheet/transcripts-model-us/)
- [Man Group: Alternative Data What Works (Man Institute)](https://www.man.com/insights)
- [Satellite Data and Hedge Funds (Berkeley Haas)](https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/)
- [AIS Shipping Data and Returns (Bakshi et al. 2019)](https://www.aeaweb.org/articles?id=10.1257/app.20180150)

### LLM/NLP
- [Profit Mirage: LLM Look-Ahead Bias (arXiv 2510.07920)](https://arxiv.org/html/2510.07920v1)
- [LiveTradeBench: LLMs in Live Trading (arXiv 2511.03628)](https://arxiv.org/html/2511.03628v1)
- [GPT Look-Ahead Bias (Glasserman & Lin, SSRN 2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4586726)
- [MarketSenseAI (Springer 2024)](https://link.springer.com/article/10.1007/s00521-024-10613-4)

### Execution & Microstructure
- [Almgren-Chriss Optimal Execution (2000)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Market Impact Empirical Evidence (arXiv 2205.07385)](https://arxiv.org/pdf/2205.07385)
- [Negative Drift of Limit Order Fill (arXiv 2407.16527)](https://arxiv.org/html/2407.16527v1)
- [QB Regime-Based Limit Order Placement (2024)](https://www.quantitativebrokers.com/whitepapers/2024-07)
- [NSE Co-Location Scam History](https://en.wikipedia.org/wiki/NSE_co-location_scam)
- [Intraday Seasonality NSE (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165057)

### Factor Decay
- [Not All Factors Crowd Equally (arXiv 2512.11913)](https://arxiv.org/pdf/2512.11913)
- [McLean & Pontiff: Factor Decay Post-Publication (JF 2016)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365)
- [Deflated Sharpe Ratio (Bailey & López de Prado)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Quantopian Post-Mortem (Business of Business)](https://www.businessofbusiness.com/articles/how-quantopian-died-shut-down-quant-investment-robinhood/)

---

*Document compiled June 2026 from 7 parallel research agents covering 300+ primary sources.*
*All Sharpe ratios from backtests should be discounted 33–50% for live deployment.*

















---
  Free (run yourself):

  1. 5yr daily OHLCV — yfinance:
  yf.download("RELIANCE.NS", period="5y",
  interval="1d")
  2. 60-day 15-min OHLCV — yfinance:
  yf.download("RELIANCE.NS", period="60d",
  interval="15m")
  3. 3yr NSE EOD bhavcopy — pip install jugaad-data →
   bhavcopy_save(start, end, "data/")
  4. India VIX historical — NSE site: search
  nseindia.com → "India VIX historical data" →       
  download CSV
  5. Nifty 50 / 500 constituent list —
  nseindia.com/products/content/equities/indices/cons
  tituent_csv.htm
  6. F&O expiry calendar — NSE: search "NSE F&O      
  expiry dates calendar"

  ---
  Shoonya API (you already have account):

  7. 15-min candles up to 400 days back —
  api.get_time_price_series(exchange='NSE',
  token='22', interval=15)
  — token IDs: search "NSE symbol token list Shoonya"
   or pull from api.searchscrip()

  ---
  Paid (subscribe yourself):

  8. TrueData — https://truedata.in — ₹300–500/month 
  — needed only at Stage 2 for Hawkes/VPIN/TFI tick  
  data

  ---
  What you don't have yet and can't get free:        
  - 3yr 15-min intraday history → only from TrueData 
  or record live for 12+ months
  - Tick-by-tick L2 order book → TrueData only  onl