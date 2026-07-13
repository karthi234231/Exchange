# MASTER RESEARCH DATABASE v3.0
## AI Algo Trading System — India (NSE/BSE Focus) | May 2026
## Institutional & Solo Trader Level | Compiled from 15+ Research Agents

> **RESEARCH ARCHIVE — Not a build spec or source of truth**
>
> This file is a large research database compiled in May 2026. It contains useful background on brokers, APIs, libraries, and market mechanics, but:
> - It mixes current recommendations, archived claims, and speculative additions
> - Some broker/regulatory/pricing facts may be stale — always verify before acting
> - It is NOT the canonical architecture spec — use `arch-spec.json` for that
> - It is NOT the current decisions record — use `MD_Files/DECISIONS.md` for that
> - Several sections contain older AngelOne-primary assumptions that have since been superseded
>
> Use this file for research context only. Do not build code directly from its instructions.

> **v3.0** — Merged from: MASTER_RESEARCH_DATABASE.md (v2.0, 30 sections) + 6 deep expansion files + fixes/additions log.
> All fixes from DB_FIXES_AND_ADDITIONS.md applied inline. Covers GitHub repos, 300+ academic papers, ML/RL/DL/NLP, market microstructure, portfolio theory, factor research, Indian market specifics, full cost models, broker APIs, SEBI compliance, backtesting frameworks, and communities/certifications.

---

## TABLE OF CONTENTS

### PART A: ORIGINAL DATABASE (Sections 1–30, fixes applied)

1. [GitHub Repositories (by Stars)](#1-github-repositories-by-stars)
2. [Academic Papers — ML for Trading](#2-academic-papers--ml-for-trading)
3. [Academic Papers — Reinforcement Learning](#3-academic-papers--reinforcement-learning)
4. [Academic Papers — Deep Learning & Time Series](#4-academic-papers--deep-learning--time-series)
5. [Academic Papers — NLP & LLMs for Finance](#5-academic-papers--nlp--llms-for-finance)
6. [Academic Papers — Market Microstructure & Execution](#6-academic-papers--market-microstructure--execution)
7. [Academic Papers — Portfolio Theory & Risk](#7-academic-papers--portfolio-theory--risk)
8. [Academic Papers — Factor Research & Alpha](#8-academic-papers--factor-research--alpha)
9. [Academic Papers — Indian Market Research](#9-academic-papers--indian-market-research)
10. [Must-Read Books](#10-must-read-books)
11. [Institutional Research Sources](#11-institutional-research-sources)
12. [Indian Broker APIs (Full Comparison)](#12-indian-broker-apis-full-comparison)
13. [Indian Market Data Sources](#13-indian-market-data-sources)
14. [SEBI Regulations & Compliance (2025–2026)](#14-sebi-regulations--compliance-20252026)
15. [Indian Market Characteristics](#15-indian-market-characteristics)
16. [Tax & Compliance for Algo Traders in India](#16-tax--compliance-for-algo-traders-in-india)
17. [Free Learning Resources](#17-free-learning-resources)
18. [Institutional Quant Strategies](#18-institutional-quant-strategies)
19. [Portfolio Optimization Research](#19-portfolio-optimization-research)
20. [Recommended Implementation Path](#20-recommended-implementation-path)
21. [Advanced Transformer & Foundation Models](#21-advanced-transformer--foundation-models)
22. [Advanced Reinforcement Learning — 100+ Papers](#22-advanced-reinforcement-learning--100-papers)
23. [Options Trading Research (India F&O)](#23-options-trading-research-india-fo)
24. [Execution Algorithms & Transaction Cost Analysis](#24-execution-algorithms--transaction-cost-analysis)
25. [Advanced Feature Engineering](#25-advanced-feature-engineering)
26. [Free & Open Datasets](#26-free--open-datasets)
27. [Backtesting Frameworks — Deep Comparison](#27-backtesting-frameworks--deep-comparison)
28. [Advanced Risk Management Research](#28-advanced-risk-management-research)
29. [Alternative Data Sources](#29-alternative-data-sources)
30. [Communities, Certifications & Learning Ecosystem](#30-communities-certifications--learning-ecosystem)

### PART B: DEEP TECHNICAL EXPANSION

- [B1: Sections 1–5 Deep Expansion](#b1-sections-15-deep-expansion)
- [B2: Sections 6–10 Deep Expansion](#b2-sections-610-deep-expansion)
- [B3: Sections 11–15 Deep Expansion](#b3-sections-1115-deep-expansion)
- [B4: Sections 16–20 Deep Expansion](#b4-sections-1620-deep-expansion)
- [B5: Sections 21–30 Deep Expansion](#b5-sections-2130-deep-expansion)

### PART C: FIXES AND NEW ADDITIONS LOG

- [Deprecated Tools](#1-deprecated-tools--remove-or-replace)
- [Rating Corrections](#2-rating-corrections)
- [Missing Additions (Mamba, KAN, OpenAlgo, Polars, SEBI Updates)](#3-missing-additions)

---

# PART A: ORIGINAL DATABASE — Sections 1–30
## (All fixes from DB_FIXES_AND_ADDITIONS.md applied inline)

---

## 1. GitHub Repositories (by Stars)

### Tier 1 — Core Frameworks (10,000+ Stars)

| Stars | Repo | URL | Relevance to This Project |
|-------|------|-----|--------------------------|
| 42,000+ | **CCXT** | https://github.com/ccxt/ccxt | Multi-exchange API; crypto patterns applicable to multi-broker NSE abstraction |
| 39,900+ | **Freqtrade** | https://github.com/freqtrade/freqtrade | Mature trading bot with FreqAI (ML module); architecture reference for live loop |
| 16,600+ | **Microsoft Qlib** | https://github.com/microsoft/qlib | Full AI quant pipeline: alpha seeking → risk → portfolio → execution. **2025 UPDATE:** RD-Agent integration automates full quant R&D loop — NeurIPS 2025 showed 2x annualized returns vs classical factors using 70% fewer factors. (Stars corrected from earlier 35K estimate.) |
| 27,000+ | **PyTorch Lightning** | https://github.com/Lightning-AI/pytorch-lightning | DL training infrastructure for LSTM/Transformer models |
| ~22,000 | **NautilusTrader** | https://github.com/nautechsystems/nautilus_trader | Rust+Python production engine, nanosecond precision. End-goal production platform |
| 21,000+ | **Backtrader** ⚠️ DEPRECATED | https://github.com/mementum/backtrader | LAST RELEASE: May 2019. Star count is legacy from tutorial era. Do NOT start new projects. Use PyBroker (ML), NautilusTrader (production), or VectorBT (vectorized research) instead. |
| 20,000+ | **Zipline** | https://github.com/quantopian/zipline | Quantopian legacy; event-driven architecture reference |
| 19,006 | **QuantConnect Lean** | https://github.com/quantconnect/lean | Cloud + local backtesting. 483K users, 100+ hedge fund clients, $45B+ monthly notional. Self-hostable (LEAN engine, air-gapped). FIX 5.0 SP2 for institutions. Rating: 9/10 |
| 15,000+ | **awesome-quant** | https://github.com/wilsonfreitas/awesome-quant | Master list of every quant library — bookmark this |
| 15,000+ | **MLflow** | https://github.com/mlflow/mlflow | Experiment tracking + model registry. Critical for 50+ training runs |
| 14,000+ | **Hummingbot** | https://github.com/hummingbot/hummingbot | Market-making + HFT bot; multi-venue execution patterns |
| 10,000+ | **AKShare** | https://github.com/akfamily/akshare | High-quality data interface; data quality patterns applicable to NSE |
| 10,000+ | **FinRL** ⚠️ Research Only | https://github.com/AI4Finance-Foundation/FinRL | DRL framework (NeurIPS 2020). DQN, PPO, SAC, A2C, TD3. IMPORTANT: Research/academic use only — not production-ready. Survivorship bias in backtests. No verified live P&L published. |
| 10,000+ | **Stefan Jansen ML4T** | https://github.com/stefan-jansen/machine-learning-for-trading | Full book code (822 pages). Best practical ML trading codebase |

### Tier 2 — Specialized Tools (2,000–10,000 Stars)

| Stars | Repo | URL | Use Case |
|-------|------|-----|----------|
| 8,000+ | **Optuna** | https://github.com/optuna/optuna | Bayesian hyperparameter optimization (TPE, CMA-ES). Critical for XGBoost/LGB tuning |
| 5,300+ | **pandas-ta** | https://github.com/twopirllc/pandas-ta | 130+ technical indicators. Primary feature library |
| 5,000+ | **VectorBT** ⚠️ OSS Stagnant | https://github.com/polakowo/vectorbt | Vectorized backtesting (signal research). NOTE: Open-source version unmaintained — active dev moved to VectorBT PRO (~$20/month commercial). For ML strategies, prefer PyBroker. |
| 5,000+ | **Jesse** | https://github.com/jesse-ai/jesse | Advanced trading bot with ML pipeline; RL integration patterns |
| 5,000+ | **Darts** | https://github.com/unit8co/darts | 20+ time series models (ARIMA → TFT). Unified API |
| 5,000+ | **Stable-Baselines3** | https://github.com/DLR-RM/stable-baselines3 | RL algorithms (PPO, SAC, TD3, A2C). Used by FinRL. Production-quality |
| 3,500+ | **tsai** | https://github.com/timeseriesAI/tsai | SOTA deep learning for time series (Transformers, ResNets) on PyTorch/fastai |
| 3,500+ | **PyPortfolioOpt** | https://github.com/robertmartin8/PyPortfolioOpt | Markowitz, Black-Litterman, HRP position sizing |
| 3,329+ | **statsmodels** | https://github.com/statsmodels/statsmodels | ARIMA, GARCH, cointegration, VAR. Statistical backbone |
| 3,100+ | **PyBroker** | https://github.com/edtechre/pybroker | Walk-forward ML backtesting with scikit-learn/XGBoost built-in. BCa bootstrap Sharpe CI. Actively maintained. |
| 3,000+ | **FinRL-Trading** | https://github.com/AI4Finance-Foundation/FinRL-Trading | Next-gen FinRL (production-oriented, broker integration, LLM-era) |
| 3,000+ | **FinGPT** | https://github.com/AI4Finance-Foundation/FinGPT | Financial LLM for sentiment (the optional daily sentiment layer) |
| 3,000+ | **awesome-systematic-trading** | https://github.com/wangzhe3224/awesome-systematic-trading | Curated list: stocks, crypto, futures, options, FX |
| 2,500+ | **Riskfolio-Lib** | https://github.com/dcajasn/Riskfolio-Lib | Advanced portfolio optimization; CVaR, MDD optimization |
| 2,500+ | **hmmlearn** | https://github.com/hmmlearn/hmmlearn | Hidden Markov Models for regime detection (bull/bear/sideways) |
| 2,500+ | **filterpy** | https://github.com/rlabbe/filterpy | Kalman filters + particle filters. Kalman smoothed price feature |
| 2,500+ | **best-of-algorithmic-trading** | https://github.com/merovinh/best-of-algorithmic-trading | Ranked list of 110+ algo trading projects. Updated weekly |
| 2,000+ | **arch** | https://github.com/bashtage/arch | ARCH/GARCH volatility forecasting. GARCH vol forecast feature #15 |
| 2,000+ | **gym-anytrading** | https://github.com/AminHP/gym-anytrading | OpenAI Gym environments for trading RL agents |
| 2,000+ | **awesome-quant-ai** | https://github.com/leoncuhk/awesome-quant-ai | AI/ML specific quant trading resources |
| 2,000+ | **rl-baselines3-zoo** | https://github.com/DLR-RM/rl-baselines3-zoo | SB3 training framework + pre-trained RL models |

### Tier 3 — Indian Market Specific

| Stars | Repo | URL | Use Case |
|-------|------|-----|----------|
| 2,000+ | **OpenAlgo** | https://github.com/marketcalls/openalgo | Multi-broker API bridge — Zerodha, Upstox, Shoonya, Fyers, Angel One through one standard interface. Webhook-based order routing. **8/10 — essential for multi-broker NSE deployment** |
| 1,500+ | **pykiteconnect** | https://github.com/zerodha/pykiteconnect | Zerodha Kite Connect Python SDK |
| 1,000+ | **nsepy** | https://github.com/swapniljariwala/nsepy | NSE historical data (legacy but useful) |
| 700+ | **smartapi-python** | https://github.com/angel-one/smartapi-python | Angel One SmartAPI official Python SDK |
| 500+ | **DhanHQ-py** | https://github.com/dhan-oss/DhanHQ-py | Dhan official Python SDK |
| 400+ | **jugaad-data** | https://github.com/jugaad-py/jugaad-data | NSE/BSE bhavcopy downloader. Best free Indian data tool |
| 300+ | **ShoonyaApi-py** | https://github.com/Shoonya-Dev/ShoonyaApi-py | Finvasia/Shoonya zero-brokerage Python SDK |
| 200+ | **Kotak-Neo-API** | https://github.com/Kotak-Neo/kotak-neo-api | Kotak Neo official SDK |
| 200+ | **fyers-api-sample-code** | https://github.com/FyersDev/fyers-api-sample-code | Fyers API samples |
| 200+ | **Breeze-Python-SDK** | https://github.com/Idirect-Tech/Breeze-Python-SDK | ICICI Direct Breeze SDK |

### Tier 4 — Supporting Libraries

| Stars | Repo | URL | Use Case |
|-------|------|-----|----------|
| 600+ | **yfinance** | https://github.com/ranaroussi/yfinance | Yahoo Finance data (free, start here) |
| 400+ | **PyTorch Forecasting** | https://github.com/jdb78/pytorch-forecasting | TFT, N-BEATS, N-HiTS implementations |
| 300+ | **ruptures** | https://github.com/deepcharles/ruptures | Change point detection for regime identification |
| 300+ | **skfolio** | https://github.com/skfolio/skfolio | Scikit-learn compatible portfolio optimization |
| 300+ | **tvdatafeed** | https://github.com/rongardF/tvdatafeed | TradingView data downloader |

---

## 2. Academic Papers — ML for Trading

### XGBoost (Primary Ensemble Model)
- **Title:** XGBoost: A Scalable Tree Boosting System
- **Authors:** Tianqi Chen, Carlos Guestrin (Google)
- **Venue:** KDD 2016
- **ArXiv:** https://arxiv.org/abs/1603.02754
- **Citations:** 50,000+ (most cited ML paper)
- **Summary:** Introduces gradient boosted decision trees with regularization, sparsity-awareness, and distributed computing. The algorithm that dominates tabular data competitions.
- **Why Critical:** Your primary ML model. Millisecond inference, calibrated probabilities, built-in feature importance. Outperforms neural networks on tabular financial data.

### LightGBM (Second Ensemble Model)
- **Title:** LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- **Authors:** Guolin Ke et al. (Microsoft Research)
- **Venue:** NeurIPS 2017
- **ArXiv:** https://arxiv.org/abs/1706.01427
- **Citations:** 15,000+
- **Summary:** Introduces leaf-wise tree growth and Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). 20x faster than XGBoost with similar accuracy.
- **Why Critical:** Faster training on large datasets (1M+ bars of tick data). Use when XGBoost is too slow for retraining pipeline.

### CatBoost (Third Ensemble Model)
- **Title:** CatBoost: Unbiased Boosting with Categorical Features
- **Authors:** Liudmila Prokhorenkova et al. (Yandex)
- **Venue:** NeurIPS 2018
- **ArXiv:** https://arxiv.org/abs/1706.09516
- **Citations:** 6,000+
- **Summary:** Ordered boosting eliminates prediction shift (a form of leakage in gradient boosting). Handles categorical features natively without encoding.
- **Why Critical:** Adds diversity to ensemble. Often catches patterns XGBoost and LightGBM miss. Ensemble of three is more robust than any single model.

### Random Forests (Baseline)
- **Title:** Random Forests
- **Authors:** Leo Breiman
- **Venue:** Machine Learning Journal, 2001
- **Link:** https://link.springer.com/article/10.1023/A:1010933404324
- **Citations:** 80,000+ (foundational)
- **Summary:** Bagging + random feature selection to build decorrelated trees. Robust baseline for tabular prediction.
- **Why Critical:** Use as baseline to validate your XGBoost/LightGBM are actually adding value.

### Advances in Financial Machine Learning (Book → Reference Papers)
- **Title:** Advances in Financial Machine Learning
- **Author:** Marcos López de Prado (AQR Capital)
- **Publisher:** Wiley, 2018
- **Key Concepts:**
  - Triple-Barrier Labeling Method
  - Purging & Embargoing for Walk-Forward Validation
  - Fractional Differentiation (making price series stationary while preserving memory)
  - Combinatorial Purged Cross-Validation (CPCV)
  - Mean Decrease Impurity vs. Mean Decrease Accuracy (feature importance)
  - Backtest Overfitting (Deflated Sharpe Ratio)
- **Companion Papers:** https://mlfortraders.com (free PDFs)
- **Why Critical:** THE foundational text. Your labeling, validation, and feature importance methods come directly from this book.

### Deflated Sharpe Ratio (Backtest Overfitting)
- **Title:** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality
- **Authors:** David H. Bailey, Marcos López de Prado
- **Venue:** Journal of Portfolio Management, 2014
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- **Summary:** Adjusts Sharpe ratio for number of trials tested. If you test 100 strategies, your "best" will look great by chance alone. Deflated Sharpe corrects for this.
- **Why Critical:** Prevents you from convincing yourself a backtest is good when it's just chance.

### Stacking Ensemble with Out-of-Fold (OOF) Predictions
- **Key Reference:** Wolpert (1992) "Stacked Generalization" — Neural Networks; Breiman (1996) "Stacked Regressions" — Machine Learning
- **Summary:** Stacking trains a meta-learner (e.g., Logistic Regression or LightGBM) on the out-of-fold predictions of base models (XGBoost, LightGBM, CatBoost) rather than on the raw features. OOF = each training sample is predicted by a fold of the model that never saw it during training → the meta-learner learns from unbiased predictions → avoids double-dipping on the training set.
- **Procedure:** (1) Split training data into K folds. (2) For each fold k, train base models on K−1 folds, predict on fold k → collect OOF predictions. (3) Train meta-learner on concatenated OOF predictions. (4) At inference: base models predict → meta-learner takes those predictions → final signal.
- **Why Critical:** More robust than simple averaging because the meta-learner learns the optimal blend of models including their correlations and failure modes. Kaggle competition standard. Expect +0.5–2% accuracy improvement over simple mean ensemble. Deep coverage in CONVO_DEEP_RESEARCH.md Part 1.
- **Warning:** OOF must respect temporal ordering for financial data. Use TimeSeriesSplit, not random K-fold, or you introduce lookahead bias into the meta-learner.
- **Python:** `sklearn.model_selection.cross_val_predict` with `TimeSeriesSplit` for OOF generation.

---

## 3. Academic Papers — Reinforcement Learning

### FinRL (The Standard Reference) — ⚠️ Research Platform, Not Production-Ready

- **Title:** FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
- **Authors:** Xiao-Yang Liu, Hongyang Yang, Qian Chen, Runjia Zhang et al.
- **Venue:** NeurIPS 2020 Workshop (Deep Reinforcement Learning)
- **ArXiv:** https://arxiv.org/abs/2011.09607
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3737859
- **GitHub:** https://github.com/AI4Finance-Foundation/FinRL
- **Corrected Rating:** 6/10 overall | Research quality: 7/10 | Production readiness: 3/10
- **Summary:** First open-source DRL library for automated stock trading. Implements DQN, DDPG, PPO, SAC, A2C, TD3 with transaction cost and liquidity modeling.
- **HONEST CAVEAT:** FinRL's own documentation states: *"treat FinRL as an experimental platform rather than a profit guarantee."* Known issues: survivorship bias, low signal-to-noise ratio, simulation-to-reality gap, policy instability. No independently verified live P&L published.
- **Why Critical:** Use for RL algorithm comparison (PPO vs SAC vs TD3) and learning concepts. Build your own NSE environment with SB3 for production.

### Stable-Baselines3 (RL Implementation)
- **Title:** Stable-Baselines3: Reliable Reinforcement Learning Implementations
- **Authors:** Antonin Raffin, Ashley Hill, Adam Gleave et al.
- **Venue:** JMLR 2021
- **Paper:** https://jmlr.org/papers/v22/20-1364.html
- **GitHub:** https://github.com/DLR-RM/stable-baselines3
- **Summary:** Clean, documented, tested PyTorch implementations of PPO, SAC, TD3, A2C, DQN. Industry standard for RL research.
- **Why Critical:** Your actual RL implementation. Use PPO for position sizing agent (continuous action space).

### Trading-R1 (2025 — LLM + RL Hybrid)
- **Title:** Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning
- **Year:** 2025
- **Search:** ArXiv query "Trading-R1 financial trading reinforcement learning"
- **Summary:** Applies R1-style reasoning (chain-of-thought RL) to financial trading decisions. LLM reasons through market conditions before acting.
- **Why Critical:** Cutting edge. Shows how to combine LLM reasoning with RL signals — relevant to your optional sentiment layer.

### PPO Algorithm (Core RL Paper)
- **Title:** Proximal Policy Optimization Algorithms
- **Authors:** John Schulman et al. (OpenAI)
- **ArXiv:** https://arxiv.org/abs/1707.06347
- **Citations:** 15,000+
- **Summary:** PPO uses clipped surrogate objective to prevent large policy updates, making RL training stable. Best general-purpose policy gradient algorithm.
- **Why Critical:** PPO is what you'll use for your RL position sizing agent. Read this to understand why it works.

### SAC (Alternative to PPO)
- **Title:** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
- **Authors:** Tuomas Haarnoja et al. (UC Berkeley)
- **ArXiv:** https://arxiv.org/abs/1801.01290
- **Citations:** 10,000+
- **Summary:** Maximum entropy RL that encourages exploration while optimizing reward. Off-policy learning (more sample efficient than PPO).
- **Why Critical:** Better than PPO for continuous action spaces (position sizing). Try SAC vs PPO in your Optuna comparison.

### Deep Q-Network (DQN) — Foundation
- **Title:** Human-level control through deep reinforcement learning
- **Authors:** Volodymyr Mnih et al. (DeepMind)
- **Venue:** Nature 2015
- **Link:** https://www.nature.com/articles/nature14236
- **Citations:** 30,000+
- **Summary:** Deep neural network learns to play Atari games from pixels at human level. Introduced experience replay and target networks.
- **Why Critical:** Foundation of all deep RL. Understanding DQN explains why PPO/SAC improvements were needed.

### FinAgent (Multi-Modal LLM+RL)
- **Title:** FinAgent: A Multimodal Foundation Agent for Financial Trading
- **Authors:** Wentao Zhang et al.
- **ArXiv:** https://arxiv.org/abs/2402.18485 (2024)
- **Summary:** Multi-modal agent combining LLM reasoning, financial knowledge graphs, and RL trading decisions. Outperforms pure RL on diverse market conditions.
- **Why Critical:** Shows how to combine your ML ensemble + LLM sentiment + RL position sizing into a unified agent framework.

---

## 4. Academic Papers — Deep Learning & Time Series

### Temporal Fusion Transformer (TFT)
- **Title:** Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Authors:** Bryan Lim, Sercan Ö. Arık et al. (Google Cloud AI)
- **Venue:** International Journal of Forecasting, 2021
- **ArXiv:** https://arxiv.org/abs/1912.09363
- **Citations:** 3,000+
- **Summary:** Combines attention mechanisms with gating networks for interpretable multi-horizon forecasting. Handles static and time-varying inputs separately.
- **Implementation:** PyTorch Forecasting library. https://github.com/jdb78/pytorch-forecasting
- **Why Critical:** Best-in-class for financial time series with exogenous features (your 23 features fit naturally).

### N-BEATS (Neural Basis Expansion Analysis)
- **Title:** N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- **Authors:** Boris N. Oreshkin et al. (Element AI)
- **Venue:** ICLR 2020
- **ArXiv:** https://arxiv.org/abs/1905.10437
- **Citations:** 1,500+
- **Summary:** Pure DL approach that decomposes time series into trend and seasonality using basis functions. Beats all statistical models including ARIMA, ETS on M4 competition.
- **Implementation:** Darts library (`darts.models.NBEATSModel`)
- **Why Critical:** Excellent for price level forecasting. Can replace or supplement Kalman-smoothed price feature.

### N-HiTS (Hierarchical Interpolation)
- **Title:** N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
- **Authors:** Cristian Challu et al.
- **Venue:** AAAI 2023
- **ArXiv:** https://arxiv.org/abs/2201.12886
- **Summary:** Improves N-BEATS with hierarchical interpolation and multi-rate signal sampling. Better at long-horizon forecasting.
- **Implementation:** Darts library (`darts.models.NHiTSModel`)
- **Why Critical:** Better than N-BEATS for longer horizon price forecasts (e.g., next-day close prediction).

### Foundation Models for Time Series (2024)
- **TimeGPT-1** — Nixtla (2023): https://arxiv.org/abs/2310.03589 — Zero-shot forecasting using pretrained transformer.
- **Lag-Llama** — (2024): https://arxiv.org/abs/2310.08278 — Open-source foundation model for time series (univariate probabilistic)
- **Moirai** — Salesforce (2024): https://arxiv.org/abs/2402.02592 — Universal forecasting transformer trained on 27B time series observations
- **TimesFM** — Google (2024): https://arxiv.org/abs/2310.10688 — Pretrained on 100B time series points from Google
- **Why Critical:** These are the GPT-equivalent for time series. Use as feature extractors or zero-shot forecasters.

### LSTM for Financial Forecasting (Classical)
- **Title:** Long Short-Term Memory
- **Authors:** Sepp Hochreiter, Jürgen Schmidhuber
- **Venue:** Neural Computation, 1997
- **Link:** https://www.bioinf.jku.at/publications/older/2604.pdf
- **Citations:** 80,000+ (foundational)
- **Why Critical:** Still widely used in industry. Understand LSTM before moving to Transformers.

### Attention Is All You Need (Transformer Foundation)
- **Title:** Attention Is All You Need
- **Authors:** Vaswani et al. (Google Brain)
- **Venue:** NeurIPS 2017
- **ArXiv:** https://arxiv.org/abs/1706.03762
- **Citations:** 100,000+
- **Why Critical:** Foundation of TFT, GPT, and every modern financial model.

---

## 5. Academic Papers — NLP & LLMs for Finance

### FinBERT (Sentiment Classification)
- **Title:** FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
- **Authors:** Dogu Araci
- **ArXiv:** https://arxiv.org/abs/1908.10063
- **GitHub:** https://github.com/ProsusAI/finbert
- **Rating:** 8/10
- **Summary:** BERT fine-tuned on financial news corpus. 3-class sentiment (positive/negative/neutral) for financial text. 110M parameters, runs on CPU.
- **Why Critical:** Best for fast batch sentiment scoring of NSE company announcements. Use `ProsusAI/finbert` from HuggingFace.

### BloombergGPT (Finance Foundation Model) — ⚠️ NOT PUBLICLY ACCESSIBLE

- **Title:** BloombergGPT: A Large Language Model for Finance
- **Authors:** Shijie Wu et al. (Bloomberg)
- **ArXiv:** https://arxiv.org/abs/2303.17564
- **Rating:** 3/10 for practical use (hype without access)
- **Summary:** 50B parameter LLM trained on 363B tokens of financial text. Outperforms general LLMs on financial NLP tasks.
- **CRITICAL CAVEAT:** Bloomberg has NEVER released BloombergGPT publicly and has no announced plans to do so. No public access, no API pricing, no waitlist. Use FinBERT or FinGPT instead.
- **Practical Replacements:** FinBERT (8/10), FinGPT (8/10), FinLLaMA (8/10)

### FinGPT (Open-Source Finance LLM)
- **Title:** FinGPT: Open-Source Financial Large Language Models
- **Authors:** Hongyang Yang et al. (AI4Finance Foundation)
- **ArXiv:** https://arxiv.org/abs/2306.06031
- **GitHub:** https://github.com/AI4Finance-Foundation/FinGPT
- **Rating:** 8/10
- **Summary:** Democratizes financial LLMs by fine-tuning open-source models (LLaMA, ChatGLM) on financial data. Free to use and modify.
- **Why Critical:** Your daily sentiment LLM. Lighter than BloombergGPT, deployable locally.

### AutoGPT for Trading — ⚠️ OBSOLETE APPROACH

- **Rating:** 3/10
- **Status:** The 2023 AutoGPT trading bot craze is over. No independently verified profitable deployments exist.
- **Modern Replacement:** TradingAgents (multi-agent LLM framework, 7/10) or CrewAI for research automation (7/10). Use LLMs as research accelerators, not autonomous trade executors.

### Sentiment Analysis for Trading (Survey)
- **Key Finding:** Sentiment from news headlines predicts next-day returns with Spearman correlation 0.05–0.15 (small but real edge).
- **Why Critical:** Quantifies what to expect from sentiment feature. If correlation < 0.05 in your backtest, don't use it.

---

## 6. Academic Papers — Market Microstructure & Execution

### Almgren-Chriss Model (Optimal Execution)
- **Title:** Optimal Execution of Portfolio Transactions
- **Authors:** Robert Almgren, Neil Chriss
- **Venue:** Journal of Risk, 2001
- **Link:** https://cims.nyu.edu/~almgren/papers/optliq.pdf
- **Summary:** Framework for optimal order splitting to minimize market impact + opportunity cost.
- **Why Critical:** Explains why LIMIT orders reduce slippage. Foundation of execution algorithms used by all institutions.

### Order Book Imbalance (Feature #16)
- **Key Paper:** Cont, Kukanov, Stoikov (2014) — "The Price Impact of Order Book Events"
- **ArXiv:** https://arxiv.org/abs/1011.6402
- **Summary:** Order book imbalance (bid volume − ask volume) / (bid + ask) is the strongest short-horizon predictor of price direction. Signal decays within 1-10 seconds.
- **Why Critical:** Feature #16 in your system. Real edge at millisecond timescales, weaker at 15-min but still valuable.

### VWAP/TWAP Execution
- **Key Reference:** Konishi (2002), Cartea & Jaimungal papers
- **Summary:** VWAP spreads orders according to predicted volume profile. TWAP splits equally in time.
- **Why Critical:** Your execution uses LIMIT orders. Understanding VWAP helps optimize limit order placement.

### Avellaneda-Stoikov Market-Making Model
- **Title:** High-frequency trading in a limit order book
- **Authors:** Marco Avellaneda, Sasha Stoikov
- **Venue:** Quantitative Finance, 2008
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1100635
- **Summary:** Derives the optimal bid/ask quotes for a market-maker accounting for inventory risk and adverse selection. The reservation price (mid adjusted for inventory) and optimal spread formulas are the foundation of modern electronic market-making.
- **Why Critical:** Even for directional traders: explains how institutions set prices around you. Understanding A-S helps predict where limit orders will cluster and when spreads will widen. Deep coverage in CONVO_DEEP_RESEARCH.md Part 1.

### VPIN (Volume-Synchronized Probability of Informed Trading)
- **Title:** Flow Toxicity and Liquidity in a High-Frequency World
- **Authors:** Easley, López de Prado, O'Hara
- **Venue:** Review of Financial Studies, 2012
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
- **Summary:** VPIN measures order flow toxicity — the probability that a given volume bucket contains trades from informed traders. High VPIN → adverse selection risk is elevated → widen spreads or reduce size. Used as a leading indicator for volatility events (VPIN spiked before the 2010 Flash Crash).
- **Formula:** VPIN = |V_buy − V_sell| / V_total per volume bucket
- **Why Critical:** A microstructure-based signal that complements price-based features. Detects when "smart money" is aggressively taking liquidity. Deep coverage in CONVO_DEEP_RESEARCH.md Part 1.
- **Python:** Compute from tick data. Volume clock: divide trading day into equal-volume buckets (e.g., 50 buckets of V_total/50 each), compute VPIN per bucket.

### Hawkes Process (Self-Exciting Point Process for Order Flow)
- **Key Paper:** "Modelling microstructure noise with mutually exciting point processes" — Bacry, Mastromatteo, Muzy (2015) — ArXiv: https://arxiv.org/abs/1401.0903
- **Summary:** Order arrivals are not Poisson (random) — they cluster. A Hawkes process models this self-exciting behavior: each order arrival increases the probability of more orders arriving soon. Captures the "avalanche" structure of order flow.
- **Finance applications:** (1) Order flow intensity forecasting — predict bursts of activity before they happen. (2) Regime identification — Hawkes baseline intensity maps to calm vs active market regimes. (3) High-frequency alpha: if the current burst intensity is unusual vs historical, expect mean reversion.
- **Python:** `pip install tick` (Inria's point process library) — `tick.hawkes.HawkesExpKern` for exponential kernel Hawkes. Alternatively: `pip install hawkeslib`.
- **India Relevance:** NSE opening (9:15–9:45) and power hour (14:30–15:20) both exhibit strong Hawkes clustering. Deep coverage in CONVO_DEEP_RESEARCH.md Part 1.

### Volume Bars and Volume Clock
- **Key Reference:** López de Prado "Advances in Financial Machine Learning" Chapter 2; Easley et al. VPIN (above)
- **Summary:** Time bars (15-min candles) sample at fixed time intervals — this creates heteroskedastic features because more events happen in some bars than others. Volume bars sample every N shares traded instead. This produces bars with more uniform information content, reducing autocorrelation in returns and stabilizing ML feature distributions.
- **Volume Clock:** Divide the trading day into equal-volume buckets (e.g., 50 buckets of equal total volume). Each "tick" of the clock is one bucket. Trades at 9:15 and 14:30 are on the same timescale of information flow, not wall-clock time.
- **Why Critical:** If your ML model trains on time bars, features computed in low-volume periods (11:00–13:00) have systematically different statistical properties than high-volume bars. Volume bars fix this. Start with time bars (simpler), but consider volume bars once you have a working baseline. Deep coverage in CONVO_DEEP_RESEARCH.md Part 1.
- **Python:** Implement custom aggregation on tick data: accumulate ticks until volume threshold hit → close bar.

---

## 7. Academic Papers — Portfolio Theory & Risk

### Modern Portfolio Theory (Foundation)
- **Title:** Portfolio Selection
- **Author:** Harry Markowitz
- **Venue:** Journal of Finance, 1952
- **Summary:** Diversification reduces portfolio variance for a given return. Efficient frontier = maximum return for each risk level.
- **Why Critical:** Foundation of all position sizing. PyPortfolioOpt implements this.

### Hierarchical Risk Parity (HRP)
- **Title:** Building Diversified Portfolios that Outperform Out-of-Sample
- **Author:** Marcos López de Prado
- **Venue:** Journal of Portfolio Management, 2016
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
- **Summary:** Uses hierarchical clustering on correlation matrix instead of matrix inversion (which is unstable). Outperforms Markowitz on OOS data consistently.
- **Why Critical:** More robust than equal weighting or Markowitz for multi-stock portfolios.

### Black-Litterman Model
- **Title:** Global Portfolio Optimization
- **Authors:** Fischer Black, Robert Litterman (Goldman Sachs)
- **Venue:** Financial Analysts Journal, 1992
- **Summary:** Combines market equilibrium returns with investor views (your ML model predictions). Produces more diversified, stable portfolios.
- **Why Critical:** How to incorporate your ML ensemble's predictions into multi-stock portfolio allocation. PyPortfolioOpt implements this.

### Kelly Criterion for Trading
- **Title:** A New Interpretation of Information Rate
- **Author:** John L. Kelly Jr. (Bell Labs)
- **Venue:** Bell System Technical Journal, 1956
- **Summary:** Optimal bet size f* = (bp − q) / b maximizes long-run logarithmic growth. Full Kelly leads to large drawdowns; fractional Kelly (1/4 to 1/2) is standard.
- **Why Critical:** Mathematical foundation of your volatility-adjusted position sizing. ATR-based sizing approximates fractional Kelly.

### CVaR / Expected Shortfall
- **Title:** Coherent Measures of Risk
- **Authors:** Artzner, Delbaen, Eber, Heath
- **Venue:** Mathematical Finance, 1999
- **Summary:** VaR is not subadditive. CVaR (Expected Shortfall) is coherent and represents average loss beyond VaR threshold.
- **Why Critical:** Basel III replaced VaR with CVaR. Your drawdown kill switch is a practical CVaR-equivalent implementation.

---

## 8. Academic Papers — Factor Research & Alpha

### 101 Formulaic Alphas (WorldQuant)
- **Title:** 101 Formulaic Alphas
- **Authors:** Zura Kakushadze, Rachev, Fabozzi (WorldQuant)
- **ArXiv:** https://arxiv.org/abs/1601.00991
- **SSRN:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701346
- **Summary:** 101 real-world alpha formulas from WorldQuant's live trading. Average holding period 0.6–6.4 days. Average pairwise correlation only 15.9% → excellent diversification.
- **Why Critical:** Source of additional features beyond your 23.

### Taming the Factor Zoo
- **Title:** Taming the Factor Zoo: A Test of New Factors
- **Authors:** Guanhao Feng, Stefano Giglio, Dacheng Xiu (Chicago Booth)
- **NBER:** https://www.nber.org/system/files/working_papers/w25481/w25481.pdf
- **Summary:** Tests 150+ published factors. ~15 factors span the entire factor zoo. 65% of published anomalies fail replication tests.
- **Why Critical:** Validates choosing a focused set of 23 features rather than 100+. Quality > quantity.

### AQR Factor Research Papers
- **Source:** https://www.aqr.com/Insights/Research
- **Free Papers:**
  - "Fact, Fiction, and Factor Investing" (2023)
  - "Value and Momentum Everywhere" (Asness et al.)
  - "Momentum Crashes" (Daniel, Moskowitz)
- **Free Datasets:** https://www.aqr.com/Insights/Datasets
- **Why Critical:** AQR is the academic-practitioner bridge. Papers are rigorous, reproducible, and practical.

### Cross-Sectional Momentum (Indian Context)
- **Key Finding:** Momentum works in Indian markets but with shorter half-life than US markets.
- **Search:** "momentum NSE India" on Google Scholar or SSRN
- **Why Critical:** Momentum is feature #2 (MACD) and #11 (EMA cross) in your system.

---

## 9. Academic Papers — Indian Market Research

### SEBI Research Papers
- **Working Papers:** https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=21&smid=0
- **NSE Working Papers:** https://www.nseindia.com/static/research/working-papers
- **Key Topics Published:**
  - Impact of algorithmic trading on market liquidity
  - HFT and price discovery on NSE
  - F&O market dynamics and spot market impact

### IIMB Research on Algorithmic Trading
- **Project:** Characterizing Algorithmic Trading in Indian Stock Markets
- **Link:** https://www.iimb.ac.in/node/3752
- **Topics:** Informativeness of options OI/volume, VaR in Indian commodity markets

### Indian Market Seasonality
- **Paper:** Stock Market Seasonality: A Study of the Indian Stock Market
- **NSE Publication:** https://nsearchives.nseindia.com/content/research/res_paper_final228.pdf
- **Key Finding:** September returns significantly higher (monsoon effect); January negative (opposite of US January effect)

### F&O Impact on Spot Market
- **Reference:** Search "NSE colocation impact market quality" + "F&O expiry spot price impact India"

---

## 10. Must-Read Books

### Tier 1 — Essential (Read Before Writing Code)

| Book | Author | Year | Why Read It |
|------|--------|------|-------------|
| **Advances in Financial Machine Learning** | Marcos López de Prado | 2018 | THE bible. Triple-barrier labeling, purge/embargo, feature importance, fractional differentiation, CPCV. Read labeling chapter BEFORE coding anything |
| **Systematic Trading** | Robert Carver | 2015 | How real systematic traders build robust systems. Position sizing, diversification, cost modeling. Author was AHL portfolio manager |
| **Machine Learning for Algorithmic Trading (2nd Ed.)** | Stefan Jansen | 2020 | 822 pages of practical Python code. 23 chapters covering ML, DL, RL, alternative data, backtesting. GitHub: https://github.com/stefan-jansen/machine-learning-for-trading |
| **Quantitative Trading (2nd Ed.)** | Ernest P. Chan | 2021 | Retail-focused, realistic about what works. Updated with ML. Best starting book for solo traders |

### Tier 2 — Advanced

| Book | Author | Year | Why Read It |
|------|--------|------|-------------|
| **Inside the Black Box (3rd Ed.)** | Rishi K. Narang | 2013+ | Non-technical guide to institutional quant systems. Alpha Model + Risk Model + Transaction Cost Model architecture |
| **Algorithmic Trading: Winning Strategies and Their Rationale** | Ernest P. Chan | 2013 | Specific strategy implementations with code: mean reversion, momentum, pairs trading |
| **High-Frequency Trading: A Practical Guide (2nd Ed.)** | Irene Aldridge | 2013 | Market microstructure, event arbitrage, execution algorithms |
| **Active Portfolio Management** | Grinold & Kahn | 1999 | The Fundamental Law of Active Management. Information Ratio = IC × √BR. Foundation of alpha research |

### Tier 3 — Optional Deep Dives

| Book | Author | Why |
|------|--------|-----|
| **Python for Finance (2nd Ed.)** | Yves Hilpisch | Python patterns for quant finance |
| **Options, Futures, and Other Derivatives** | John C. Hull | F&O fundamentals for Indian F&O strategies |
| **The Man Who Solved the Market** | Gregory Zuckerman | How Renaissance's Medallion fund works (narrative) |

---

## 11. Institutional Research Sources

### Two Sigma
- **Venn Insights:** https://www.venn.twosigma.com/insights
- **Focus:** Factor analysis, quantitative research, risk decomposition
- **Team Scale:** 250+ PhDs, 4,800 person-years of R&D
- **Key Insight:** Two Sigma uses ML and AI as core trading infrastructure, not add-ons

### AQR Capital
- **Research Hub:** https://www.aqr.com/Insights/Research
- **White Papers:** https://www.aqr.com/Insights/Research/White-Papers
- **Free Datasets:** https://www.aqr.com/Insights/Datasets
- **Core Thesis:** Diversify across Value, Momentum, Carry, Defensive factors

### Man AHL (Quantitative Division of Man Group)
- **Founded:** 1987. 30+ years of systematic research
- **Team:** 150 investment professionals, 110 researchers
- **Philosophy:** Only trade ideas that can be tested and proven empirically
- **Publications:** https://oxford-man.ox.ac.uk/selected-publications/

### WorldQuant
- **Key Contribution:** 101 Formulaic Alphas (see Section 8)
- **Virtual Research Program:** Allows solo researchers to discover alphas
- **Platform:** WebSim (simulated trading environment)

### Renaissance Technologies (Reference Only)
- **Medallion Fund:** 66% annual return before fees (1988–2018)
- **Key Insight:** Pure statistical edge, not prediction. Pattern recognition across thousands of uncorrelated signals.

---

## 12. Indian Broker APIs (Full Comparison)

### OpenAlgo (Multi-Broker Bridge) — Recommended: 8/10

- **GitHub:** https://github.com/marketcalls/openalgo
- **Type:** Open-source broker API gateway / unified interface
- **Rating:** 8/10 | India Relevance: HIGH
- **Docs:** https://docs.openalgo.in/

**What OpenAlgo solves:** Normalizes all broker APIs (Zerodha, Upstox, Shoonya, Fyers, Angel One, Dhan, ICICI Breeze, Kotak Neo) into one standard interface.

**How it works:**
```
Your Strategy (Python/TradingView/AlgoTest)
    ↓ REST/Webhook HTTP call (standardized)
OpenAlgo Gateway (on your local machine or VPS)
    ↓ Translates to broker-specific API calls
Zerodha / Shoonya / Angel One / Fyers
    ↓ NSE/BSE Exchange
```

**Integration:** If your strategy uses the `BrokerBase` abstract interface from `data/broker_base.py`, add an OpenAlgo adapter — switch brokers without rewriting strategy.

### Tier 1 — Zero/Low Cost (Best for Algo)

#### Shoonya/Finvasia (PRIMARY BROKER — Use for All Live and Paper Trading)
- **API Cost:** Free | **Brokerage:** ₹0 (zero brokerage on equity delivery, F&O, and ETF)
- **Website:** https://shoonya.com/
- **API Docs:** https://shoonya.com/api-documentation
- **GitHub SDK:** https://github.com/Shoonya-Dev/ShoonyaApi-py
- **Auth:** TOTP 2FA required. Sessions last ~24 hrs and do NOT expire mid-day.
- **Best For:** All live execution. Zero brokerage + free API = lowest cost for high-frequency strategies. No TOTP session expiry mid-trading-day (unlike Angel One — see Critique 24).

#### Angel One SmartAPI (DEVELOPMENT REFERENCE ONLY — Do NOT use for live execution)
- **API Cost:** Free | **Brokerage:** ₹20/order
- **Docs:** https://smartapi.angelbroking.com/docs
- **Python SDK:** `pip install smartapi-python`
- **GitHub SDK:** https://github.com/angel-one/smartapi-python
- **WebSocket URL:** `wss://smartapisocket.angelone.in/smart-stream`
- **Limits:** 3 WebSocket connections max; 1000 token limit per socket; 20 orders/sec
- **Auth:** TOTP 2FA required
- **Markets:** NSE Equity, F&O, Currency, BSE, MCX
- **Best For:** Learning and development only — best documentation, largest community, easiest onboarding for understanding API patterns.
- **WARNING (Critique 24):** Angel One TOTP sessions expire mid-day. A live system will lose its connection during market hours and miss trades. **Do NOT use Angel One for live or paper trading execution.**
- **WARNING (Critique 10):** Angel One WebSocket data is throttled to 1–3 updates/second, which can cause missed price moves on fast market candles. Use Shoonya for all real-time data feeds.

#### Dhan (DhanHQ) (Best for Depth Data)
- **API Cost:** Free | **Brokerage:** ₹20/order
- **Docs:** https://dhanhq.co/docs/v2/
- **Python SDK:** https://github.com/dhan-oss/DhanHQ-py
- **Unique Feature:** 200-level depth directly on WebSocket; advanced option chain with Greeks
- **Best For:** Order book imbalance feature (Feature #16, #23)

#### Fyers (Flexible)
- **API Cost:** Free | **Brokerage:** ₹20/order
- **API Portal:** https://myapi.fyers.in/
- **Python SDK:** `pip install fyers-apiv3`
- **Best For:** Flexibility, active community

### Tier 2 — Established Platforms

#### Zerodha Kite Connect (Largest Ecosystem)
- **API Cost:** ₹500/month | **Brokerage:** ₹20/order
- **Docs:** https://kite.trade/
- **Python SDK:** https://github.com/zerodha/pykiteconnect
- **Unique:** Most mature ecosystem; 10 years of historical intraday data included
- **Best For:** Retail algo ecosystem; accessing existing community strategies

#### Upstox API v3
- **API Cost:** Free (basic) | **Docs:** https://upstox.com/developer/api-documentation/
- **Note:** v2 WebSocket deprecated Aug 22, 2025. Use v3 only.
- **Unique:** Up to 200-level order book depth

#### Kotak Neo
- **API Cost:** Free | **Brokerage:** ₹0 (Trade Free plan)
- **Docs:** https://www.kotakneo.com/platform/kotak-neo-trade-api/
- **GitHub:** https://github.com/Kotak-Neo/kotak-neo-api

#### ICICI Breeze
- **API Cost:** Free | **Docs:** https://api.icicidirect.com/breezeapi/documents/index.html
- **SDK:** `pip install breeze-connect` | **Limits:** 75 req/min, 5000 calls/day
- **Note:** Resident Indians only (no NRIs)

### Complete Broker Comparison Table

| Broker | API Cost | Brokerage | WebSocket | OPS | Depth | Best For |
|--------|----------|-----------|-----------|-----|-------|----------|
| Angel One | Free | ₹20 | Yes | 20/sec | Standard | Learning |
| Shoonya | Free | ₹0 | Yes | 10+/sec | Standard | Production |
| Dhan | Free | ₹20 | Yes | 10+/sec | 200-level | Depth data |
| Fyers | Free | ₹20 | Yes (v3) | 10+/sec | Standard | Community |
| Zerodha | ₹500/mo | ₹20 | Yes | 10+/sec | Standard | Ecosystem |
| Upstox | Free | ₹20 | Yes (v3) | 10+/sec | 200-level | Alternative |
| Kotak Neo | Free | ₹0 | REST | 10+/sec | Standard | Speed+cost |
| ICICI Breeze | Free | ~₹20 | Yes | 75 req/min | Standard | Established |

**Broker Selection Recommendation (Stage 1 — EMA Cross Paper Trading):**
- **PRIMARY:** Shoonya (Finvasia) — zero brokerage, no TOTP session expiry, free API. Use from Day 1 for all data streaming and order placement.
- **DEV REFERENCE:** Angel One SmartAPI — use only while building/debugging code against its documentation. Switch all live connections to Shoonya immediately.
- **BACKUP:** Dhan — best order-book depth (200 levels) if Feature #16 (order book imbalance) becomes important.
- **DO NOT** treat "Move to Shoonya later" as an upgrade path. Shoonya is the primary broker from the start.

---

## 13. Indian Market Data Sources

### Free Historical Data

| Source | Install | Coverage | Best For |
|--------|---------|----------|----------|
| **Jugaad Data** | `pip install jugaad-data` | NSE/BSE bhavcopy, F&O, economic | Primary historical source |
| **NSEPython** | `pip install nsepython` | Live quotes, option chains, FII/DII | Live market snapshot |
| **yfinance** | `pip install yfinance` | `RELIANCE.NS` format, daily OHLCV | Backtesting (daily data) |
| **AKShare** | `pip install akshare` | Multi-market including India | Broad data pipeline |
| **NSETools** | `pip install nsetools` | Quick quote retrieval | Simple prototyping |

**Note on yfinance intraday limits:** Free tier gives ~60 days of 15-min data. For 2 years, use broker historical API or TrueData.

### Professional Tick Data

| Provider | Cost | Coverage | Link |
|----------|------|----------|------|
| **TrueData** | ₹500-2,000/month | Full tick, zero data loss, Options Greeks | https://www.truedata.in/ |
| **Global Datafeeds** | ₹500-1,500/month | NSE/BSE/MCX live + historical | https://globaldatafeeds.in/ |
| **Twelve Data** | Free tier + paid | Global including NSE, 15-min intraday | API key required |

### Index & Reference Data

| Source | URL | What |
|--------|-----|------|
| Nifty 50 via Yahoo | `^NSEI` in yfinance | Index data for relative strength feature |
| India VIX | `^INDIAVIX` in yfinance | Volatility regime indicator |
| NSE FII/DII | https://www.nseindia.com/reports/fii-dii | Daily institutional flows (post-market) |
| NSDL FPI | https://www.fpi.nsdl.co.in/ | Final FII figures |

---

## 14. SEBI Regulations & Compliance (2025–2026)

### Primary Circulars

| Document | Date | Link |
|----------|------|------|
| SEBI Retail Algo Circular | Feb 4, 2025 | https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html |
| NSE Operational Modalities | Jul 22, 2025 | https://nsearchives.nseindia.com/content/circulars/INVG67858.pdf |
| SEBI Extension Timeline | Sep 30, 2025 | https://www.sebi.gov.in/legal/circulars/sep-2025/extension-of-timeline-for-implementation-of-sebi-circular-dated-february-04-2025-on-safer-participation-of-retail-investors-in-algorithmic-trading-_96979.html |

### Implementation Timeline

| Milestone | Date | Requirement |
|-----------|------|-------------|
| Phase 1 | Oct 1, 2025 | Broker infrastructure ready |
| Phase 2 | Oct 31, 2025 | Static IP registration enabled |
| Phase 3 | Nov 30, 2025 | Strategy approval process live |
| Phase 4 | Jan 3, 2026 | Algorithm categorization complete |
| **Full Mandate** | **Apr 1, 2026** | **All requirements mandatory** |

### Key Requirements (Mandatory Apr 2026)

- **Static IP:** Whitelist your static IPv4 with broker. IP changes max once/week.
- **Strategy ID:** All orders >10 OPS must carry exchange-assigned Strategy ID.
- **2FA:** Two-factor authentication on all API access.
- **OPS Limit:** Max 10 orders/second per exchange (your code limits to 8 as buffer).
- **Audit Trail:** 5-year minimum log of every order, trade, and model decision.
- **Infrastructure:** All algorithms must run on broker-approved Indian servers.
- **Daily Logout:** Automatic session logout before pre-open (9:00 AM).
- **Kill Switch:** Emergency stop mechanism mandatory.
- **SHAP Explainability:** SEBI requires explainability documentation for ML-based algo decisions.

### SEBI Compliance Checklist

```
[ ] Static public IPv4 from ISP or Indian VPS
[ ] Registered with broker (Shoonya — primary; Angel One for dev reference only)
[ ] 2FA enabled on API access
[ ] Daily session auto-logout before 9:00 AM
[ ] OPS monitoring code (max 8/sec, SEBI limit 10)
[ ] Strategy ID tagged on every order (>10 OPS)
[ ] 5-year audit trail in QuestDB/SQLite with all fields
[ ] All infrastructure on Indian servers (AWS Mumbai, DigitalOcean Bangalore)
[ ] Broker approval for API connectivity before go-live
[ ] Kill switch logic tested and confirmed working
[ ] SHAP explainability report generated after every training run
```

---

## 15. Indian Market Characteristics

### Trading Session Structure

| Time (IST) | Phase | Behavior |
|------------|-------|----------|
| 9:00–9:15 | Pre-open session | Order accumulation, price discovery |
| 9:15–9:45 | Opening auction spillover | High volatility, wide spreads |
| 9:45–11:00 | Morning session | Directional moves, institutional flow |
| 11:00–13:00 | Dead zone | Low volume, choppy, avoid scalping |
| 13:00–14:30 | Midday recovery | Gradual volume pickup |
| 14:30–15:20 | Power hour | Institutional rebalancing, high volume |
| 15:20–15:30 | Close alert | Reduce/exit all intraday positions |
| 15:30 | Market close | Closing auction |

### NSE Circuit Breakers

| Movement | Before 1 PM | 1 PM – 2:30 PM | After 2:30 PM |
|----------|-------------|-----------------|----------------|
| 10% | 45 min halt | 15 min halt | No halt |
| 15% | 1 hr 45 min halt | 45 min halt | Remainder of day |
| 20% | Full-day halt | Full-day halt | Full-day halt |

### F&O Expiry Calendar

| Type | Expiry Day | Impact |
|------|-----------|--------|
| Weekly (Nifty, BankNifty) | Every Wednesday | Faster theta decay; moderate volatility |
| Monthly (all F&O) | Last Thursday of month | Highest volatility day; reduce position size 50% |

**Settlement Price:** Average Nifty 50 from 3:00–3:30 PM IST on expiry day.

### India VIX Signal Interpretation

| India VIX Level | Market State | Action |
|----------------|--------------|--------|
| < 12 | Extreme complacency | Reduce size (volatility expansion likely) |
| 12–18 | Normal range | Normal position sizing |
| 18–24 | Elevated volatility | Reduce size by 25% |
| > 24 | Fear regime | Reduce size by 50%, tighten stops |

### Cost Structure (Full India Model)

| Component | Equity Intraday | Equity Delivery | F&O |
|-----------|----------------|-----------------|-----|
| Brokerage (Angel) | ₹20/order | ₹20/order | ₹20/order |
| Brokerage (Shoonya) | ₹0 | ₹0 | ₹0 |
| STT (new Apr 2026) | 0.025% sell | 0.1% both | 0.05% futures, 0.15% options |
| Exchange charges | ~0.00345% | ~0.00345% | ~0.05% |
| GST | 18% on brokerage+exchange | 18% | 18% |
| SEBI charges | 0.0001% | 0.0001% | 0.0001% |
| Stamp duty | ~0.003% buy | ~0.015% buy | Varies by state |
| **Slippage (model this!)** | **0.05–0.10% per side** | **0.03–0.05%** | **0.05–0.15%** |

---

## 16. Tax & Compliance for Algo Traders in India

### Income Classification

- **F&O Trading:** Non-speculative business income (Section 43(5) ITA 1961)
- **Tax Rate:** Your income tax slab (NOT capital gains rate)
- **ITR Form:** ITR-3 mandatory for F&O traders
- **Reference:** https://www.sahi.com/blogs/income-tax-on-f-and-o-trading-in-india-complete-guide-for-fy-2025-26

### Audit Requirements (FY 2025–26)

Audit required if ANY of:
- F&O turnover > ₹10 crore (≥95% digital transactions)
- F&O turnover > ₹1 crore (<95% digital transactions)
- Profit < 6% of turnover AND income exceeds exemption limit

### Loss Carry Forward Rules

- **Period:** 8 years
- **Critical:** Must file ITR-3 BEFORE due date. Late filing forfeits carry-forward permanently.

### Deductible Expenses

- STT (100% deductible), GST on brokerage (100% deductible), Exchange charges, Brokerage fees, Internet/VPS costs

---

## 17. Free Learning Resources

### Indian Market Education

| Resource | URL | What |
|----------|-----|------|
| **Zerodha Varsity** | https://zerodha.com/varsity/ | Free, comprehensive, stock market fundamentals to F&O |
| **NSE Academy** | https://www.nseacademy.com/ | NCFM modules including algo trading (₹7,670 for exam) |
| **NSE Knowledge Hub** | https://nseknowledgehub.edcastcloud.com/ | Free algo trading open course |
| **QuantInsti Blog** | https://blog.quantinsti.com/ | Algo trading tutorials, Python code |

### Global Platforms

| Resource | URL | What |
|----------|-----|------|
| **QuantConnect** | https://www.quantconnect.com/ | Cloud + local backtesting. **483,000+ users** (updated 2025), 100+ hedge fund clients. Mia V2 AI writes LEAN algorithm code from natural language. Free tier available. Rating: 9/10 |
| **QuantLib** | https://www.quantlib.org/ | Open-source quant finance C++ library |

### Coursera/Online Courses

| Course | Platform | Link |
|--------|----------|------|
| Machine Learning for Trading Specialization | Coursera (Google + NYIF) | https://www.coursera.org/specializations/machine-learning-trading |
| ML & RL in Finance Specialization | Coursera (NYU) | https://www.coursera.org/specializations/machine-learning-reinforcement-finance |

### YouTube Channels

| Channel | URL | What |
|---------|-----|------|
| **QuantInsti** | https://www.youtube.com/channel/UC8kXgHG13XdgsigIPRmrIyA | Python + ML + HFT tutorials |
| **Varsity by Zerodha** | https://www.youtube.com/@varsitybyzerodha | Indian market fundamentals |

---

## 18. Institutional Quant Strategies

### Statistical Arbitrage (Pairs Trading)

- **Core:** Find two cointegrated assets. Long the underperformer, short the outperformer when spread deviates >2σ.
- **Library:** statsmodels Engle-Granger + Johansen cointegration tests
- **Reference:** https://hudsonthames.org/an-introduction-to-cointegration/
- **Stefan Jansen Code:** https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/09_time_series_models/06_statistical_arbitrage_with_cointegrated_pairs.ipynb

### Mean Reversion

- **Core:** Price deviates from mean → reverts. Calculate Z-score; enter at ±2.0, exit at ±0.5.
- **Best on:** Very short (intraday scalp) or very long (monthly) timeframes
- **Library:** Custom Z-score on rolling mean + Kalman filter for adaptive mean

### Momentum / Trend Following

- **Core:** 3–12 month return momentum predicts future returns
- **Indian market:** Works but with shorter half-life than US
- **Library:** pandas-ta EMA cross, MACD, ADX for trend strength
- **Reference:** Robert Carver's "Systematic Trading" (Chapter 3 onwards)

### Market Making (Retail Concepts)

- **Institutional HFT:** Requires colocation (~₹8,000+/month at NSE)
- **Retail Application:** Use limit orders to capture bid-ask spread on less-liquid stocks
- **Minimum edge needed:** Bid-ask spread > 2× slippage + brokerage

### Factor Model Alphas

- **Value:** P/E, P/B, P/FCF ratios relative to sector
- **Momentum:** Past 6–12 month total return
- **Quality:** ROE, debt/equity, earnings stability
- **Low Volatility:** 30/60-day realized volatility (paradoxically, lower vol = higher returns)
- **Size:** Small cap premium (risky in India due to liquidity)

---

## 19. Portfolio Optimization Research

### Hierarchical Risk Parity (HRP) — Recommended

- **Paper:** López de Prado (2016) https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
- **Library:** PyPortfolioOpt `HRPOpt` class
- **Advantage over Markowitz:** No matrix inversion → stable with many assets
- **When to use:** Portfolio of 5–20 stocks with ML signals

### Black-Litterman with ML Views

- **Framework:** Use ML ensemble probability as "investor view" in Black-Litterman
- **Library:** PyPortfolioOpt `BlackLittermanModel`
- **How:** If XGBoost says 75% probability of +1 for RELIANCE → that's your "view" with confidence

### CVaR Portfolio Optimization

- **Library:** Riskfolio-Lib https://github.com/dcajasn/Riskfolio-Lib
- **Use case:** Minimize conditional tail loss instead of variance

### Change Point Detection (Regime Switching)

- **Library:** ruptures https://github.com/deepcharles/ruptures
- **Use for:** Detecting when market regime changed (invalidates old model)
- **Combined with HMM:** ruptures for offline detection, HMM for online real-time detection

---

## 20. Recommended Implementation Path

### Phase 1: Foundation (Weeks 1–3)
```
✓ Broker account: Shoonya/Finvasia (PRIMARY — zero brokerage, no TOTP expiry)
✓ Static IP: Get from ISP or AWS Mumbai (ap-south-1)
✓ Register IP with Shoonya
✓ Download 2yr historical data: jugaad-data + Shoonya historical API
✓ Build candle_store.py (QuestDB OHLCV — docker run -d -p 9000:9000 questdb/questdb)
✓ Build websocket_feed.py (live 15-min candles via Shoonya NorenAPI WebSocket)
✓ Compute 23 features (pandas-ta + statsmodels + hmmlearn + filterpy + arch)
[Angel One SmartAPI: set up account for dev/code-reference purposes only — do NOT use for live data or orders]
```

### Phase 2: Research & Labeling (Weeks 4–7)
```
✓ Implement triple-barrier labeling (López de Prado)
✓ Implement walk-forward with purge + embargo
✓ Train XGBoost + LightGBM + CatBoost ensemble
✓ Tune with Optuna (300 trials per model)
✓ Track with MLflow (every experiment logged)
✓ GATE: Walk-forward Sharpe > 1.0 before proceeding
```

### Phase 3: Risk & Execution (Weeks 8–10)
```
✓ Volatility-adjusted position sizing (ATR-based, approximate Kelly)
✓ Full India cost model (STT + GST + exchange + slippage)
✓ Risk manager kill switches (2% daily loss, 10% drawdown)
✓ LIMIT order logic with 2-candle fallback to MARKET
✓ QuestDB trade logger (all fields from Section 10 of Blueprint)
✓ Paper trading for 4+ weeks minimum
```

### Phase 4: Enhancements (Weeks 11+)
```
✓ HMM regime filter (block trades in sideways market)
✓ RL position sizing (Stable-Baselines3 PPO) via FinRL patterns
✓ LLM sentiment (FinGPT or FinBERT — daily, not per candle)
✓ Telegram bot for trade/error/P&L alerts
✓ Shoonya is already primary — confirm zero-brokerage is active and all live orders route through Shoonya only
✓ Weekly model retraining pipeline
```

### Gate Criteria (Do NOT skip)

| Gate | Metric | Threshold |
|------|--------|-----------|
| Before tuning | Walk-forward Sharpe | > 1.0 |
| Before paper trading | Walk-forward Sharpe (tuned) | > 1.5 |
| Before live trading | Paper trading P&L (4+ weeks) | Consistently profitable |
| Live position size cap | Max per trade | 5% of capital |
| Daily kill switch | Daily loss | 2% of capital |
| System kill switch | Total drawdown from peak | 10% |

---

## Quick Reference: Critical URLs

### Regulations
- SEBI Feb 2025: https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html
- NSE Circular: https://nsearchives.nseindia.com/content/circulars/INVG67858.pdf

### Broker APIs
- Angel One: https://smartapi.angelbroking.com/docs
- Shoonya: https://shoonya.com/api-documentation
- Dhan: https://dhanhq.co/docs/v2/
- Zerodha: https://kite.trade/
- Fyers: https://myapi.fyers.in/

### Data
- NSE FII/DII: https://www.nseindia.com/reports/fii-dii
- TrueData: https://www.truedata.in/
- Jugaad Data: https://github.com/jugaad-py/jugaad-data

### Papers (Free)
- XGBoost: https://arxiv.org/abs/1603.02754
- LightGBM: https://arxiv.org/abs/1706.01427
- FinRL: https://arxiv.org/abs/2011.09607
- 101 Alphas: https://arxiv.org/abs/1601.00991
- TFT: https://arxiv.org/abs/1912.09363
- López de Prado papers: https://mlfortraders.com

---

## 21. Advanced Transformer & Foundation Models

### Why Transformers for Finance

Financial time series has long-range dependencies (trends spanning hundreds of bars), non-stationarity, and heterogeneous inputs (price, volume, fundamentals). Vanilla LSTMs handle these poorly. Transformer attention mechanisms learn arbitrary-length dependencies and can integrate multi-modal inputs.

### Informer (Long Sequence Transformer)
- **Title:** Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **Authors:** Haoyi Zhou et al.
- **Venue:** AAAI 2021 Best Paper
- **ArXiv:** https://arxiv.org/abs/2012.07436
- **Citations:** 5,000+
- **Summary:** ProbSparse self-attention reduces O(L²) to O(L log L). Distilling encoder halves sequence length each layer.
- **Why Critical:** Baseline for forecasting 100+ bars ahead. Most attention heads are negligible — ProbSparse prunes them.

### Autoformer (Decomposition + Correlation)
- **Title:** Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- **Authors:** Haixu Wu et al. (Tsinghua)
- **Venue:** NeurIPS 2021
- **ArXiv:** https://arxiv.org/abs/2106.13008
- **Summary:** Decomposes series into trend-cyclical + seasonal components. Auto-Correlation replaces self-attention.
- **Why Critical:** Trend/seasonal decomposition maps to financial regime detection.

### PatchTST (Patch-based Transformer)
- **Title:** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Authors:** Yuqi Nie, Nam H. Nguyen et al. (Princeton)
- **Venue:** ICLR 2023
- **ArXiv:** https://arxiv.org/abs/2211.14730
- **GitHub:** https://github.com/yuqinie98/PatchTST
- **Summary:** Divides time series into subseries patches. Patch-level attention reduces sequence length significantly. SOTA on long-horizon benchmarks.
- **Why Critical:** Treat each trading hour (4 candles) as one patch. Self-supervised pretraining on unlabeled price data before fine-tuning.

### iTransformer (Inverted Transformer)
- **Title:** iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **Authors:** Yong Liu et al. (Tsinghua)
- **Venue:** ICLR 2024
- **ArXiv:** https://arxiv.org/abs/2310.06625
- **GitHub:** https://github.com/thuml/iTransformer
- **Summary:** Applies attention on variate dimension (features) instead of time dimension. Tokens represent different variates → captures multivariate correlations naturally.
- **Why Critical:** Your 23 features are multi-variate. iTransformer learns cross-feature correlations. State-of-the-art on LTSF-Linear benchmark as of 2024.

### TimesNet (2D Temporal Variation)
- **Title:** TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
- **Authors:** Haixu Wu et al. (Tsinghua)
- **Venue:** ICLR 2023
- **ArXiv:** https://arxiv.org/abs/2210.02186
- **GitHub:** https://github.com/thuml/TimesNet
- **Summary:** Transforms 1D time series into 2D space by folding on dominant periods (FFT-detected). CNNs capture 2D intraperiod and interperiod variations simultaneously.
- **Why Critical:** NSE has intraday, daily, weekly, and monthly cycles all simultaneously.

### MOMENT (Foundation Model — Open Source)
- **Title:** MOMENT: A Family of Open Time-Series Foundation Models
- **Authors:** Mononito Goswami et al. (CMU)
- **ArXiv:** https://arxiv.org/abs/2402.03885
- **GitHub:** https://github.com/moment-research/MOMENT
- **HuggingFace:** https://huggingface.co/AutonLab/MOMENT-1-large
- **Summary:** Open-source foundation model. Supports classification, imputation, anomaly detection, and forecasting. MIT license.
- **Why Critical:** Truly open. Use as feature extractor: freeze MOMENT encoder, fine-tune head on Indian stock data.

### TimeGPT-1 (Zero-Shot Forecasting API)
- **ArXiv:** https://arxiv.org/abs/2310.03589
- **API:** https://docs.nixtla.io/
- **Python:** `pip install nixtla`
- **Summary:** Zero-shot forecasting without fine-tuning. Trained on 100B+ time series data points.
- **Why Critical:** Zero-shot benchmark — compare your trained XGBoost/LightGBM against TimeGPT zero-shot.

### Lag-Llama (Open Probabilistic Foundation)
- **ArXiv:** https://arxiv.org/abs/2310.08278
- **GitHub:** https://github.com/time-series-foundation-models/lag-llama
- **Summary:** Decoder-only LLM architecture for univariate probabilistic forecasting. Generates prediction intervals (10th, 50th, 90th percentile).
- **Why Critical:** Use the width of these intervals as a dynamic volatility feature replacing or supplementing GARCH.

### Moirai (Salesforce Universal Forecaster)
- **Title:** Unified Training of Universal Time Series Forecasting Transformers
- **Authors:** Gerald Woo et al. (Salesforce Research)
- **ArXiv:** https://arxiv.org/abs/2402.02592
- **GitHub:** https://github.com/SalesforceAIResearch/uni2ts
- **Summary:** Trained on LOTSA dataset (27B observations across 9 domains). Multi-variate zero-shot.
- **Why Critical:** Feed in all 23 features simultaneously and get multi-step forecasts. No Indian training data needed.

### TimesFM (Google Research)
- **Title:** A Decoder-Only Foundation Model for Time-Series Forecasting
- **Authors:** Abhimanyu Das et al. (Google Research)
- **ArXiv:** https://arxiv.org/abs/2310.10688
- **GitHub:** https://github.com/google-research/timesfm
- **Summary:** 200M parameter decoder-only model pretrained on 100B time series points. Outperforms statistical models zero-shot.

### Mamba (State Space Models / SSMs) — HIGH Priority for 2025–2026

- **Key Paper:** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — Gu & Dao (2023) — ArXiv: https://arxiv.org/abs/2312.00752
- **GitHub:** https://github.com/state-spaces/mamba
- **Install:** `pip install mamba-ssm` (requires CUDA for fast training; CPU: `mamba-minimal`)
- **Rating:** Maturity 6/10 | Impact 3Y: 7/10 | Solo Actionability: 7/10 | **HIGH Priority**

**What Mamba does differently:** Standard Transformer attention is O(L²). Mamba uses Selective State Space Models that process sequences in O(L) linear time. Mamba **selects** which parts of the sequence to remember vs. forget based on the input — unlike LSTM which has fixed forget gates.

**Finance-specific results (2024–2025):**
- **CrossMamba:** R² up to 0.963 on 5-day S&P 500 forecasting
- **CMDMamba (2025):** Outperformed Reformer and Transformer on 4 financial datasets
- Hybrid Mamba-Transformer approaches dominate pure variants on high-frequency data

```python
# pip install mamba-ssm causal-conv1d>=1.4.0
from mamba_ssm import Mamba
import torch

mamba_layer = Mamba(
    d_model=64,      # dimension of your feature vector
    d_state=16,      # SSM state expansion factor
    d_conv=4,        # local convolution width
    expand=2,        # block expansion factor
)
# Input: (batch, seq_len, d_model)
features = torch.randn(32, 256, 64)   # (batch=32, seq_len=256 bars, features=64)
output = mamba_layer(features)         # same shape, richer representation
```

### KAN Networks (Kolmogorov-Arnold Networks) — HIGH Priority

- **Key Paper:** "KAN: Kolmogorov-Arnold Networks" — Liu et al. (MIT, 2024) — ArXiv: https://arxiv.org/abs/2404.19756 | **Venue: ICLR 2025**
- **GitHub:** https://github.com/KindXiaoming/pykan
- **Install:** `pip install pykan`
- **Rating:** Maturity 5/10 | Impact 3Y: 8/10 | Solo Actionability: 7/10 | **HIGH Priority**

**What KANs do differently from MLPs:** Standard MLPs have fixed activation functions on nodes and learnable weights on edges. KANs invert this: **learnable activation functions on edges** (B-splines), fixed summation on nodes. Each connection's function is visualizable — you can see what relationship the network learned.

**Finance-specific results (2024–2025):**
- **KASPER framework:** R² = 0.89, Sharpe = 12.02 on Yahoo Finance data with KAN + regime detection
- **LSTM-KAN hybrids:** Significantly reduced prediction errors vs LSTM alone on NIFTY 50 data
- **Physics-informed KAN:** Improved cumulative returns, Sharpe, and drawdown vs standard neural networks

**Why especially important for this NSE project:**
1. **Interpretability:** SEBI audit trail requirements → KANs can show WHY each feature contributes
2. **Symbolic extraction:** Distill a trained KAN into a closed-form formula → discover pricing relationships
3. **Smaller models = less overfitting:** On NSE data with limited history (~10 years), smaller models generalize better

```python
# pip install pykan
import torch
from kan import KAN

model = KAN(
    width=[23, 64, 32, 3],    # 23 input features → hidden → 3 output classes
    grid=5,                    # B-spline grid points
    k=3,                       # B-spline polynomial degree
    seed=42
)
# Visualization — see what each connection learned
model.plot()  # shows learned activation functions per edge
# Symbolic regression — extract formula
model.auto_symbolic()  # identifies known functions (sin, x², log, etc.)
```

### Implementation Recommendations

```python
# Priority order for trying transformers:
# 1. PatchTST (patched input, best balance of speed/accuracy)
# 2. iTransformer (inverted, best for multivariate)
# 3. TimesNet (2D, captures multiple periodicities well)
# 4. Use Darts or HuggingFace for unified API

pip install darts          # PatchTST, NHiTS, TFT all available
pip install momentfm       # MOMENT foundation model
pip install nixtla         # TimeGPT API
pip install mamba-ssm      # Mamba SSM (CUDA required)
pip install pykan          # KAN networks
```

### Transformer Comparison for Financial Time Series
- **Title:** Are Transformers Effective for Time Series Forecasting?
- **Authors:** Ailing Zeng et al. (MSRA)
- **Venue:** AAAI 2023
- **ArXiv:** https://arxiv.org/abs/2205.13504
- **Key Finding:** Simple DLinear (linear model on decomposed series) outperforms complex Transformers on many benchmarks.
- **Why Critical:** Ground truth check — before deploying any Transformer, compare to DLinear baseline.

---

## 22. Advanced Reinforcement Learning — 100+ Papers

### Core RL Algorithms (Used in FinRL)

| Algorithm | Paper | ArXiv | Best For |
|-----------|-------|-------|----------|
| **PPO** | Schulman et al. (OpenAI 2017) | https://arxiv.org/abs/1707.06347 | Position sizing (stable, easy to tune) |
| **SAC** | Haarnoja et al. (Berkeley 2018) | https://arxiv.org/abs/1801.01290 | Continuous actions (better exploration than PPO) |
| **TD3** | Fujimoto et al. (McGill 2018) | https://arxiv.org/abs/1802.09477 | Deterministic policy, less hyperparameter-sensitive |
| **A2C** | Mnih et al. (DeepMind 2016) | https://arxiv.org/abs/1602.01783 | Fast, simple baseline |
| **DDPG** | Lillicrap et al. (DeepMind 2015) | https://arxiv.org/abs/1509.02971 | Original continuous action RL for trading |
| **DQN** | Mnih et al. (DeepMind 2015) | https://www.nature.com/articles/nature14236 | Discrete action spaces |
| **Rainbow DQN** | Hessel et al. (DeepMind 2017) | https://arxiv.org/abs/1710.02298 | Best DQN variant (combines 6 improvements) |
| **Dreamer v3** | Hafner et al. (Google 2023) | https://arxiv.org/abs/2301.04104 | World model RL — learns market dynamics |

### FinRL Ecosystem

#### Core Papers
- **FinRL** (NeurIPS 2020): https://arxiv.org/abs/2011.09607 — Original paper, DQN/DDPG/PPO/SAC/A2C/TD3
- **FinRL-Meta** (2022): https://arxiv.org/abs/2211.03107 — Market environments, standardized benchmarks
- **FinRL-Trading** (2023): https://github.com/AI4Finance-Foundation/FinRL-Trading — Production-ready with LLM integration
- **ElegantRL** (2021): https://github.com/AI4Finance-Foundation/ElegantRL — Lightweight RL backbone for FinRL

#### FinRL Reported Results — ⚠️ Treat With Caution

- SPY (S&P 500) buy-and-hold: Sharpe 0.79
- FinRL PPO on DJIA 30: Sharpe 1.27 (60% improvement vs buy-and-hold)
- FinRL ensemble (A2C+PPO+DDPG): Sharpe 1.46

**Caveat:** These results use a specific test period (2019–2020) favorable to adaptive agents. Survivorship bias in DJIA stock selection. Results have NOT been independently replicated across longer periods or different markets. Do not use as benchmarks for NSE data.

### Offline RL (Learn from Historical Data — No Exploration)

**Key concept:** In finance, we can't run trial-and-error experiments on live markets. Offline RL trains entirely from historical data (the replay buffer = your backtest database).

| Algorithm | Paper | ArXiv | Key Idea |
|-----------|-------|-------|----------|
| **Decision Transformer** | Chen et al. (Berkeley 2021) | https://arxiv.org/abs/2106.01345 | Reframes RL as sequence modeling — GPT-style. Input: (return, state, action) → output: next action. |
| **CQL (Conservative Q-Learning)** | Kumar et al. (Berkeley 2020) | https://arxiv.org/abs/2006.04779 | Penalizes out-of-distribution actions |
| **IQL (Implicit Q-Learning)** | Kostrikov et al. 2021 | https://arxiv.org/abs/2110.06169 | Avoids querying unseen actions entirely. Most stable offline RL |
| **BCQ** | Fujimoto et al. 2019 | https://arxiv.org/abs/1812.02900 | Constrains policy near behavioral policy in training data |
| **TD3+BC** | Fujimoto & Gu 2021 | https://arxiv.org/abs/2106.06860 | Simplest offline RL baseline — TD3 + behavioral cloning |
| **BEAR** | Kumar et al. 2019 | https://arxiv.org/abs/1906.00949 | Bootstrap Error Accumulation Reduction |

**Why Critical:** Your 2yr historical dataset IS an offline RL dataset. Decision Transformer + IQL are most practical.

### Reward Engineering for Finance

| Approach | Key Insight |
|----------|------------|
| **Sharpe Ratio Reward** | Direct Sharpe optimization is more stable than P&L reward |
| **Differential Sharpe** | Incremental Sharpe computation: numerically stable |
| **Calmar Ratio Reward** | Reward = P&L / max_drawdown — optimizes both simultaneously |
| **Risk-Adjusted Reward** | (daily_return - 0.5 × volatility) — explicit risk aversion |
| **Transaction-Aware Reward** | Penalize excessive trading to reduce costs |

```python
def reward_function(portfolio_value, prev_value, transaction_cost, daily_vol):
    daily_return = (portfolio_value - prev_value) / prev_value
    reward = daily_return - transaction_cost - 0.5 * daily_vol
    return reward
```

### 2025 SOTA RL Papers for Trading

| Paper | ArXiv/Source | Key Innovation |
|-------|-------------|----------------|
| **Trading-R1** | ArXiv 2509.11420 | R1-style chain-of-thought RL for trading. Outperforms direct RL by 15–30% on multiple benchmarks. |
| **FinRLlama** (2025) | Search "FinRLlama 2025 ArXiv" | LLaMA fine-tuned with RL reward from trading environment |
| **AlphaAgent** (2025) | Search "AlphaAgent LLM trading 2025" | LLM agent discovering and testing alphas autonomously |
| **StockAgent** (2024) | https://arxiv.org/abs/2402.09314 | LLM agent with memory, reflection, and tool use |

### Continual Learning / EWC (Preventing Catastrophic Forgetting)

**Why critical for India:** NSE regimes shift — a model trained in 2022 (post-COVID recovery) may forget 2020 patterns. Continual Learning methods prevent catastrophic forgetting when retraining.

| Method | Paper | Key Idea |
|--------|-------|---------|
| **EWC (Elastic Weight Consolidation)** | Kirkpatrick et al. (DeepMind 2017) | https://arxiv.org/abs/1612.00796 — Quadratic penalty on weights important for past tasks |
| **Progressive Neural Networks** | Rusu et al. (DeepMind 2016) | https://arxiv.org/abs/1606.04671 — Lateral connections to old task networks |
| **Online EWC** | Schwarz et al. 2018 | https://arxiv.org/abs/1805.06370 — Online version of EWC, practical for streaming data |
| **PackNet** | Mallya & Lazebnik 2018 | https://arxiv.org/abs/1711.05769 — Binary masks to protect important weights |

```python
# EWC implementation for NSE model retraining
# After initial training, compute Fisher information matrix
# During retraining, add EWC penalty to prevent forgetting

from torch import nn
import torch

class EWCTrainer:
    def __init__(self, model, fisher_importance=1000):
        self.model = model
        self.importance = fisher_importance
        self.params_old = {}
        self.fisher = {}
    
    def compute_fisher(self, dataloader):
        """Compute Fisher information matrix after training on old data"""
        self.model.eval()
        for name, param in self.model.named_parameters():
            self.params_old[name] = param.data.clone()
            self.fisher[name] = torch.zeros_like(param)
        
        for X, y in dataloader:
            self.model.zero_grad()
            output = self.model(X)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
    
    def ewc_penalty(self):
        """EWC regularization term to add to loss"""
        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                penalty += (self.fisher[name] * 
                           (param - self.params_old[name]) ** 2).sum()
        return self.importance / 2 * penalty
```

---

## 23. Options Trading Research (India F&O)

### Volatility Modeling

#### Black-Scholes (Foundation)
- **Title:** The Pricing of Options and Corporate Liabilities
- **Authors:** Fischer Black, Myron Scholes
- **Venue:** Journal of Political Economy, 1973
- **Why Critical:** All options pricing starts here. Nifty PCR, India VIX, and Greeks are all Black-Scholes-derived.
- **Python:** `from py_vollib.black_scholes import black_scholes` or `import mibian`

#### Heston Stochastic Volatility Model
- **Author:** Steven L. Heston
- **Venue:** Review of Financial Studies, 1993
- **Python:** `import QuantLib as ql` — `ql.HestonModel`
- **Key Parameters:** κ (mean reversion speed), θ (long-run variance), σ (vol of vol), ρ (price-vol correlation), v₀ (initial variance)
- **Why Critical:** India VIX smiles are non-flat → Black-Scholes misprices options. Heston captures vol smile.

### NSE/Nifty Options Research

#### Nifty PCR (Put-Call Ratio) as Signal

| PCR Level | Market Signal | Action |
|-----------|--------------|--------|
| PCR > 1.5 | Extreme bearish sentiment (contrarian bullish) | Consider long calls or close shorts |
| PCR 1.2–1.5 | Bearish to neutral | Reduce long exposure |
| PCR 0.8–1.2 | Neutral/balanced | Normal trading |
| PCR 0.6–0.8 | Bullish to neutral | Reduce short exposure |
| PCR < 0.6 | Extreme bullish sentiment (contrarian bearish) | Consider long puts or close longs |

**Note:** PCR works as a CONTRARIAN indicator at extremes.

#### Max Pain Theory (Options Pinning)
- **Concept:** Max Pain = strike at which total option buyers (both put + call) lose the most money.
- **Evidence:** NSE Nifty weekly expiry exhibits ~60% probability of closing within ±0.5% of Max Pain on expiry Thursday.

### Options Strategy Research

#### Iron Condor (Premium Selling)
- **Strategy:** Sell OTM put + buy further OTM put + sell OTM call + buy further OTM call
- **Best For:** Low VIX environments (India VIX < 15) — collect premium when market stays range-bound
- **Library:** `pip install optionlab` — implements iron condor P&L and Greeks

#### Calendar Spread (Volatility Term Structure)
- **Strategy:** Sell near-month option, buy far-month option at same strike
- **Best For:** When IV term structure is inverted (near-term IV > far-term IV) — normal after event risk
- **India Specific:** Week of monthly expiry, near-term IV spikes → calendar spread profitable

### Greeks Calculation Libraries

| Library | Install | Strengths | Speed |
|---------|---------|-----------|-------|
| **py_vollib** | `pip install py_vollib` | Full Greeks (delta, gamma, theta, vega, rho), IV solver | Fast |
| **py_vollib_vectorized** | `pip install py_vollib_vectorized` | NumPy-vectorized, 1000x faster for batch calcs | Very Fast |
| **mibian** | `pip install mibian` | Simple, clean API, B-S + Merton + B76 | Fast |
| **QuantLib** | `pip install QuantLib` | Full Heston/SABR/Local Vol, most accurate | Slow |
| **optionlab** | `pip install optionlab` | Strategy P&L diagrams, multi-leg strategies | Medium |

---

## 24. Execution Algorithms & Transaction Cost Analysis

### Classic Execution Algorithms

#### VWAP (Volume-Weighted Average Price)

| Time (IST) | % of Day's Volume | VWAP Weight |
|------------|-------------------|-------------|
| 9:15–9:45 | 18–22% | Heavy opening |
| 9:45–11:30 | 15–18% | Declining |
| 11:30–13:00 | 8–10% | Lightest period |
| 13:00–14:30 | 10–12% | Pickup |
| 14:30–15:20 | 28–35% | Heaviest — institutions |
| 15:20–15:30 | 3–5% | Closing |

```python
def vwap_schedule(total_qty, volume_profile_pct, candle_bars=26):
    schedule = np.round(total_qty * volume_profile_pct).astype(int)
    schedule[-1] += total_qty - schedule.sum()  # fix rounding
    return schedule
```

#### TWAP (Time-Weighted Average Price)
- **Formula:** qty_per_candle = total_qty / n_candles
- **When to use:** Small orders, illiquid stocks where volume profile is unstable

#### Implementation Shortfall (IS) / Arrival Price
- **Decomposition:**
  1. **Delay cost:** Market moves while you're deciding
  2. **Market impact:** Your order moves the price against you
  3. **Opportunity cost:** Unexecuted portion missed a favorable move
  4. **Timing risk:** Variance of execution price around VWAP

### Almgren-Chriss Optimal Execution Model

- **URL:** https://cims.nyu.edu/~almgren/papers/optliq.pdf
- **Key Insight:** Trade-off between market impact (execute too fast = move price against yourself) and timing risk (execute too slow = price moves away)

**Model Parameters:**
```
η = temporary market impact coefficient
γ = permanent market impact coefficient
σ = asset volatility (use ATR)
λ = risk aversion (how much variance you'll tolerate)
T = time to complete execution (e.g., 26 candles = 6.5 hours)
X = total quantity to execute
```

**Practical NSE Parameters:**
- For Nifty 50 stocks with ADV > ₹100Cr: η ≈ 0.05–0.15, γ ≈ 0.01–0.05
- For midcap stocks: η ≈ 0.2–0.5, γ ≈ 0.05–0.2

### Kyle's Lambda (Price Impact Measure)
- **Author:** Albert S. Kyle
- **Venue:** Econometrica, 1985
- **Concept:** λ = price change per unit of net order flow. Proxy for market depth and liquidity.
- **Formula:** ΔP = λ × (order_imbalance)
- **NSE Usage:** Estimate λ from order book; use as feature for execution timing

### NSE-Specific Execution Considerations

| Factor | Impact | Mitigation |
|--------|--------|-----------|
| Pre-open auction (9:00–9:15) | Price discovery, high uncertainty | Enter positions only after 9:15 close |
| Opening volatility (9:15–9:45) | Wide spreads, 2–3× normal slippage | Wait 15–30 min before trading |
| Expiry Thursdays (monthly) | 50–200% volume spike + volatility | Reduce position size 50% on expiry week |
| Economic events (RBI, Budget) | Potential 3–5% overnight gap | Reduce exposure 1 day before |
| Circuit filter | 2%, 5%, 10%, 20% individual stock limits | Monitor circuit proximity in order_manager.py |

```python
# NSE F&O lot sizes (update periodically from NSE website)
LOT_SIZES = {
    'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 40,
    'MIDCPNIFTY': 75, 'RELIANCE': 250, 'TCS': 150, 'INFY': 300,
}
```

### ZeroMQ 3-Process Architecture for Live Trading
- **Library:** `pip install pyzmq`
- **Why ZeroMQ:** Standard Python threads share memory and fight for the GIL. For real-time trading, you need three concurrent workloads: (1) data ingestion from broker WebSocket, (2) ML feature computation + signal generation, (3) order execution + logging. A single threaded loop introduces latency — the feature engine blocks while waiting for the order to confirm. ZeroMQ provides lightweight message-passing between processes with microsecond latency and no GIL contention.
- **The 3-Process Architecture:**
```
Process 1: DATA COLLECTOR (broker WebSocket → ZeroMQ PUB socket)
   → Subscribes to Shoonya NorenAPI WebSocket
   → On each tick/candle: publish to ZMQ PUB socket on tcp://127.0.0.1:5555
   → Runs independently; never blocks on signal generation or orders

Process 2: SIGNAL GENERATOR (ZeroMQ SUB → feature engine → ZeroMQ PUSH)
   → Subscribes to DATA COLLECTOR's PUB socket
   → Computes 23 features + ML ensemble prediction
   → If signal generated: PUSH order request to tcp://127.0.0.1:5556
   → Runs independently; CPU-bound work does not delay data ingestion

Process 3: ORDER EXECUTOR (ZeroMQ PULL → broker API → trade logger)
   → Listens on PULL socket for order requests from SIGNAL GENERATOR
   → Places LIMIT orders via Shoonya API
   → Logs fills to QuestDB
   → Sends Telegram alerts
```
- **Key benefit:** If Process 2 (heavy ML computation) takes 200ms on a slow bar, it does not delay Process 1 from collecting the next tick. Each process runs at its own pace.
- **Pattern:** PUB/SUB for broadcast (market data → multiple consumers); PUSH/PULL for work queues (signal → executor); REQ/REP for request-response (status checks).
- **Deep coverage:** CONVO_DEEP_RESEARCH.md Part 1 has full implementation code for this architecture with Shoonya integration.
- **India Relevance:** HIGH — especially important during power hour (14:30–15:20) when tick frequency spikes and a single-threaded loop creates backpressure.

---

## 25. Advanced Feature Engineering

### Library Overview

| Library | GitHub | Stars | Features | Install |
|---------|--------|-------|---------|---------|
| **tsfresh** | https://github.com/blue-yonder/tsfresh | 8K+ | 794 time series features | `pip install tsfresh` |
| **TSFEL** | https://github.com/fraunhoferportugal/tsfel | 1K+ | 65+ features across domains | `pip install tsfel` |
| **catch22** | https://github.com/chloe-zh/catch22 | 500+ | 22 canonical features | `pip install pycatch22` |
| **stumpy** | https://github.com/TDAmeritrade/stumpy | 3K+ | Matrix Profile | `pip install stumpy` |
| **featuretools** | https://github.com/alteryx/featuretools | 7K+ | Automated feature engineering | `pip install featuretools` |

### SHAP (SHapley Additive exPlanations)

- **Title:** A Unified Approach to Interpreting Model Predictions
- **Authors:** Scott Lundberg, Su-In Lee
- **Venue:** NeurIPS 2017
- **ArXiv:** https://arxiv.org/abs/1705.07874
- **GitHub:** https://github.com/slundberg/shap
- **Why Critical:** Explains WHY your XGBoost/LightGBM made each prediction. SEBI audit trail requires explainability.

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### fracdiff (Fractional Differentiation)

- **Book Reference:** López de Prado "AFML" Chapter 5
- **Key Concept:** Price series is I(1) non-stationary. Fractional differentiation finds the minimal order d that makes series stationary while preserving maximum memory.

```python
from fracdiff.sklearn import FracdiffStat
fd = FracdiffStat()
fd.fit(price_series.reshape(-1, 1))
d_optimal = fd.d_  # typically 0.3–0.5 for price series
fracdiff_price = fd.transform(price_series.reshape(-1, 1))
```

### SMOTE for Label Imbalance

- **Python:** `from imblearn.over_sampling import SMOTE`
- **Why Critical:** Triple-barrier labeling produces class imbalance — often 40% class 0 (timeout), 35% class +1 (buy), 25% class -1 (sell). SMOTE synthetically creates minority class samples.
- **Warning:** Only apply SMOTE to training set (not validation/test). In walk-forward, apply separately to each fold's train set.

---

## 26. Free & Open Datasets

### Global Macroeconomic Data

| Dataset | Source | URL | Python Access |
|---------|--------|-----|---------------|
| **FRED** | St. Louis Fed | https://fred.stlouisfed.org/ | `pip install fredapi` |
| **World Bank Open Data** | World Bank | https://data.worldbank.org/ | `pip install wbdata` |

```python
from fredapi import Fred
fred = Fred(api_key='your_key')  # free at fred.stlouisfed.org
us_10yr = fred.get_series('GS10')  # US 10-year yield
vix = fred.get_series('VIXCLS')  # CBOE VIX
```

### Indian Market Free Data

| Dataset | Source | URL | Coverage |
|---------|--------|-----|----------|
| **NSE Bhavcopy** | NSE via jugaad-data | https://github.com/jugaad-py/jugaad-data | Daily EOD for all NSE stocks (2001–present) |
| **NSE F&O Bhavcopy** | jugaad-data | Same as above | Options chain snapshots at close |
| **RBI Statistics** | RBI | https://www.rbi.org.in/Scripts/Statistics.aspx | Interest rates, forex, banking data |
| **India VIX History** | NSE | https://www.nseindia.com/reports/india-vix-historical-data | India VIX daily |
| **NSE Bulk Deals** | NSE | https://www.nseindia.com/reports/bulk-deals | Large block trades (>0.5% of equity) |
| **NSE Insider Trading** | NSE | https://www.nseindia.com/companies-listing/corporate-filings-insider-trading | Promoter buy/sell |

```python
from jugaad_data.nse import stock_df
from datetime import date

df = stock_df(symbol="RELIANCE", from_date=date(2020,1,1), to_date=date(2024,12,31))
```

### AWS FinSpace — ⚠️ DEPRECATED: DO NOT USE

**Status:** SHUTTING DOWN — AWS announced October 2026 end-of-life. **Rating: 2/10**

Any new project using AWS FinSpace is inheriting a dead-end dependency. AWS is terminating the service.

**Replacement: QuestDB** (rated 10/10 technical, 8/10 India retail usefulness)
- Write speed: 4.3M rows/second benchmark
- SQL interface with ASOF JOIN and LATEST ON
- Free single-node tier — runs on any Linux VPS
- Docker: `docker pull questdb/questdb`
- Docs: https://questdb.io/docs/

---

## 27. Backtesting Frameworks — Deep Comparison

### Feature Comparison Matrix

| Feature | Backtrader | VectorBT | PyBroker | Zipline | QuantConnect | NautilusTrader |
|---------|-----------|---------|---------|---------|-------------|----------------|
| **Stars** | 21K ⚠️ | 5K ⚠️ | 3.1K ✓ | 20K | 19K ✓ | 22.6K ✓ |
| **Speed** | Slow | Very Fast | Fast | Slow | Fast (C#) | Extremely Fast (Rust) |
| **ML Integration** | Manual | Built-in | Excellent | Manual | Lean + Python | Manual |
| **Walk-Forward** | Manual | Built-in | Built-in | Backtesting.py | Built-in | Manual |
| **Live Trading** | Via broker adapters | No | Yes (planned) | No | Yes | Yes (production-grade) |
| **Active Maintenance** | Low | Medium | High | Low | High | High |

**Star count notes:** ⚠️ = misleading/stale stars (legacy from inactive era); ✓ = current, actively maintained. Backtrader (21K) and VectorBT OSS (5K) stars accumulated during active years — both are now largely unmaintained.

### Backtrader (21K Stars) — ⚠️ EFFECTIVELY ABANDONED

- **GitHub:** https://github.com/mementum/backtrader
- **Rating:** 4/10
- **CRITICAL WARNING:** Last PyPI release was version 1.9.74.123 on **May 30, 2019**. Creator considers it "complete" (i.e., not maintained). The 21,000 stars reflect historical popularity from 2015–2019 tutorials — not current health. Do not start new projects on Backtrader.

- **Replacement decision tree:**
  - Building ML-based strategy and need walk-forward? → **PyBroker**
  - Need fast parameter sweeps / signal screening? → **VectorBT**
  - Building production live trading system? → **NautilusTrader**
  - Need institutional-quality backtesting with options? → **QuantConnect Lean**

### VectorBT (5K Stars) — ⚠️ Open-Source Version Effectively Unmaintained

- **GitHub:** https://github.com/polakowo/vectorbt
- **Rating:** 6/10 (open-source) | 7.5/10 (VectorBT PRO)
- **Critical Note:** Active development has migrated to **VectorBT PRO** (closed-source commercial, ~$20/month). The open-source repo has seen minimal commits since 2022.
- **What VectorBT OSS still does well:** Fully vectorized NumPy/Numba backtesting — run 10,000 parameter combinations in under a minute. Excellent for rapid signal screening.
- **Better alternatives for ML strategies:** Use **PyBroker** instead.

### PyBroker (3.1K Stars) — Recommended for ML Strategy Research

- **GitHub:** https://github.com/edtechre/pybroker
- **Docs:** https://www.pybroker.com/en/latest/
- **Rating:** 7.5/10 for ML backtesting | Production readiness: 4/10
- **Maintenance:** Actively maintained (v1.2.11, December 2025). Install: `pip install lib-pybroker`
- **Key unique feature — BCa Bootstrap:** Uses Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals for Sharpe ratios. This is the correct approach when evaluating backtests with limited data.
- **NSE Integration:** Use custom data source with `jugaad-data` or broker historical API.

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance
from xgboost import XGBClassifier

def train_xgb(train_data, target, hyperparams):
    model = XGBClassifier(**hyperparams)
    model.fit(train_data, target)
    return model

def buy_rule(ctx):
    if ctx.model('xgb').predict(ctx.df.tail(1))[0] == 1:
        ctx.buy_shares = 10

config = StrategyConfig(initial_cash=100_000, fee_type=pybroker.FeeType.PERCENT, fee=0.002)
strategy = Strategy(YFinance(), '2022-01-01', '2024-12-31', config)
strategy.add_execution(buy_rule, ['RELIANCE.NS'],
                       models=[pybroker.ModelSource('xgb', train_xgb, pretrained=False)])
result = strategy.walkforward(windows=5, train_size=0.7)
result.metrics_df  # includes BCa bootstrap Sharpe intervals
```

### NautilusTrader (~22,000 Stars) — Upgraded Assessment

- **GitHub:** https://github.com/nautechsystems/nautilus_trader
- **Docs:** https://nautilustrader.io/
- **Rating:** 8/10 overall | Architecture: 10/10 | India readiness: 6/10
- **Star Note:** Stars surged from ~5,000 (2023 estimates) to ~22,000 (2025 actual).
- **Best For:** Production trading — Rust core eliminates simulation-to-reality gap. Backtest code and live trading code are identical.
- **Confirmed broker integrations (2025):** Interactive Brokers, Binance, Databento, Bybit, OKX, dYdX. **No native NSE/Indian broker adapters** — requires custom implementation.
- **Complexity:** Very High. Start with PyBroker for ML research, migrate to NautilusTrader for live production.

### QuantConnect Lean (~19,000 Stars) — Upgraded Assessment: 9/10

- **GitHub:** https://github.com/quantconnect/lean
- **Platform:** https://www.quantconnect.com/
- **Rating:** 9/10 (research-to-production pipeline)
- **Scale (2025):** 483,000 registered users | 100+ hedge fund clients | $45B+ monthly notional
- **Self-hostable:** Air-gapped deployment — run LEAN entirely on your own servers
- **Mia V2 AI assistant:** Writes LEAN algorithm code from natural language
- **India Relevance:** MEDIUM for direct NSE trading (data integration work needed), HIGH as architectural reference.

### Key Backtesting Pitfalls (Papers)

| Pitfall | Paper | Solution |
|---------|-------|---------|
| **Look-Ahead Bias** | Bailey et al. (2014) | Never use future data in indicator computation. Use `.shift(1)` on all features |
| **Survivorship Bias** | Brown et al. (1992) | Include delisted stocks in universe. NSE has removed 200+ stocks since 2010. |
| **Overfitting** | López de Prado (2018) CPCV | Use Combinatorial Purged CV, not standard k-fold |
| **Deflated Sharpe** | Bailey & López de Prado (2014) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 |
| **Transaction Costs** | Frazzini et al. AQR (2015) | Include full cost model: STT + GST + slippage. Real costs are 3–7× naive |

---

## 28. Advanced Risk Management Research

### Regime Detection

#### BOCPD (Bayesian Online Change Point Detection)
- **ArXiv:** https://arxiv.org/abs/0710.3742
- **Why Critical:** Detects the EXACT moment a regime changes (online, real-time). Unlike ruptures (offline), BOCPD processes each new candle and gives P(regime_change | new_data).

#### Turbulence Index (Mahalanobis Distance)
- **Concept:** T_t = (y_t - μ) Σ⁻¹ (y_t - μ)ᵀ where y_t = vector of asset returns
- **Why Critical:** Your multi-stock portfolio assumes stable correlations. Turbulence index detects crisis periods.

```python
from scipy.spatial.distance import mahalanobis

def turbulence_index(returns_df, lookback=252):
    turbulence = []
    for t in range(lookback, len(returns_df)):
        hist = returns_df.iloc[t-lookback:t]
        mu = hist.mean().values
        sigma = hist.cov().values
        today = returns_df.iloc[t].values
        try:
            VI = mahalanobis(today, mu, np.linalg.inv(sigma))**2
        except np.linalg.LinAlgError:
            VI = 0
        turbulence.append(VI)
    return pd.Series(turbulence, index=returns_df.index[lookback:])
```

### CVaR / Expected Shortfall Implementation

```python
import numpy as np

def cvar(returns, confidence=0.95):
    """CVaR (Expected Shortfall) at given confidence level"""
    sorted_returns = np.sort(returns)
    cutoff_idx = int((1 - confidence) * len(sorted_returns))
    return -sorted_returns[:cutoff_idx].mean()

def rolling_cvar(returns_series, window=252, confidence=0.95):
    return returns_series.rolling(window).apply(
        lambda x: cvar(x.values, confidence)
    )
```

### Drawdown Analytics

```python
def drawdown_analysis(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    annual_return = equity_curve.pct_change().mean() * 252
    calmar = annual_return / abs(max_dd)
    return {'max_drawdown': max_dd, 'calmar_ratio': calmar}
```

### Key Risk Management Papers

| Paper | Year | Key Contribution |
|-------|------|----------------|
| Riskfolio-Lib docs | 2021 | https://riskfolio-lib.readthedocs.io/ — 30+ portfolio optimization methods |
| Hierarchical Risk Parity | López de Prado (2016) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 |
| Equal Risk Contribution | Maillard et al. (2010) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1245743 |

---

## 29. Alternative Data Sources

### Sentiment & News Data

#### Reddit & Social Media
- **PRAW (Python Reddit API):** `pip install praw`
- **Key Subreddits:** r/IndiaInvestments (300K+), r/DalalStreetBets (150K+), r/algotrading (190K+)

```python
import praw
reddit = praw.Reddit(client_id='...', client_secret='...', user_agent='finance')
subreddit = reddit.subreddit('IndiaInvestments')
posts = list(subreddit.new(limit=100))
# Sentiment: use FinBERT on post titles
```

#### Google Trends (Retail Attention Signal)
- **Library:** `pip install pytrends`
- **Paper:** Da, Engelberg, Gao (2011) "In Search of Attention" — Google Trends for stocks predicts next-week returns
- **NSE Application:** High search volume for a stock = retail attention = potential price impact

```python
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-IN', tz=330)  # India timezone
pytrends.build_payload(['RELIANCE NSE', 'buy RELIANCE'], geo='IN', timeframe='today 3-m')
interest = pytrends.interest_over_time()
```

### NSE Option Chain (Programmatic Access)

```python
import requests

def get_nse_option_chain(symbol="NIFTY"):
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=headers)
    url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
    response = session.get(url, headers=headers)
    return response.json()['records']['data']
```

### FII/DII Flow Data (Programmatic)

```python
def get_fii_dii_data(from_date, to_date):
    """Fetch FII/DII provisional data from NSE"""
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=headers)
    params = {'startDate': from_date, 'endDate': to_date}
    return session.get(url, headers=headers, params=params).json()
```

### RBI & Macro Data (India)

| Source | URL | Data |
|--------|-----|------|
| **RBI DBIE** | https://dbie.rbi.org.in/ | Monetary policy rates, bank credit, forex reserves |
| **MOSPI** | https://mospi.gov.in/ | GDP, CPI, IIP (industrial production) |
| **Screener.in** | https://www.screener.in/ | Free Indian stock fundamentals (scrapable) |

---

## 30. Communities, Certifications & Learning Ecosystem

### Professional Certifications

| Certification | Cost | Duration | Focus | Relevance |
|--------------|------|----------|-------|-----------|
| **CFA (Level I → III)** | ~$1,500–3,000 | 3–4 years | Portfolio management | High |
| **FRM (GARP)** | ~$1,200 | 1–2 years | Financial risk management | Very High |
| **CQF** | ~£4,000 | 6 months | ML finance, derivatives, risk | Highest — specifically for quant trading |
| **CMT** | ~$1,200 | 1–2 years | Technical analysis | Moderate |
| **NSE NCFM** | ₹7,670 | Self-paced | Indian market regulations, F&O | High for India compliance |
| **QuantInsti EPAT** | ₹2–3 lakhs | 6 months | Algo trading Python, ML | Very High — India-specific |
| **Coursera ML for Trading** | ~$50/month | 3–4 months | ML + Google NYIF | High for ML fundamentals |

### Online Communities

| Community | URL | Members | Focus |
|-----------|-----|---------|-------|
| **r/algotrading** | https://www.reddit.com/r/algotrading/ | 190K+ | Code, strategies, broker APIs |
| **r/IndiaInvestments** | https://www.reddit.com/r/IndiaInvestments/ | 300K+ | Indian market discussion |
| **r/DalalStreetBets** | https://www.reddit.com/r/DalalStreetBets/ | 150K+ | Indian retail traders |
| **Quantocracy** | https://quantocracy.com/ | — | Daily curated quant blog posts |
| **Quant Stack Exchange** | https://quant.stackexchange.com/ | 20K+ questions | Technical quant Q&A |
| **QuantConnect Community** | https://www.quantconnect.com/forum | 100K+ | Strategy forum, code sharing |
| **OpenAlgo Discord** | discord.gg/openalgo | — | Indian broker API community |

### Blogs & Research Platforms

| Resource | URL | Must-Read Posts |
|----------|-----|----------------|
| **Quantpedia** | https://quantpedia.com/ | 1,200+ documented strategies |
| **Flirting with Models** | https://blog.thinknewfound.com/ | Factor investing, risk management |
| **Algo Trading India** | https://blog.quantinsti.com/ | India-specific algo trading tutorials |
| **OpenAlgo Docs** | https://docs.openalgo.in/ | Multi-broker abstraction layer |
| **Alpha Architect** | https://alphaarchitect.com/blog/ | Factor investing, research replication |

### GitHub Awesome Lists

| List | URL | Description |
|------|-----|-------------|
| **awesome-quant** | https://github.com/wilsonfreitas/awesome-quant | Master list, 15K stars |
| **awesome-systematic-trading** | https://github.com/wangzhe3224/awesome-systematic-trading | Stocks/crypto/futures/options/FX |
| **awesome-quant-ai** | https://github.com/leoncuhk/awesome-quant-ai | AI/ML specific quant resources |
| **best-of-algorithmic-trading** | https://github.com/merovinh/best-of-algorithmic-trading | 110+ ranked algo trading projects |

### Quick Reference: All Critical URLs (v3)

#### Core Papers (Free)
- XGBoost: https://arxiv.org/abs/1603.02754
- LightGBM: https://arxiv.org/abs/1706.01427
- CatBoost: https://arxiv.org/abs/1706.09516
- FinRL: https://arxiv.org/abs/2011.09607
- PPO: https://arxiv.org/abs/1707.06347
- SAC: https://arxiv.org/abs/1801.01290
- Decision Transformer: https://arxiv.org/abs/2106.01345
- TFT: https://arxiv.org/abs/1912.09363
- PatchTST: https://arxiv.org/abs/2211.14730
- iTransformer: https://arxiv.org/abs/2310.06625
- Mamba: https://arxiv.org/abs/2312.00752
- KAN: https://arxiv.org/abs/2404.19756
- TimeGPT: https://arxiv.org/abs/2310.03589
- Lag-Llama: https://arxiv.org/abs/2310.08278
- Moirai: https://arxiv.org/abs/2402.02592
- MOMENT: https://arxiv.org/abs/2402.03885
- 101 Alphas: https://arxiv.org/abs/1601.00991
- SHAP: https://arxiv.org/abs/1705.07874
- Deep Hedging: https://arxiv.org/abs/1802.03042
- BOCPD: https://arxiv.org/abs/0710.3742
- Almgren-Chriss: https://cims.nyu.edu/~almgren/papers/optliq.pdf
- EWC: https://arxiv.org/abs/1612.00796
- López de Prado papers: https://mlfortraders.com
- Deflated Sharpe: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

---

*Part A complete — 30 sections | All fixes from DB_FIXES_AND_ADDITIONS.md applied*

---

# PART B: DEEP TECHNICAL EXPANSION

---

## B1: Sections 1–5 Deep Expansion

*Source: DB_DEEP_S01_S05.md*

### 1.1 Microsoft Qlib — Deep Dive
- **URL:** https://github.com/microsoft/qlib
- **First file to study:** `/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
- **Heavy lifting class:** `qlib.workflow.R` (experiment manager) + `qlib.contrib.model.gbdt.LGBModel`

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
with R.start(experiment_name="lightgbm_alpha158"):
    model = init_instance_by_config({"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt"})
    dataset = init_instance_by_config({"class": "DatasetH", "kwargs": {"handler": {}, "segments": {}}})
    model.fit(dataset)
    recorder = R.get_recorder()
    SignalRecord(model, dataset, recorder).generate()
```

### 1.2 NautilusTrader — Deep Dive
- **First file to study:** `/examples/backtest/fx_ema_cross_audusd_bars_internal.py`
- **Heavy lifting class:** `nautilus_trader.backtest.engine.BacktestEngine` + `Strategy` base class

```python
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.trading.strategy import Strategy

config = BacktestEngineConfig(trader_id="TESTER-001")
engine = BacktestEngine(config=config)
engine.add_venue(venue=Venue("NSE"), oms_type=OmsType.NETTING,
                 account_type=AccountType.MARGIN, base_currency=INR,
                 starting_balances=[Money(1_000_000, INR)])
engine.add_instrument(equity)
engine.add_data(bars)
engine.add_strategy(strategy=MyStrategy(config=MyStrategyConfig(bar_type=bar_type)))
engine.run()
```

### 1.3 QuantConnect Lean — Deep Dive

```python
from AlgorithmImports import *

class MyAlgo(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetCash(1000000)
        self.symbol = self.AddEquity("SBIN", Resolution.Minute).Symbol
        self.sma = self.SMA(self.symbol, 20, Resolution.Minute)

    def OnData(self, data):
        if not self.sma.IsReady: return
        if data[self.symbol].Close > self.sma.Current.Value and not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 0.95)
        elif data[self.symbol].Close < self.sma.Current.Value:
            self.Liquidate(self.symbol)
```

### 1.4 FinRL — Deep Dive

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

env = StockTradingEnv(df=train_df, stock_dim=30, hmax=100,
                      initial_amount=1e6, transaction_cost_pct=0.001,
                      reward_scaling=1e-4, state_space=30*3+2, action_space=30)
agent = DRLAgent(env=env)
model_ppo = agent.get_model("ppo", model_kwargs={"n_steps": 2048, "learning_rate": 3e-4})
trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=500_000)
```

### 1.5 Stable-Baselines3 — Deep Dive

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

vec_env = make_vec_env(lambda: TradingEnv(df=train_df), n_envs=4)
eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model/",
                             eval_freq=10_000, deterministic=True)
model = PPO("MlpPolicy", vec_env, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1)
model.learn(total_timesteps=1_000_000, callback=eval_callback)
```

### 1.6 pandas-ta — Deep Dive

```python
import pandas_ta as ta
import pandas as pd

df = pd.read_parquet("nse_15min.parquet")
MyStrategy = ta.Strategy("NSE_Features",
    ta=[{"kind": "rsi", "length": 14}, {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20}, {"kind": "atr", "length": 14},
        {"kind": "adx", "length": 14}, {"kind": "supertrend"},
        {"kind": "stoch", "k": 14, "d": 3}, {"kind": "obv"}])
df.ta.strategy(MyStrategy)  # adds ~20 columns in one call
```

### 1.7 hmmlearn — Deep Dive

```python
import numpy as np
from hmmlearn.hmm import GaussianHMM

returns = np.diff(np.log(prices)).reshape(-1, 1)
vol = np.abs(returns)
X = np.column_stack([returns, vol])  # 2-feature observation

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model.fit(X)
regimes = model.predict(X)        # 0=bear, 1=sideways, 2=bull (check means to assign)
regime_probs = model.predict_proba(X)  # soft assignment for feature use
```

### 1.8 Optuna — Deep Dive

```python
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
              "max_depth": trial.suggest_int("max_depth", 3, 9),
              "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
              "subsample": trial.suggest_float("subsample", 0.6, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
              "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0)}
    model = XGBClassifier(**params, tree_method="hist", eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    return f1_score(y_val, model.predict(X_val), average="macro")

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=200, n_jobs=4)
```

---

## SECTION 2 — ML Papers: Technical Details (B1 Expansion)

### 2.1 XGBoost — Mathematical Core

**Objective Function:**
```
L(φ) = Σᵢ l(ŷᵢ, yᵢ) + Σₖ [γTₖ + ½λ||wₖ||²]

where:
  l = loss function (logistic for classification)
  Tₖ = number of leaves in tree k
  wₖ = leaf weight vector
  γ = min gain to make a split (regularization)
  λ = L2 leaf weight regularization
```

**Split Gain Formula:**
```
Gain = ½ [ GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ) ] - γ

where GL, GR = sum of first-order gradients (left/right child)
      HL, HR = sum of second-order gradients (Hessians)
Split only if Gain > 0
```

**NSE-tuned XGBClassifier:**
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    gamma=1.0,                  # min split gain — critical for noisy financial data
    reg_lambda=1.0,             # L2 regularization
    reg_alpha=0.1,              # L1 regularization
    scale_pos_weight=2.5,       # handle class imbalance
    tree_method="hist",         # faster on large datasets
    eval_metric=["logloss", "auc"],
    early_stopping_rounds=50, random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

### 2.2 LightGBM — Algorithmic Innovations

**GOSS:** sort by |gradient|; keep top a% (high-info) + random b% of remainder; amplify low-gradient samples by (1-a)/b. Typical: a=0.2, b=0.1 → 30% data, 20x speedup, ~90% info retained.

**EFB:** bundle mutually exclusive sparse features (non-zero values don't overlap) into single feature with offset; O(F) → O(B), B<<F. Critical for one-hot categoricals.

**Leaf-wise vs Level-wise:**
- Level-wise (XGBoost): grow all nodes at current depth → uniform, lower variance, safer for small data
- Leaf-wise (LightGBM): always split max-delta-loss leaf → asymmetric trees, faster convergence; NSE: `num_leaves=63, min_child_samples=50`

### 2.3 CatBoost — Key Mechanisms

**Ordered Boosting:** standard GB computes residual for sample i using model trained on ALL data including i → "prediction shift" (leakage). CatBoost fix: random permutation σ; residual for sample i uses model trained only on σ(1)..σ(i-1) → unbiased gradient.

**Symmetric Trees:** all nodes at same depth share identical split (feature_idx, split_value) → balanced binary tree → O(depth) inference, evaluable as table lookup; lower variance on noisy financial data.

### 2.4 Triple-Barrier Labeling (López de Prado, AFML Ch.3)

**Barrier Definitions:**
```
Entry price:    pt  at bar t
Upper barrier:  pt + h_upper = pt + (upper_mult × ATR14)   → target (default mult=2.0)
Lower barrier:  pt - h_lower = pt - (lower_mult × ATR14)   → stop-loss (default mult=1.0)
Time barrier:   t + max_holding_bars                        → default=20 bars (5h on 15min)

Label:
  +1  if upper barrier touched first
  -1  if lower barrier touched first
   0  if time barrier reached without either barrier being touched
```

**Complete Python Implementation:**
```python
import numpy as np
import pandas as pd

def triple_barrier_labels(
    close: pd.Series, atr: pd.Series,
    upper_mult: float = 2.0, lower_mult: float = 1.0,
    max_bars: int = 20
) -> pd.Series:
    """Label each bar using triple-barrier method."""
    labels = pd.Series(index=close.index, dtype=int)

    for i in range(len(close) - max_bars):
        entry = close.iloc[i]
        upper = entry + upper_mult * atr.iloc[i]
        lower = entry - lower_mult * atr.iloc[i]
        window = close.iloc[i+1 : i+1+max_bars]

        label = 0  # default: time barrier
        for j, price in enumerate(window):
            if price >= upper:
                label = 1
                break
            if price <= lower:
                label = -1
                break
        labels.iloc[i] = label

    return labels
```

### 2.5 Walk-Forward Validation with Purge + Embargo

**Concepts:**
- **Purge:** label window [t, t+max_bars] overlaps test start T → remove training samples in [T-max_bars, T)
- **Embargo:** skip n_embargo=5 additional bars post-purge (rolling feature leakage via HMM/GARCH)
- **Min folds: 5** (AFML statistical validity)

```python
import numpy as np
from typing import Iterator, Tuple

class WalkForwardSplit:
    def __init__(self, n_splits: int = 5, purge_bars: int = 20, embargo_bars: int = 5):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for fold in range(self.n_splits):
            test_start = (fold + 1) * fold_size
            test_end = test_start + fold_size
            purge_start = max(0, test_start - self.purge_bars - self.embargo_bars)
            train_idx = np.arange(0, purge_start)
            test_idx = np.arange(test_start, min(test_end, n))
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
```

### 2.6 Deflated Sharpe Ratio

**Formula:**
```
DSR = SR_hat × √((T-1)/T) − Z_α × √(1/T + (κ-1)/4 × SR_hat²)

where:
  SR_hat    = observed annualized Sharpe ratio
  T         = number of observations (bars)
  κ         = excess kurtosis of returns
  Z_α       = z-score for confidence level (1.645 for 95%)
  
DSR > 0 → significant after selection bias correction | DSR < 0 → likely data mining artifact
```

---

## SECTION 3 — RL Papers: Technical Details (B1 Expansion)

### 3.1 PPO (Proximal Policy Optimization)

**Clipped Surrogate Objective:**
```
L^CLIP(θ) = Ê_t [ min( r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t ) ]

where:
  r_t(θ)   = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)    (probability ratio)
  Â_t      = GAE advantage estimate at time t
  ε        = 0.2 (clip range, standard value)
  
Full objective:
  L(θ) = L^CLIP(θ) − c₁·L^VF(θ) + c₂·H[π_θ](sₜ)
  c₁=0.5 (value coefficient), c₂=0.01 (entropy bonus)

GAE:
  Â_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
  λ=0.95, γ=0.99 (standard for trading)
```

**Implementation notes for NSE trading env:**
```
State: [23 features + position + unrealized_pnl + time_in_trade] → dim=27
Action: continuous ∈ [-1, 1] → map to [-max_pos, max_pos]
Reward: log(pnl) − transaction_cost − λ·position_volatility
n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2
```

### 3.2 SAC (Soft Actor-Critic)

**Maximum Entropy Objective:**
```
J(π) = Σ_t E_{(sₜ,aₜ)~ρ_π} [ r(sₜ,aₜ) + α·H(π(·|sₜ)) ]

where:
  H(π(·|s)) = -E_{a~π}[log π(a|s)]   (entropy of policy)
  α          = temperature parameter (controls exploration-exploitation)

Bellman backup for soft Q-function:
  Q(sₜ,aₜ) = r(sₜ,aₜ) + γ·E_{sₜ₊₁}[ V(sₜ₊₁) ]
  V(s) = E_{a~π}[ Q(s,a) − α·log π(a|s) ]
```

**Automatic Temperature Tuning:**
```
α* = argmin_α E_{aₜ~π*}[ −α·log π*(aₜ|sₜ) − α·H̄ ]

where H̄ = target entropy = -dim(A) (negative action dimension)
For continuous trading action (dim=1): H̄ = -1

Gradient: ∇_α L(α) = E_{aₜ~π_φ}[ −log π_φ(aₜ|sₜ) − H̄ ]
Update: α ← α − β_α · ∇_α L(α)

SAC vs PPO: off-policy (replay buffer) → 2-3x fewer steps; entropy prevents always-hold collapse
```

### 3.3 DQN — Foundation

**Bellman Optimality:**
```
Q*(s,a) = E[ r + γ · max_{a'} Q*(s', a') | s, a ]

Loss:  L(θ) = E[(r + γ·max_{a'} Q(s',a'; θ⁻) − Q(s,a; θ))²]

Target network soft update:
  θ⁻ ← τ·θ + (1−τ)·θ⁻  (τ = 0.005)

Double DQN:
  y = r + γ · Q(s', argmax_{a'} Q(s',a'; θ); θ⁻)
  (action selection with online net, evaluation with target net)
```

**NSE note:** DQN requires discrete actions → use for buy/hold/sell baseline only. Use PPO/SAC for continuous position sizing.

---

## SECTION 4 — Deep Learning: Technical Details (B1 Expansion)

### 4.1 Transformer — Self-Attention

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

Multi-Head Attention:
  head_i = Attention(Q·Wᵢ^Q, K·Wᵢ^K, V·Wᵢ^V)
  MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O
  h=8 heads, dₖ=dᵥ=dmodel/h=64 (for dmodel=512)

Complexity: O(n²·d) per layer — quadratic in sequence length
```

### 4.2 TFT — Gated Residual Network

```
GRN(a, c) = LayerNorm( a + GLU(η₁) )

where:
  η₁ = W₁·ELU(W₂·a + b₂ + W_c·c) + b₁
  GLU(x) = x[:,: d/2] ⊙ σ(x[:, d/2:])     (gated linear unit)

TFT Components:
  1. Variable Selection Networks (VSN): learn feature importance per timestep
  2. GRN for local processing + context injection
  3. Sequence-to-sequence LSTM encoder/decoder
  4. Temporal self-attention (interpretable multi-head attention)
  5. Point-wise feed-forward + quantile loss output
```

### 4.3 iTransformer — Inverted Attention

```
Standard Transformer:   token = one time step, attention across time
iTransformer:           token = one variate (feature), attention across features

For multivariate time series with N variates, T timesteps:
  Standard: X ∈ R^{T×N}, each row is a token
  iTransformer: X ∈ R^{N×T}, each column (variate history) is a token

NSE (23 variates, 500 bars): O(23²) vs O(500²) → 470x cheaper
```

### 4.4 Mamba — Selective SSM

```
Continuous-time SSM:
  ẋ(t) = A·x(t) + B·u(t)
  y(t) = C·x(t)

Discretized:
  h_t = Ā·h_{t-1} + B̄·x_t
  y_t = C·h_t

Selective mechanism:
  B, C, Δ = functions of input x_t (NOT fixed parameters)
  → model selectively "forgets" or "remembers" based on content
  
Hardware-efficient via parallel scan (O(L log L) but GPU-parallel)
```

---

## SECTION 5 — NLP / LLMs: Technical Details (B1 Expansion)

### 5.1 FinBERT — Architecture

```
Base model:  BERT-base-uncased (110M params)
Fine-tuning: Financial PhraseBank (4,840 sentences, 3 classes)
Input:       [CLS] text [SEP] → 12 Transformer layers, 768 hidden dim
Output:      positive | negative | neutral
Accuracy:    ~97% on positive, ~97% on negative, ~94% on neutral
```

**Python inference (5 lines):**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
inputs = tokenizer("RBI hikes interest rates by 50bps, markets fall sharply",
                   return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
sentiment = ["positive", "negative", "neutral"][logits.argmax().item()]
```

### 5.2 FinGPT — LoRA Fine-Tuning

**LoRA (Low-Rank Adaptation):**
```
Original weight matrix:  W ∈ R^{d×k}
LoRA decomposition:      W' = W + B·A
  where B ∈ R^{d×r},  A ∈ R^{r×k},  rank r << min(d, k)

Typical: d=4096, k=4096, r=8 → trainable params per layer = 65,536
vs full fine-tune:  4096 × 4096 = 16,777,216 params per layer
Reduction factor:   256x per layer

For LLaMA-2-13B: LoRA (r=8) → ~8M trainable params → ~24GB GPU RAM
Initialization:  A ~ N(0, σ²),  B = 0
Scaling:         W' = W + (α/r)·B·A  where α=16
```

**Pipeline:**
```
Data sources → clean → label (FinBERT as weak labeler)
→ Format as instruction pairs:
  {"instruction": "Analyze sentiment", "input": headline, "output": "positive|negative|neutral"}
→ Base LLM: LLaMA-2-7B or Mistral-7B
→ Apply QLoRA (4-bit quantized base + LoRA) → fits on single 24GB GPU
→ Inference: run at 9:15 AM IST on overnight news → sentiment score ∈ [-1,+1]
```

---

## B1 Quick Reference: Key Formulas

| Formula | Use |
|---------|-----|
| `L^CLIP = min(r·Â, clip(r,1-ε,1+ε)·Â)` | PPO policy update, ε=0.2 |
| `J(π) = Σ[r + α·H(π)]` | SAC max-entropy objective |
| `Q*(s,a) = r + γ·max Q*(s',a')` | DQN Bellman equation |
| `Attention = softmax(QKᵀ/√dₖ)V` | Transformer self-attention |
| `h_t = Ā·h_{t-1} + B̄·x_t` | Mamba SSM recurrence |
| `W' = W + B·A, r<<d` | LoRA weight adaptation |
| `Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ` | XGBoost split criterion |
| `DSR = SR·√((T-1)/T) - Zα·√(1/T + (κ-1)/4·SR²)` | Deflated Sharpe Ratio |

*End of B1 — Sections 1–5 Deep Expansion*

---

## B2: Sections 6–10 Deep Expansion

*Source: DB_DEEP_S06_S10.md*

### 6.1 Bid-Ask Spread Decomposition (Glosten-Milgrom-Harris)
```
S = 2c + 2|z|λQ
c = order processing cost  |  z = inventory imbalance (signed)
λ = adverse selection fraction  |  Q = trade direction ±1
```
```python
# Roll's spread estimator (from price changes):
cov_dp = returns.rolling(2).cov().shift(1)    # cov(ΔP_t, ΔP_{t-1})
c_roll = np.sqrt(-cov_dp.clip(upper=0))        # Roll spread = 2√(-cov)
# Adverse selection fraction λ/(λ+c) ≈ 30-40% for NSE liquid stocks
# Inventory component dominates illiquid mid-cap; adverse selection dominates options
```

### 6.2 Kyle's Lambda — Price Impact
```
λ = Cov(ΔP, x) / Var(x)     [₹/share per unit signed order flow]
```
```python
from sklearn.linear_model import LinearRegression
X = signed_order_flow.values.reshape(-1, 1)
model = LinearRegression().fit(X, mid_price_change.values)
kyle_lambda = model.coef_[0]           # slope = λ
# ΔP_impact = λ × Q  →  slippage estimate for order size Q
# NSE Nifty 50: λ ≈ 0.0001–0.0005 ₹/share/lot (15-min bars)
```

### 6.3 Amihud Illiquidity
```
ILLIQ = (1/T) × Σ |R_t| / (VOL_t × P_t)    [higher = more illiquid]
```
```python
def amihud_illiquidity(returns, volume, price, window=21):
    return (returns.abs() / (volume * price)).rolling(window).mean()
# NSE mid-cap ILLIQ typically 10x–100x Nifty 50 constituents
```

### 6.4 VWAP
```
VWAP_t = Σ(Pᵢ × Vᵢ) / ΣVᵢ
```
```python
def vwap(price, volume): return (price * volume).cumsum() / volume.cumsum()
# Feature #8: dev = (close - vwap) / vwap
# Positive dev → price above VWAP → bearish mean-reversion (intraday)
# NSE volume profile: U-shaped (heavy at 09:15-09:45 and 15:00-15:30)
```

### 6.5 Almgren-Chriss Optimal Execution
```
x*(t) = x₀ × sinh(κ(T−t)) / sinh(κT)     κ = √(η / (σ²τ²))
η = temporary impact  |  σ² = price variance  |  τ = time step
```
```python
def almgren_chriss_trajectory(x0, T, kappa):
    t = np.arange(T + 1)
    return -np.diff(x0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T))
# κ → 0: pure TWAP  |  κ → ∞: execute immediately
# Retail (<0.1% ADV): negligible impact; use LIMIT orders
# Institutional (>1% ADV): must use AC framework or VWAP algo
```

### 6.6 Order Book Imbalance (Feature #16)
```python
def obi(bid_vol, ask_vol): return (bid_vol - ask_vol) / (bid_vol + ask_vol)
# Range [-1,+1]; Spearman corr with next 5-tick return ≈ 0.08–0.15
# Half-life ~10 sec on NSE; at 15-min bar: aggregate mean of tick-level OBI
```

---

## SECTION 7 — Portfolio Theory: Technical Details (B2 Expansion)

### 7.1 Markowitz Mean-Variance Optimization
```
min σ²_p = wᵀΣw    s.t. wᵀμ = μ_p,  wᵀ1 = 1,  w ≥ 0
w* = (1/γ) × Σ⁻¹ × (μ − λ₁·1)    [Lagrangian, long-short]
```
```python
from pypfopt import EfficientFrontier, risk_models, expected_returns
mu = expected_returns.mean_historical_return(prices)
S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()   # shrinkage!
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
# Ledoit-Wolf shrinkage: Σ_shrunk = (1-δ)Σ_sample + δ×F  minimizes MSE
```

### 7.2 CVaR / Expected Shortfall
```
VaR_α = α-th quantile of loss distribution
CVaR_α = VaR_α + (1/(1−α)) × E[max(−X − VaR_α, 0)]
       = average of losses beyond VaR_α
```
```python
def cvar(returns, alpha=0.05):
    var = np.percentile(returns, alpha * 100)
    return -returns[returns <= var].mean()
# VaR not subadditive → CVaR preferred (coherent risk measure)
# Your kill switch ≈ 99% CVaR breach trigger
```

### 7.3 Drawdown, Calmar, MDD
```
MDD = max_{t∈[0,T]} [ max_{s∈[0,t]} X(s) − X(t) ]
Calmar = CAGR / |MDD|
```
```python
def max_drawdown(eq):
    return ((eq - eq.cummax()) / eq.cummax()).min()

def calmar(eq, freq=252):
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (freq/len(eq)) - 1
    return cagr / abs(max_drawdown(eq))
# Thresholds: MDD < 15% (retail). Calmar > 1.0 acceptable; > 2.0 good
```

### 7.4 Sharpe, Sortino, Information Ratio
```
SR     = (R_p − R_f) / σ_p × √252
Sortino = (R_p − MAR) / σ_downside × √252
IR     = α / σ_α    (active return / tracking error)
```
```python
def sharpe(returns, rf=0.065/252):     # India 10Y ≈ 6.5%
    e = returns - rf
    return e.mean() / e.std() * np.sqrt(252)

def sortino(returns, mar=0.0):
    down = returns[returns < mar] - mar
    return (returns.mean() - mar) / np.sqrt((down**2).mean()) * np.sqrt(252)
# Live thresholds: SR < 0.5 → don't go live; SR > 1.5 → acceptable
```

### 7.5 Kelly Criterion & ATR Sizing
```
f* = (b·p − q) / b         [binary Kelly]
f* = μ / σ²  = SR² / σ     [continuous Kelly]
f_actual = f* / k          [k=2: half Kelly; k=4: quarter Kelly]
Half Kelly: -75% variance, -25% geometric growth vs full Kelly
```
```python
def atr_position_size(capital, atr, risk_pct=0.01):
    """Risk risk_pct of capital per trade. Stop = 2×ATR below entry."""
    return int(capital * risk_pct / (2.0 * atr))
    # Approximates fractional Kelly when ATR ≈ σ × price
```

---

## SECTION 8 — Factor Research: Technical Details (B2 Expansion)

### 8.1 Fama-French Factor Models
```
3F (1993):  R_i − R_f = α + β_mkt(Rm−Rf) + β_SMB·SMB + β_HML·HML
4F Carhart: + β_MOM·MOM   [MOM = top decile − bottom decile, 12M ex last month]
5F (2015):  + β_RMW·RMW + β_CMA·CMA
  RMW = Robust − Weak profitability (high ROE − low ROE)
  CMA = Conservative − Aggressive investment (low asset growth − high)
India: SMB effect stronger (EM small-cap premium); HML weaker than US
```
```python
import statsmodels.api as sm
def factor_regression(returns, factors):
    model = sm.OLS(returns, sm.add_constant(factors)).fit()
    return {'alpha': model.params['const'], 't_alpha': model.tvalues['const'],
            'r2': model.rsquared, 'betas': model.params.drop('const').to_dict()}
    # alpha t-stat > 2.0 required to claim genuine alpha
```

### 8.2 WorldQuant Alpha Formulas
**Alpha #1 — momentum-vol interaction:**
```python
def alpha_001(close, returns):
    sp = lambda x, a: np.sign(x) * np.abs(x)**a
    sig = np.where(returns < 0, returns.rolling(20).std(), close)
    return pd.DataFrame(sp(sig, 2)).rolling(5).apply(np.argmax).rank(pct=True) - 0.5
```
**Alpha #12 — volume-price divergence:**
```python
def alpha_012(close, volume):
    return np.sign(volume.diff()) * (-close.diff())
    # Positive: vol up + price down → accumulation (bullish)
```
**Alpha #20 — gap vs prior range:**
```python
def alpha_020(o, h, l, c):
    r = lambda x: x.rank(axis=1, pct=True)
    return (-r(o - h.shift(1))) * r(o - c.shift(1)) * r(o - l.shift(1))
```

### 8.3 Factor Decay & Half-Life
```
f_t = ρ·f_{t-1} + ε_t     [AR(1)]
HL  = −ln(2) / ln(ρ)      [half-life in same units as series]
```
```python
def factor_half_life(series):
    y, x = series.iloc[1:].values, series.iloc[:-1].values
    rho = sm.OLS(y, sm.add_constant(x)).fit().params[1]
    return -np.log(2) / np.log(rho) if 0 < rho < 1 else np.inf
# HL < 1 day: HFT signal; 1–5 days: good for 15-min system; >20 days: slow factor
# MACD signal HL ≈ 3–5 days; 6M momentum HL ≈ 30–60 days
```

### 8.4 Factor IC & Orthogonalization
```
IC_t = Spearman_corr(f_t, R_{t+1})
ICIR = mean(IC) / std(IC)    [ICIR > 0.3: use it; > 0.5: strong]
```
```python
def orthogonalize(factors):
    orth = {factors.columns[0]: factors.iloc[:, 0]}
    for col in factors.columns[1:]:
        orth[col] = sm.OLS(factors[col], sm.add_constant(
                           pd.DataFrame(orth))).fit().resid
    return pd.DataFrame(orth)
```

---

## SECTION 9 — Indian Market Research: Technical Details (B2 Expansion)

### 9.1 NSE vs NYSE Structural Comparison
| Metric | NSE (Nifty 50) | NYSE (S&P 500) |
|--------|----------------|----------------|
| Intraday vol (ann.) | 20–25% | 15–18% |
| Session | 09:15–15:30 IST | 09:30–16:00 ET |
| F&O expiry | Last Thursday monthly | 3rd Friday monthly |
| Circuit breakers | 10/15/20% market-wide | 7/13/20% |
| Settlement | T+1 (2024+) | T+1 (2024) |
| Brokerage (discount) | ₹20/order flat | $0–1/trade |
| STT (intraday sell) | 0.0125% | None |
| Round-trip all-in | ≈0.08–0.12% intraday | ≈0.02–0.05% |

### 9.2 F&O Expiry Effect
- Spot vol spikes 2–3 days before last Thursday of month
- NSE F&O turnover ≈ 15× spot volume (one of highest globally)
- Intraday range (H−L)/close increases ~30–40% in expiry week

```python
def expiry_proximity(dates, n_days=3):
    """Feature #24: 1 if within n_days of monthly F&O expiry."""
    def last_thu(year, month):
        import calendar
        cal = calendar.monthcalendar(year, month)
        thursdays = [w[3] for w in cal if w[3] > 0]
        return pd.Timestamp(year, month, thursdays[-1])
    return pd.Series([int((last_thu(d.year, d.month) - d).days <= n_days)
                      for d in dates], index=dates)
```

### 9.3 Calendar Anomalies
| Month | Bias | Driver |
|-------|------|--------|
| Jan | Negative | DII tax-loss selling (India FY = Apr-Mar) |
| Mar | Negative | FY-end rebalancing |
| Apr | Positive | New FY inflows; Q4 results |
| Sep | Positive | Post-monsoon; FII rebalancing |
| Budget Day (Feb 1) | ±3–5% | Unpredictable direction; go flat |

**Monday effect:** NSE Monday returns −0.03% to −0.06% (opposite of US).

**Budget day protocol:** No overnight hold. Widen stops 2×. ATR = 3–5× normal.

### 9.4 FII Flow as Leading Indicator
```python
def fii_feature(fii_net, threshold=2000):
    """FII net in ₹ Crore. Use with 1-day lag."""
    return pd.cut(fii_net, bins=[-np.inf, -threshold, threshold, np.inf],
                  labels=[-1, 0, 1]).astype(int)
# FII >+₹2000 Cr → next-day Nifty corr ≈ +0.25–0.35
# FII <−₹2000 Cr → next-day corr ≈ −0.20–0.30
```

---

## SECTION 10 — Must-Read Books: Technical Details (B2 Expansion)

### 10.1 AFML — López de Prado

**(A) Triple-Barrier Labeling:**
```python
def triple_barrier(close, t0, atr, upper_mult=2.0, lower_mult=1.0, max_bars=20):
    upper = close[t0] + upper_mult * atr
    lower = close[t0] - lower_mult * atr
    w = close[t0: t0 + max_bars]
    tu = w[w >= upper].index[0] if (w >= upper).any() else None
    tl = w[w <= lower].index[0] if (w <= lower).any() else None
    if tu and tl: return 1 if tu < tl else -1
    return 1 if tu else (-1 if tl else 0)   # 0 = time barrier
```

**(B) Fractional Differentiation:**
```
FracDiff(d) = Σ_{k=0}^∞ w_k(d) · x_{t-k}
w_0=1,  w_k = −w_{k-1} × (d−k+1)/k
```
```python
def fracdiff_weights(d, n):
    w = [1.0]
    for k in range(1, n):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])
# Optimal d ∈ [0.3, 0.5]: preserves memory, achieves stationarity
```

**(C) CPCV:**
```
N_paths = C(N_splits, N_test_splits)
Use N=6, k=2 → 15 backtest paths → distribution of SR (not single value)
```

### 10.2 Carver Systematic Trading

**(A) Forecast Scaling:**
```python
def scale_forecast(raw, target=10.0):
    return (raw * target / raw.abs().mean()).clip(-20, 20)
```

**(B) Instrument Diversification Multiplier (IDM):**
```
IDM = 1 / √(wᵀ ρ w)    [ρ = return correlation matrix, w = weights; IDM ≥ 1]
```
```python
def idm(w, rho): return 1.0 / np.sqrt(w @ rho @ w)
```

**(C) Position Size:**
```
N = (Capital × IDM × weight × τ) / (price × σ_annual)
τ = target portfolio vol (e.g. 0.20)
```

### 10.3 Options — BSM + Greeks

```python
from scipy.stats import norm

def bsm(S, K, r, T, sig):
    d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(mkt, S, K, r, T, tol=1e-6):
    sig = 0.3
    for _ in range(200):
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        diff = mkt - bsm(S, K, r, T, sig)
        if abs(diff) < tol: break
        sig += diff / (S * np.sqrt(T) * norm.pdf(d1))   # Newton-Raphson
    return sig
```

**(B) Vol Surface — SVI Parameterization (Gatheral):**
```python
def svi(k, a, b, rho, m, sig):
    return a + b * (rho*(k-m) + np.sqrt((k-m)**2 + sig**2))
# w(k) = a + b[ρ(k−m) + √((k−m)²+σ²)]
# k = log(K/F),  w = total variance
```

**(C) Gamma-Theta Breakeven:**
```python
def gamma_breakeven(gamma, theta, dt=1.0):
    return np.sqrt(2 * abs(theta) * dt / gamma)
# If ATR < breakeven → long gamma losing money
```

**(D) India VIX Regime:**
```python
def vix_regime(vix):
    """0=low(<12), 1=normal(12-20), 2=high(>20)"""
    return pd.cut(vix, [0,12,20,np.inf], labels=[0,1,2]).astype(int)
```

---

## B2 Quick Reference — Formula Index

| Formula | Section | Key Use |
|---------|---------|---------|
| S = 2c + 2\|z\|λQ | 6.1 | Spread decomposition |
| λ = Cov(ΔP,x)/Var(x) | 6.2 | Kyle's price impact |
| ILLIQ = (1/T)Σ\|R\|/(VOL·P) | 6.3 | Amihud illiquidity |
| x*(t) = x₀·sinh(κ(T-t))/sinh(κT) | 6.5 | Almgren-Chriss execution |
| CVaR = average losses beyond VaR | 7.2 | Coherent risk |
| MDD = max[peak−trough] | 7.3 | Drawdown |
| SR = (Rp-Rf)/σp × √252 | 7.4 | Sharpe (daily) |
| f* = μ/σ² | 7.5 | Kelly fraction |
| HL = -ln(2)/ln(ρ) | 8.3 | Factor decay |
| IDM = 1/√(wᵀρw) | 10.2 | Diversification multiplier |
| d₁ = [ln(S/K)+(r+σ²/2)T]/(σ√T) | 10.3 | BSM / IV solver |
| w(k) = a+b[ρ(k-m)+√((k-m)²+σ²)] | 10.3 | SVI vol surface |

*End of B2 — Sections 6–10 Deep Expansion*

---

## B3: Sections 11–15 Deep Expansion

---

## SECTION 11 — Institutional Research Sources (Deep Expansion)

### Two Sigma — Technical Research Depth

**Venn Platform Factor Decomposition:**
Venn decomposes portfolio returns into Two Sigma's proprietary factor set:
- Equity factors: Equity, Interest Rates, Credit, Commodities, Emerging Markets
- Style factors: Equity Short Volatility, Trend Following, Small Cap
- API access: `pip install venn-api` (institutional accounts only)

**What to extract for NSE research:**
- Factor return attribution — run your NSE backtest P&L through Venn to see how much is pure "equity beta" vs genuine alpha
- Correlation of India factors with global risk-off episodes (2020 COVID, 2022 rate hikes)
- Two Sigma whitepaper: "Factor Investing in Emerging Markets" (search Venn Insights)

**Key R&D metrics (public disclosures):**
- 1,600+ researchers across quant finance, ML, distributed systems
- 45+ petabytes of financial data under management
- Research cycle: hypothesis → simulation → live paper trading → production (avg 18 months)

---

### AQR Capital — Directly Applicable Research

**AQR Free Datasets (download these):**
```
https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Daily
https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-10-Quality-Components-Daily
https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly-Returns
https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Portfolios-Monthly
```

**Most cited paper for NSE application:**
"Fact, Fiction and Momentum Investing" (Asness et al.) — key findings:
1. Momentum works in 40+ countries including emerging markets
2. Half-life of momentum signal in emerging markets: 6–9 months (shorter than US 12-month)
3. Transaction costs matter more in EM: use only large-cap NSE 100 stocks for momentum

**AQR's 4-factor model implementation:**
```python
# AQR factors: Value, Momentum, Carry, Defensive
# For NSE equities, approximate with:
value_score     = -1 * (price / book_value_per_share)       # lower P/B = higher score
momentum_score  = returns_12m - returns_1m                  # skip 1-month reversal
quality_score   = roe * (1 - debt_to_equity / 10)          # profitability * leverage
low_vol_score   = -1 * realized_vol_30d                    # lower vol = higher score
composite = 0.25*value_score + 0.25*momentum_score + 0.25*quality_score + 0.25*low_vol_score
```

---

### Man AHL — Trend-Following Insights for NSE

**Oxford Man Institute Papers (NSE-relevant):**
- "Does trend following work on stocks?" — YES on NSE, but only for Nifty 50 + 100 constituents
- Cross-asset momentum: Nifty futures + USD/INR + Gold MCX — diversification improves Sharpe

**Man AHL's time-series momentum rule (exact form):**
```python
# TSMOM signal (from Moskowitz, Ooi, Pedersen 2012)
def tsmom_signal(returns_series, lookback=252):
    past_return = returns_series.rolling(lookback).apply(lambda x: x.prod() - 1)
    vol = returns_series.rolling(63).std() * np.sqrt(252)  # annualized vol
    signal = past_return / vol  # vol-scaled signal
    # Apply to NSE: lookback=60 (3 months) works better than 252
    return np.sign(signal) * min(abs(signal), 2.0)  # cap at 2.0
```

---

### WorldQuant — Alpha Mining for NSE

**101 Formulaic Alphas (most NSE-applicable):**

Alpha 1 (rank correlation with volume):
```python
# Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0 ? stddev(returns, 20) : close)), 2.), 5)) - 0.5)
# NSE implementation
def alpha_001(close, returns, volume):
    cond = returns < 0
    x = np.where(cond, close.rolling(20).std(), close)
    return x**2
```

Alpha 12 (volume and price change signal):
```python
# Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha_012(close, volume):
    return np.sign(volume.diff(1)) * (-1 * close.diff(1))
```

Alpha 20 (open-close momentum):
```python
# Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
def alpha_020(open_, high, low, close):
    return ((-1 * (open_ - high.shift(1)).rank(pct=True))
            * (open_ - close.shift(1)).rank(pct=True)
            * (open_ - low.shift(1)).rank(pct=True))
```

**WorldQuant WebSim alternative (free):** Use QuantConnect Alpha Streams to test WorldQuant-style alphas with Indian data.

---

## SECTION 12 — Indian Broker APIs (Deep Technical Expansion)

### Full API Comparison Table

| Broker | API Cost | Brokerage (Intraday) | Rate Limit | WebSocket | Market Depth | Historical Candles | TOTP Required |
|--------|----------|----------------------|------------|-----------|--------------|-------------------|---------------|
| Zerodha Kite | ₹500/mo | ₹20/order | 10 req/s REST | Yes (port 443) | 5-level | 2000 candles/req, 10yr daily | Yes |
| Angel One SmartAPI | Free | ₹20/order | 20 req/s | Yes (wss://) | 5-level | 2000 candles/req | Yes |
| Dhan (DhanHQ) | Free | ₹20/order | ~10 req/s | Yes (v2) | 200-level | 2000 candles/req | Yes |
| Upstox v3 | Free | ₹20/order | ~10 req/s | Yes (v3) | 200-level | 2000 candles/req | Yes |
| Shoonya/Finvasia | Free | ₹0 | ~10 req/s | Yes (NorenAPI) | 5-level | Available | Yes |
| Fyers v3 | Free | ₹20/order | 100 req/min REST | Yes | 5-level | 2000 candles/req | Yes |
| Kotak Neo | Free | ₹0 | Variable | REST polling | 5-level | Limited | Yes |
| ICICI Breeze | Free | ~₹20/order | 75 req/min | Yes | 5-level | 1 year intraday | Yes |

**Rate limit notes:**
- "10 req/s" means per API endpoint; WebSocket is event-driven, not counted
- Burst limits apply: most brokers allow 20-req bursts before throttling
- F&O option chain calls count separately from equity calls

---

### Zerodha Kite Connect — Technical Depth

**Authentication flow (TOTP):**
```python
import pyotp
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="your_api_key")

# Step 1: Generate login URL
login_url = kite.login_url()
# Open login_url in browser, complete login, grab request_token from redirect URL

# Step 2: Exchange request_token for access_token
# TOTP generated by broker app — for automation use pyotp:
totp = pyotp.TOTP("YOUR_TOTP_SECRET_BASE32")
current_otp = totp.now()  # pass this to automated browser flow (selenium/playwright)

# Step 3: Generate session
data = kite.generate_session("request_token_from_redirect", api_secret="your_secret")
kite.set_access_token(data["access_token"])
```

**historical_data() exact parameters:**
```python
# Signature: kite.historical_data(instrument_token, from_date, to_date, interval, continuous, oi)
data = kite.historical_data(
    instrument_token=408065,         # NSE:NIFTY 50 instrument token
    from_date="2025-01-01 09:15:00", # datetime string or datetime object
    to_date="2025-01-31 15:30:00",
    interval="15minute",             # minute, 3minute, 5minute, 10minute, 15minute,
                                     # 30minute, 60minute, day, week, month
    continuous=False,                # True for continuous F&O contracts
    oi=True                          # Include open interest (F&O only)
)
# Returns: list of dicts with keys: date, open, high, low, close, volume, oi
```

**Max candle limits per interval:**
```
minute      → 60 days per request
3minute     → 100 days per request
5minute     → 100 days per request
10minute    → 100 days per request
15minute    → 200 days per request
30minute    → 200 days per request
60minute    → 400 days per request
day         → 2000 days per request (~5.5 years)
week        → 2000 weeks per request
month       → 2000 months per request
```
**Pagination approach for 10 years of 15-min data:**
```python
import pandas as pd
from datetime import timedelta

def fetch_full_history(kite, token, interval="15minute", years=10):
    end = pd.Timestamp.now(tz='Asia/Kolkata').replace(tzinfo=None)
    start = end - pd.DateOffset(years=years)
    chunk_days = 200  # max for 15minute
    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        data = kite.historical_data(token, chunk_start, chunk_end, interval)
        frames.extend(data)
        chunk_start = chunk_end + timedelta(seconds=1)
    return pd.DataFrame(frames).set_index("date")
```

**WebSocket tick JSON format (exact):**
```json
{
  "tradable": true,
  "mode": "full",
  "instrument_token": 408065,
  "last_price": 22450.50,
  "last_traded_quantity": 75,
  "average_traded_price": 22380.25,
  "volume_traded": 12345678,
  "total_buy_quantity": 500000,
  "total_sell_quantity": 480000,
  "ohlc": {
    "open": 22300.00,
    "high": 22500.00,
    "low": 22250.00,
    "close": 22350.00
  },
  "change": 0.45,
  "last_trade_time": "2025-01-15 14:32:10",
  "exchange_timestamp": "2025-01-15 14:32:10",
  "oi": 45678900,
  "oi_day_high": 46000000,
  "oi_day_low": 44000000,
  "depth": {
    "buy": [
      {"price": 22449.00, "quantity": 150, "orders": 3},
      {"price": 22448.50, "quantity": 225, "orders": 5},
      {"price": 22448.00, "quantity": 300, "orders": 7},
      {"price": 22447.50, "quantity": 450, "orders": 9},
      {"price": 22447.00, "quantity": 600, "orders": 12}
    ],
    "sell": [
      {"price": 22451.00, "quantity": 75, "orders": 2},
      {"price": 22451.50, "quantity": 150, "orders": 4},
      {"price": 22452.00, "quantity": 225, "orders": 6},
      {"price": 22452.50, "quantity": 375, "orders": 8},
      {"price": 22453.00, "quantity": 525, "orders": 11}
    ]
  }
}
```

**Subscription modes:** `ltp` (price only) | `quote` (OHLC + volume) | `full` (above + depth)

**Common Zerodha errors and fixes:**
```
TokenException: Invalid checksum      → Access token expired; regenerate daily before 9:00 AM
InputException: Invalid `from` date   → from_date > to_date, or date outside available range
NetworkException: Quota exceeded      → Rate limit hit; add time.sleep(0.1) between calls
DataException: No data                → Holiday or instrument not traded on that date; check NSE calendar
OrderException: Insufficient margin   → F&O margin check failed; use kite.order_margins() before placing
```

---

### Angel One SmartAPI — Technical Depth

**Authentication with TOTP (complete working code):**
```python
import pyotp
from smartapi import SmartConnect

def angel_login(api_key: str, client_id: str, password: str, totp_secret: str) -> SmartConnect:
    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret)
    data = obj.generateSession(client_id, password, totp.now())
    if data['status'] is False:
        raise ConnectionError(f"Angel One login failed: {data['message']}")
    auth_token = data['data']['jwtToken']
    refresh_token = data['data']['refreshToken']
    feed_token = obj.getfeedToken()
    return obj  # obj now holds active session
```

**getCandleData() exact parameters:**
```python
# historicDataV2 (preferred over getCandleData for stability)
params = {
    "exchange": "NSE",           # NSE, BSE, NFO, MCX, CDS
    "symboltoken": "3045",       # instrument token (NOT trading symbol)
    "interval": "FIFTEEN_MINUTE", # ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE,
                                  # TEN_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE,
                                  # ONE_HOUR, ONE_DAY, ONE_WEEK, ONE_MONTH
    "fromdate": "2025-01-01 09:15",  # format: YYYY-MM-DD HH:MM
    "todate": "2025-01-31 15:30"
}
data = obj.getCandleData(params)
# Returns: {"status": true, "data": [["2025-01-01T09:15:00+05:30", open, high, low, close, volume], ...]}
```

**Max candle limits (Angel One):**
```
ONE_MINUTE      → 30 days per request
THREE_MINUTE    → 60 days per request
FIVE_MINUTE     → 100 days per request
FIFTEEN_MINUTE  → 200 days per request
ONE_HOUR        → 400 days per request
ONE_DAY         → 2000 days per request
```

**Instrument token lookup (critical — symbol strings NOT accepted):**
```python
import requests, zipfile, io, pandas as pd

def get_angel_instrument_df() -> pd.DataFrame:
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    resp = requests.get(url)
    df = pd.DataFrame(resp.json())
    # Columns: token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg, tick_size
    return df

def get_token(df, symbol: str, exchange: str = "NSE") -> str:
    row = df[(df['symbol'] == symbol) & (df['exch_seg'] == exchange)]
    if row.empty:
        raise ValueError(f"Token not found for {symbol} on {exchange}")
    return row.iloc[0]['token']

# Usage
df_instruments = get_angel_instrument_df()
reliance_token = get_token(df_instruments, "RELIANCE-EQ", "NSE")  # returns "2885"
nifty_fut_token = get_token(df_instruments, "NIFTY25JANFUT", "NFO")
```

**WebSocket (SmartWebSocketV2) format:**
```python
from smartapi.smartWebSocketV2 import SmartWebSocketV2

# Subscription request format
correlation_id = "algo_stream_1"
action = 1  # 1=subscribe, 0=unsubscribe
mode = 3    # 1=LTP, 2=Quote, 3=SnapQuote (full depth)

token_list = [
    {"exchangeType": 1, "tokens": ["3045", "2885"]},  # exchangeType: 1=NSE, 2=NFO, 3=BSE, 4=MCX
]
sws.subscribe(correlation_id, mode, token_list)

# Incoming tick JSON structure (SnapQuote mode):
{
  "subscription_mode": 3,
  "exchange_type": 1,
  "token": "3045",
  "sequence_number": 12345678,
  "exchange_timestamp": 1705312930000,  # Unix ms
  "last_traded_price": 2245050,         # DIVIDED BY 100 to get actual price
  "last_traded_quantity": 1,
  "average_traded_price": 2238025,
  "volume_trade_for_the_day": 12345678,
  "total_buy_quantity": 500000.0,
  "total_sell_quantity": 480000.0,
  "open_price_of_the_day": 2230000,
  "high_price_of_the_day": 2250000,
  "low_price_of_the_day": 2225000,
  "closed_price": 2235000,
  "best_5_buy_data": [
    {"flag": 0, "quantity": 150, "price": 2244900, "no_of_orders": 3}
  ],
  "best_5_sell_data": [
    {"flag": 0, "quantity": 75, "price": 2245100, "no_of_orders": 2}
  ]
}
# NOTE: All prices from Angel One WS are in paise (1/100 rupee). Divide by 100.
```

**Common Angel One errors and fixes:**
```
AB1010: Invalid Token            → Token expired; re-run generateSession()
AB1004: Rate limit exceeded      → Sleep 50ms between REST calls; max 20/sec
AB2000: Invalid symboltoken      → Re-download OpenAPIScripMaster.json (updated daily)
AB1001: Session expired          → Add daily auto-login before 9:00 AM pre-open
AB6000: Order rejected (margin)  → Check margin via obj.rmsLimit() before placing
Invalid format for fromdate      → Use exact format "YYYY-MM-DD HH:MM" (no seconds)
```

---

### Dhan (DhanHQ) — Technical Depth

**Order book depth API (REST v2):**
```python
import dhanhq

dhan = dhanhq.dhanhq("client_id", "access_token")

# Full market depth (up to 200 levels)
depth = dhan.get_market_depth(
    security_id="1333",              # Dhan's internal security ID
    exchange_segment=dhan.NSE        # dhan.NSE, dhan.BSE, dhan.NFO, dhan.MCX
)
# Response includes up to 200 bid/ask levels — unique to Dhan
# Keys: buy_quantity, sell_quantity, buy_orders, sell_orders, buy_price, sell_price
```

**Option chain endpoint:**
```python
# Dhan option chain — Greeks included
option_chain = dhan.option_chain(
    UnderlyingScrip="13",            # 13=Nifty50, 25=BankNifty
    UnderlyingSeg="IDX_I",
    Expiry="2025-01-30"              # YYYY-MM-DD format
)
# Response includes: strike, CE/PE LTP, IV, delta, gamma, theta, vega, OI, volume
```

**Authentication:**
```python
# Dhan uses access token (generated from developer portal, valid 30 days)
# No TOTP during API calls — only portal login requires TOTP
import dhanhq
dhan = dhanhq.dhanhq("your_client_id", "your_access_token")
# Regenerate access token from https://dhanhq.co/developer before expiry
```

**Common Dhan errors:**
```
DH-901: Invalid access token    → Token expired (30-day validity); regenerate from portal
DH-904: Rate limit              → Max 1000 req/10min on historical; add sleep(0.6) per call
DH-908: Invalid security_id     → Download updated instrument list from Dhan developer portal
```

---

### Shoonya/Finvasia (NorenAPI) — Technical Depth

**Authentication with TOTP:**
```python
import pyotp
from api_helper import ShoonyaApiPy

api = ShoonyaApiPy()

def shoonya_login(userid, password, vendor_code, api_secret, imei, totp_secret):
    factor2 = pyotp.TOTP(totp_secret).now()
    ret = api.login(
        userid=userid,
        password=password,
        twoFA=factor2,
        vendor_code=vendor_code,
        api_secret=api_secret,
        imei=imei            # any string, e.g. "abc1234"
    )
    if ret is None:
        raise ConnectionError("Shoonya login returned None")
    return api
```

**NorenAPI differences from REST-style APIs:**
```python
# Shoonya uses a proprietary NorenAPI format — NOT standard REST JSON
# All calls go through a single endpoint with "jKey" (session token)

# Historical data
history = api.get_time_price_series(
    exchange="NSE",
    token="22",               # 22 = Nifty 50 index
    starttime=1704067200,     # Unix timestamp (seconds, NOT milliseconds)
    endtime=1704153600,
    interval="15"             # in minutes: "1", "3", "5", "10", "15", "30", "60", "D"
)
# Returns: list of dicts with keys: ssboe (timestamp), into, inth, intl, intc, intv, intoi

# Key difference: timestamps are strings of Unix seconds (not datetime objects)
# Convert: pd.to_datetime(df['ssboe'].astype(int), unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
```

**WebSocket tick format (Shoonya):**
```json
{
  "t": "tf",
  "e": "NSE",
  "tk": "22",
  "lp": "22450.50",
  "pc": "0.45",
  "v": "123456789",
  "bp1": "22449.00", "bq1": "150", "bo1": "3",
  "sp1": "22451.00", "sq1": "75",  "so1": "2",
  "bp2": "22448.50", "bq2": "225", "bo2": "5",
  "sp2": "22451.50", "sq2": "150", "so2": "4",
  "o": "22300.00", "h": "22500.00", "l": "22250.00", "c": "22350.00",
  "oi": "45678900"
}
```
Keys: `t`=type(tf=tick), `lp`=last price, `pc`=percent change, `v`=volume, `bp/bq/bo`=bid price/qty/orders, `sp/sq/so`=ask

**Zero brokerage math — when Shoonya beats Zerodha:**
```
Zerodha cost per round-trip (buy+sell):
  Brokerage:      ₹20 + ₹20 = ₹40
  STT (0.025%):   on sell side only for intraday
  Exchange:       ~0.00345% each side
  GST (18%):      on brokerage + exchange charges
  
Shoonya cost per round-trip:
  Brokerage:      ₹0
  STT:            same as Zerodha (regulatory, cannot avoid)
  Exchange:       same as Zerodha (regulatory, cannot avoid)
  GST:            18% only on exchange charges (brokerage = 0)

Break-even trade size (Shoonya saves ≥ ₹40 vs Zerodha):
  Every single trade saves ₹40 fixed regardless of size.
  At 10 round-trips/day: ₹400/day, ₹8,800/month, ₹1,05,600/year savings.
  At ₹10L capital: that's 10.56% annual return just from cost savings.
  
Shoonya ALWAYS beats Zerodha on brokerage if you trade >0 times.
The ₹500/month Kite API fee adds to Zerodha's disadvantage.
```

**Common Shoonya errors:**
```
"Not logged in"              → Session expired; re-login (Shoonya sessions last ~24hrs)
None returned from API       → Network timeout or invalid token; add retry logic
Empty history data           → Holiday or market closed; check NSE calendar
"scrip not found"            → Token format wrong; use api.searchscrip() to find correct token
```

---

### TOTP Automation — All Brokers

**Generating TOTP secret (one-time setup):**
```python
# When broker gives you QR code for TOTP app setup:
# 1. Use a QR decoder to extract the otpauth:// URL
# 2. The 'secret' parameter IS your TOTP_SECRET (base32 encoded)
# Example URL: otpauth://totp/AngelOne:client123?secret=JBSWY3DPEHPK3PXP&issuer=AngelOne
# TOTP_SECRET = "JBSWY3DPEHPK3PXP"

import pyotp
totp = pyotp.TOTP("JBSWY3DPEHPK3PXP")
print(totp.now())           # 6-digit OTP valid for 30 seconds
print(totp.at(0))           # OTP for a specific counter value
print(totp.verify("123456"))  # Verify an OTP string
```

**pyotp installation and usage:**
```bash
pip install pyotp
```

**TOTP timing window — critical for automation:**
```python
import pyotp, time

def get_otp_with_buffer(secret: str, min_remaining_seconds: int = 5) -> str:
    """Wait if OTP is about to expire to avoid clock-skew rejection."""
    totp = pyotp.TOTP(secret)
    remaining = 30 - (int(time.time()) % 30)
    if remaining < min_remaining_seconds:
        time.sleep(remaining + 1)  # wait for next OTP
    return totp.now()
```

---

## SECTION 13 — Indian Market Data Sources (Deep Technical Expansion)

### NSE Bhavcopy — Exact URLs

**Equity bhavcopy (daily OHLCV for all NSE stocks):**
```
URL pattern:
https://archives.nseindia.com/content/historical/EQUITIES/{YEAR}/{MONTH_ABBR}/cm{DD}{MON}{YYYY}bhav.csv.zip

Examples:
Jan 15, 2025: https://archives.nseindia.com/content/historical/EQUITIES/2025/JAN/cm15JAN2025bhav.csv.zip
Mar 31, 2025: https://archives.nseindia.com/content/historical/EQUITIES/2025/MAR/cm31MAR2025bhav.csv.zip

Month abbreviations: JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
```

**Python download function:**
```python
import requests, zipfile, io, pandas as pd
from datetime import date

def download_equity_bhavcopy(dt: date) -> pd.DataFrame:
    month_abbr = dt.strftime("%b").upper()   # JAN, FEB, etc.
    day_str    = dt.strftime("%d")           # 01, 15, 31
    year_str   = dt.strftime("%Y")
    url = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
           f"{year_str}/{month_abbr}/cm{day_str}{month_abbr}{year_str}bhav.csv.zip")
    
    headers = {"User-Agent": "Mozilla/5.0"}  # NSE blocks default requests user-agent
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        df = pd.read_csv(z.open(csv_name))
    return df
    # Columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL,
    #          TIMESTAMP, TOTALTRADES, ISIN, Unnamed:13
```

**F&O bhavcopy (daily futures and options settlement):**
```
Futures: https://archives.nseindia.com/content/historical/DERIVATIVES/{YEAR}/{MONTH_ABBR}/fo{DD}{MON}{YYYY}bhav.csv.zip
Options daily: Same URL pattern but file inside zip differs

Alternate (combined F&O): 
https://www.nseindia.com/api/reports?archives=[{"name":"F%26O%20Bhavcopy","type":"daily","category":"derivatives","section":"equity"}]&date={DD-Mon-YYYY}&type=daily&category=derivatives

Simpler direct download (F&O bhavcopy):
https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
Example: https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20250115_F_0000.csv.zip
```

**F&O bhavcopy columns:**
```
FinInstrmTp (EQ/FUT/OPT), TckrSymb, XpryDt, StrkPric, OptnTp (CE/PE),
OpnPric, HghPric, LwPric, ClsPric, SttlmPric, TtlTradgVol, TtlTrfVal, OpnIntrst
```

---

### NSE Option Chain — Unofficial API

**Endpoint (unofficial but stable since 2019):**
```
https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY
https://www.nseindia.com/api/option-chain-equities?symbol=RELIANCE
```

**Headers required (must match browser session):**
```python
import requests

def get_nse_option_chain(symbol: str = "NIFTY") -> dict:
    session = requests.Session()
    # Step 1: Prime cookies by hitting homepage
    session.get("https://www.nseindia.com", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    })
    # Step 2: Fetch option chain
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    resp = session.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com/option-chain",
        "Accept": "application/json, text/plain, */*"
    })
    return resp.json()

# Response structure:
# data['records']['data'] → list of strikes
# Each strike: {'strikePrice': 22000, 'expiryDate': '30-Jan-2025',
#   'CE': {'openInterest': 12345, 'changeinOpenInterest': 234, 'iv': 12.5,
#          'lastPrice': 450.0, 'totalTradedVolume': 9876, 'delta': 0.45, 'gamma': 0.002, ...},
#   'PE': {...same keys...}}
```

---

### FII/DII Daily Data — Exact URLs

**NSE official provisional data (~6:30 PM daily):**
```
https://www.nseindia.com/api/fiidiiTradeReact
Response: JSON with FII and DII buy/sell values for current day (equity + derivative)
```

**Download daily FII/DII programmatically:**
```python
def get_fii_dii_today() -> dict:
    session = requests.Session()
    session.get("https://www.nseindia.com", headers={"User-Agent": "Mozilla/5.0 ..."})
    data = session.get("https://www.nseindia.com/api/fiidiiTradeReact").json()
    # Returns list of dicts: category (FII/FPI or DII), buyValue, sellValue, netValue
    return {item['category']: item for item in data}
```

**Historical FII data (date range):**
```
https://www.nseindia.com/api/historical/fiidii?from=01-01-2025&to=31-01-2025
```

**NSDL FPI final data:**
```
https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx
Direct CSV (monthly):
https://www.fpi.nsdl.co.in/web/Reports/ReportDetail.aspx?RepCode=RDAILYRPT
```

---

### Bulk and Block Deals — Exact URLs

**Bulk deals (current day):**
```
https://www.nseindia.com/api/bulk-deals
```

**Block deals (current day):**
```
https://www.nseindia.com/api/block-deals
```

**Historical bulk deals:**
```
https://archives.nseindia.com/content/equities/bulk.csv
(Single file with all historical bulk deals — refreshed daily)
```

**Download and filter:**
```python
def get_bulk_deals() -> pd.DataFrame:
    df = pd.read_csv(
        "https://archives.nseindia.com/content/equities/bulk.csv",
        parse_dates=['Date']
    )
    # Columns: Date, Symbol, Name, Type (BUY/SELL), Quantity, Price, Exchange
    return df
```

---

### jugaad-data — Complete API Reference

**Installation:**
```bash
pip install jugaad-data
```

**NSE Equity — full API:**
```python
from jugaad_data.nse import stock_df, index_df, NSELive

# Historical OHLCV for any NSE stock
df = stock_df(
    symbol="RELIANCE",
    from_date=date(2024, 1, 1),
    to_date=date(2025, 1, 1),
    series="EQ"                   # EQ (equity), BE (trade-to-trade), BL, etc.
)
# Returns DataFrame with: CH_TIMESTAMP, CH_OPENING_PRICE, CH_TRADE_HIGH_PRICE,
# CH_TRADE_LOW_PRICE, CH_CLOSING_PRICE, CH_TOT_TRADED_QTY, CH_TOT_TRADED_VAL

# Historical index data
df_nifty = index_df(
    symbol="Nifty 50",
    from_date=date(2024, 1, 1),
    to_date=date(2025, 1, 1)
)

# Live quotes
live = NSELive()
quote = live.stock_quote("RELIANCE")    # full live quote with 52w high/low, circuit limits
indices = live.live_indices()           # all indices live
```

**NSE F&O — bhavcopy and positions:**
```python
from jugaad_data.nse import NSELive, index_df
from jugaad_data.nse.archives import NSEArchives

arch = NSEArchives()

# F&O bhavcopy for a date
fo_data = arch.fo_bhavcopy(date(2025, 1, 15))
# Returns DataFrame with futures and options settlement prices

# Participant-wise open interest (FII/DII/Client/Prop positions in F&O)
# Access from: https://www.nseindia.com/api/historical/partitionedInterest
```

**BSE data via jugaad-data:**
```python
from jugaad_data.bse import stock_df as bse_stock_df

df = bse_stock_df(
    code="500325",           # BSE numeric code (not symbol)
    from_date=date(2024, 1, 1),
    to_date=date(2025, 1, 1)
)
```

**Rate limiting with jugaad-data:**
```python
# jugaad-data has built-in rate limiting for NSE requests
# For bulk downloads, NSE may block; add delay:
import time
for symbol in nifty50_list:
    df = stock_df(symbol, from_date, to_date)
    time.sleep(0.5)  # 500ms between requests to avoid 429 errors
```

---

## SECTION 14 — SEBI Regulations 2025–2026 (Deep Technical Expansion)

### Circular SEBI/HO/MIRSD/MIRSD-PoD/P/2025/0000013 — What It Actually Requires

**Full title:** "Safer Participation of Retail Investors in Algorithmic Trading"
**Date:** February 4, 2025
**Effective (final phase):** April 1, 2026

**Core problem SEBI is solving:**
Retail investors were running automated scripts via broker APIs without any oversight. These scripts caused errant orders, market disruptions, and losses. SEBI's circular creates a registration and audit framework.

**Who is covered:**
- Any person using API-based automated order placement (regardless of order frequency)
- "Algo trading" defined as: any software that automatically generates or routes orders without manual intervention per trade
- Explicitly includes Python scripts using broker APIs

**What the circular mandates — exact requirements:**

1. **Broker Registration of API Keys:**
   - Every API key must be registered with the broker
   - Broker validates and maintains records of all API users
   - Automatic deactivation if registered details change

2. **Strategy Registration (for >10 OPS):**
   - Strategies generating >10 orders/second require exchange-level approval
   - Strategy must have unique Strategy ID assigned by exchange
   - Strategy ID format: alphanumeric, assigned by NSE/BSE after review

3. **Static IP Mandate:**
   - API calls must originate from pre-registered static IPv4 address
   - IP changes require broker approval (max 1 change/week)
   - Dynamic IPs are non-compliant as of April 1, 2026

4. **Audit Trail Requirements:**
   - All API orders must include timestamps accurate to millisecond
   - Records must be maintained for minimum 5 years
   - Must be available for SEBI inspection within 24 hours of request

5. **Kill Switch:**
   - Every algo deployment must have a tested kill switch
   - Must halt all orders within 30 seconds of activation
   - Broker must be able to activate remotely

6. **Two-Factor Authentication:**
   - All API sessions require 2FA (TOTP acceptable)
   - Session tokens must expire daily

---

### August 1, 2025 Deadline (Original, Extended to April 1, 2026)

**What was required by Aug 1, 2025 (Phase 1–2):**
- Brokers: Complete infrastructure for API key registration
- Traders: Register API key with broker (static IP association)
- No strategy approval required yet in Phase 1

**Steps to register algo strategy with NSE/BSE (Phase 3 onward):**
```
Step 1: Register with broker
  → Login to broker portal (Shoonya — primary; or Zerodha / Angel One if applicable)
  → Navigate to: Developer > API Settings > Algo Registration
  → Submit: API key, static IP, strategy description, risk parameters

Step 2: Broker submits to exchange
  → Broker bundles all customer strategy registrations
  → Submits to NSE/BSE via Exchange portal (SEBI mandate for brokers)

Step 3: Exchange assigns Strategy ID
  → NSE format: {BROKER_CODE}-{YEAR}-{SEQ_NUM} (e.g., "ANGEL-2025-00234")
  → Communicated to trader via email + broker portal within 5-10 business days

Step 4: Embed Strategy ID in every API order
  → See code examples below
```

---

### Strategy ID — Format and Embedding in API Calls

**Strategy ID format:**
```
Exchange-assigned alphanumeric code
Example: "NSE-ALGO-2025-00234" or broker-specific format
Length: typically 10-20 characters
Case: alphanumeric, uppercase
```

**Embedding Strategy ID in Angel One orders:**
```python
order_params = {
    "variety": "NORMAL",
    "tradingsymbol": "RELIANCE-EQ",
    "symboltoken": "2885",
    "transactiontype": "BUY",
    "exchange": "NSE",
    "ordertype": "LIMIT",
    "producttype": "INTRADAY",
    "duration": "DAY",
    "price": "2500.00",
    "quantity": "1",
    "triggerprice": "0",
    "squareoff": "0",
    "stoploss": "0",
    "tag": "NSE-ALGO-2025-00234"   # Strategy ID goes in 'tag' field (max 20 chars)
    # Angel One uses 'tag' field for strategy identification
    # This appears in audit trail and order book
}
order_id = obj.placeOrder(order_params)
```

**Embedding Strategy ID in Zerodha orders:**
```python
order_id = kite.place_order(
    tradingsymbol="RELIANCE",
    exchange=kite.EXCHANGE_NSE,
    transaction_type=kite.TRANSACTION_TYPE_BUY,
    quantity=1,
    order_type=kite.ORDER_TYPE_LIMIT,
    product=kite.PRODUCT_MIS,
    price=2500.0,
    tag="NSE-ALGO-2025-00234"       # 'tag' field in Zerodha — max 8 chars (Kite limitation)
    # Note: Zerodha tag limit is 8 characters; use abbreviated strategy code
)
```

---

### Order-to-Trade Ratio (OTR) — NSE Limit

**NSE mandated limit: 500:1 per session**
- Meaning: for every 500 orders placed, at least 1 must result in a trade
- Counted per trading session (9:15 AM to 3:30 PM)
- Separate counters for each exchange (NSE and BSE)
- Penalty: exchange-imposed fine + possible suspension for persistent violators

**OTR monitoring code:**
```python
class OTRMonitor:
    def __init__(self, limit: int = 450):  # keep buffer below 500
        self.orders_placed = 0
        self.trades_executed = 0
        self.limit = limit

    def record_order(self):
        self.orders_placed += 1
        self._check_ratio()

    def record_trade(self):
        self.trades_executed += 1

    def _check_ratio(self):
        if self.trades_executed == 0:
            if self.orders_placed > 50:  # early warning before ratio becomes meaningful
                ratio = self.orders_placed
            else:
                return
        else:
            ratio = self.orders_placed / self.trades_executed
        if ratio > self.limit:
            raise RuntimeError(f"OTR {ratio:.0f} exceeds limit {self.limit}. Halt new orders.")

    def reset_session(self):
        self.orders_placed = 0
        self.trades_executed = 0
```

---

### Empaneled Platforms (Already Compliant)

| Platform | Type | Link | Status |
|----------|------|------|--------|
| Streak | No-code algo builder | https://streak.tech | Empaneled with Zerodha, Angel One |
| Sensibull | Options analytics + execution | https://sensibull.com | Empaneled (NSE registered) |
| Tradetron | Strategy marketplace | https://tradetron.tech | Empaneled |
| AlgoBulls | Institutional-grade | https://algobulls.com | Empaneled |
| Quantman | NSE-certified platform | https://quantman.in | Empaneled |
| uTrade Algos | White-label | https://utradealgos.com | Empaneled |

**Using an empaneled platform vs DIY:**
- Empaneled platforms handle SEBI registration on your behalf
- DIY (own Python scripts) requires self-registration via broker portal
- Empaneled platforms charge subscription fees (₹500–₹5,000/month)

---

### SHAP Documentation Requirement

**What SEBI requires for ML-based strategies:**
SEBI's operational modalities (NSE circular Jul 22, 2025) require algo traders to document the "logic and decision criteria" of their strategy. For ML models, SHAP (SHapley Additive exPlanations) satisfies this requirement by providing interpretable feature attribution.

**SHAP documentation to prepare:**
```python
import shap, mlflow

def generate_shap_documentation(model, X_train, X_test, feature_names, strategy_id):
    """Generate SEBI-compliant SHAP documentation for ML strategy."""
    
    # 1. SHAP explainer
    explainer = shap.TreeExplainer(model)  # for XGBoost/LightGBM/CatBoost
    shap_values = explainer.shap_values(X_test)
    
    # 2. Global feature importance plot (save as PNG for regulator)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(f"shap_global_{strategy_id}.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # 3. Feature importance table (for written documentation)
    mean_abs_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    mean_abs_shap.to_csv(f"shap_importance_{strategy_id}.csv", index=False)
    
    # 4. Log to MLflow for audit trail
    with mlflow.start_run():
        mlflow.log_artifact(f"shap_global_{strategy_id}.png")
        mlflow.log_artifact(f"shap_importance_{strategy_id}.csv")
        mlflow.set_tag("strategy_id", strategy_id)
        mlflow.set_tag("sebi_compliant_doc", "true")
    
    return mean_abs_shap

# What to submit to broker/SEBI:
# 1. Strategy description (plain English, 1-2 pages)
# 2. SHAP global importance plot showing top features
# 3. SHAP importance table with feature names and relative weights
# 4. Backtest performance summary (not source code)
# 5. Risk parameter table: max position size, stop-loss rules, kill switch logic
```

---

## SECTION 15 — Indian Market Characteristics (Deep Technical Expansion)

### NSE Trading Hours — Exact Schedule

| Time (IST) | Session | Technical Detail |
|------------|---------|-----------------|
| 09:00:00 | Pre-open start | Order entry opens; no matching |
| 09:00–09:08 | Order collection | Only limit/market orders accepted; cancel/modify allowed |
| 09:08:00 | IEP calculation begins | NSE computes Indicative Equilibrium Price (IEP) |
| 09:08–09:12 | Order matching at IEP | All orders matched at single equilibrium price |
| 09:12–09:15 | Buffer period | Pre-open orders not matched transition to continuous |
| 09:15:00 | Continuous trading starts | Normal bid-ask matching, price priority then time priority |
| 09:15–15:30 | Continuous session | Standard OHLCV data recorded in real time |
| 15:20:00 | Power hour end warning | Start reducing intraday positions |
| 15:30:00 | Closing call auction | 3:30–3:40 order collection for closing price |
| 15:30–15:40 | Call auction matching | NSE computes closing auction price |
| 15:40:00 | Official close | All unexecuted IOC/day orders cancelled |
| 15:40–16:00 | Post-close session | T+1 security borrowing, post-trade processing |

**IEP (Indicative Equilibrium Price) — what algo traders need to know:**
- IEP is the price that maximizes matched volume in pre-open
- First candle of the day opens AT the IEP (not the closing price of previous day)
- Gap between previous close and IEP = overnight gap
- High IEP volume vs previous OI = directional signal for opening trade

**Friday special sessions (F&O expiry weeks):**
- No change to session times
- Last 30 min (3:00–3:30 PM): Settlement price computed as volume-weighted average
- Weekly options expiry: position auto-square-off by broker at 3:30 PM for OTM options

---

### Tick Sizes by Price Band

| Price Range (₹) | Tick Size (₹) | Minimum Move |
|-----------------|---------------|--------------|
| 0.01 – 10.00 | 0.01 | 0.01 |
| 10.01 – 25.00 | 0.05 | 0.05 |
| 25.01 – 50.00 | 0.05 | 0.05 |
| 50.01 – 100.00 | 0.10 | 0.10 |
| 100.01 – 250.00 | 0.25 (some 0.05) | 0.05 or 0.25 |
| 250.01 – 500.00 | 0.25 | 0.25 |
| 500.01 – 1,000.00 | 0.25 | 0.25 |
| 1,000.01 – 2,500.00 | 0.25 | 0.25 |
| 2,500.01 and above | 0.25 | 0.25 |

**F&O tick sizes (different from equity):**
```
Nifty 50 futures/options: ₹0.05 per unit
BankNifty futures/options: ₹0.05 per unit
Stock futures:            ₹0.05 per unit (same as equity but enforced on futures price)
Currency futures (USD/INR): 0.0025 per unit
```

**Why tick size matters for algo:**
```python
def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to valid tick for NSE order placement."""
    return round(round(price / tick_size) * tick_size, 2)

# Critical: Angel One / Zerodha will REJECT orders with invalid tick sizes
# Error: "Price is not a valid multiple of tick size"
def get_tick_size(price: float) -> float:
    if price <= 10:    return 0.01
    if price <= 100:   return 0.05
    if price <= 250:   return 0.25   # approximate — verify per symbol
    return 0.25
```

---

### F&O Lot Sizes and Contract Specifications

| Instrument | Lot Size | Tick Size | Margin (approx) | Contract Value (Nifty ~22,500) |
|------------|----------|-----------|-----------------|-------------------------------|
| Nifty 50 | 75 | ₹0.05 | ~₹1,10,000 | ₹16,87,500 |
| BankNifty | 15 | ₹0.05 | ~₹50,000 | ~₹6,00,000 |
| FinNifty | 40 | ₹0.05 | ~₹30,000 | ~₹8,00,000 |
| MidcapNifty | 75 | ₹0.05 | ~₹1,00,000 | ~₹15,00,000 |
| Nifty Next 50 | 25 | ₹0.05 | ~₹75,000 | ~₹12,50,000 |
| Sensex (BSE) | 10 | ₹0.05 | ~₹1,20,000 | ~₹8,00,000 |
| Bankex (BSE) | 15 | ₹0.05 | ~₹60,000 | ~₹7,50,000 |

**Lot size change history note:** SEBI revised lot sizes effective Nov 2024 to ensure minimum contract value of ₹15L. Verify current lot sizes via:
```python
# From Angel One instrument master
df = get_angel_instrument_df()
nifty_row = df[(df['symbol'].str.contains('NIFTY')) & (df['exch_seg'] == 'NFO')].head()
print(nifty_row[['symbol', 'lotsize', 'tick_size', 'strike', 'expiry']])
```

---

### Expiry Structure — Complete Calendar Logic

**Weekly expiry schedule (as of 2025–2026):**
```
Nifty 50 weekly:     Every Thursday
BankNifty weekly:    Every Wednesday (changed from Thursday in 2024)
FinNifty weekly:     Every Tuesday
MidcapNifty weekly:  Every Monday
Sensex (BSE):        Every Friday
Bankex (BSE):        Every Monday

NOTE: Only ONE index weekly expiry per day (SEBI regulation from Oct 2023
to reduce speculative activity — each exchange limited to 1 weekly index expiry product)
As of 2025: NSE has Nifty (Thu), BankNifty (Wed), FinNifty (Tue), MidcapNifty (Mon)
```

**Monthly expiry:**
```
All monthly F&O contracts: Last Thursday of the month
If last Thursday is a holiday: penultimate Thursday
Monthly settlement: cash-settled at final settlement price (volume-weighted 3:00-3:30 PM)
```

**Expiry effects on trading — quantified:**
```
Day before weekly expiry (Nifty Thursday):
  - Implied volatility typically drops 15-25% post 2 PM (theta crush)
  - Volume spikes 40-60% above average in last 30 minutes
  - OTM options 0-2 strikes away from ATM have negative gamma risk

Monthly expiry Thursday:
  - Max OI pin: underlying tends to drift toward max pain price
  - Max pain calculation: price where sum of all ITM option losses is minimized
  - Gamma risk: market makers aggressively hedge delta near max pain

Code to find max pain:
```
```python
def calculate_max_pain(option_chain_df: pd.DataFrame) -> float:
    """Calculate max pain strike from option chain."""
    # option_chain_df: columns = strike, CE_OI, PE_OI
    strikes = option_chain_df['strike'].values
    pain = []
    for test_strike in strikes:
        # CE holder loss: sum of (test_strike - strike) * OI for all CE with strike < test_strike
        ce_pain = sum(max(test_strike - s, 0) * oi
                      for s, oi in zip(strikes, option_chain_df['CE_OI']))
        # PE holder loss: sum of (strike - test_strike) * OI for all PE with strike > test_strike
        pe_pain = sum(max(s - test_strike, 0) * oi
                      for s, oi in zip(strikes, option_chain_df['PE_OI']))
        pain.append(ce_pain + pe_pain)
    option_chain_df['total_pain'] = pain
    return option_chain_df.loc[option_chain_df['total_pain'].idxmin(), 'strike']
```

---

### Circuit Breakers — Complete Rules

**Stock-level circuit filters (individual stocks):**
| Band | Trigger | Trading Action |
|------|---------|----------------|
| 2% | Stock moves ±2% | No halt (just monitoring for illiquid stocks) |
| 5% | Stock moves ±5% | Pre-defined by NSE for specific stocks; 15-min cooling |
| 10% | Stock moves ±10% | 15-min halt on that stock, then resumes |
| 20% | Stock moves ±20% | Full-day halt on that stock |

**Which stocks get which band:**
- Nifty 50 / Nifty 100 stocks: 10% and 20% bands (no 5%)
- Midcap and smallcap: 5%, 10%, 20% bands
- Penny stocks / trade-to-trade segment: 2% or 5% band

**Index-level market-wide circuit breakers:**
| Index Move | Before 1:00 PM | 1:00 PM – 2:30 PM | After 2:30 PM |
|------------|----------------|---------------------|----------------|
| +/- 10% | 45 min halt | 15 min halt | Trading continues |
| +/- 15% | 1 hr 45 min halt | 45 min halt | Halt for remainder of day |
| +/- 20% | Halt for rest of day | Halt for rest of day | Halt for rest of day |

**Trigger instrument:** Either Nifty 50 OR BSE Sensex, whichever breaches first.

**Impact on algo systems:**
```python
# Handle circuit breaker in WebSocket feed
def on_tick(tick):
    if tick.get('circuit_breaker_status') in ('upper_circuit', 'lower_circuit'):
        logger.warning(f"Circuit breaker triggered: {tick['symbol']}")
        # Cancel all pending orders for this symbol immediately
        cancel_all_orders_for_symbol(tick['symbol'])
        # Mark symbol as untradeable for this session
        CIRCUIT_STOCKS.add(tick['symbol'])
        return
    if tick['symbol'] in CIRCUIT_STOCKS:
        return  # Skip processing
```

---

### NSE Holidays 2026 (Complete List)

| Date | Day | Holiday |
|------|-----|---------|
| Jan 26, 2026 | Monday | Republic Day |
| Feb 19, 2026 | Thursday | Chhatrapati Shivaji Maharaj Jayanti |
| Mar 20, 2026 | Friday | Gudi Padwa (Maharashtra) |
| Mar 31, 2026 | Tuesday | Id-Ul-Fitr (Ramzan Eid) — tentative |
| Apr 2, 2026 | Thursday | Ram Navami (tentative) |
| Apr 10, 2026 | Friday | Good Friday |
| Apr 14, 2026 | Tuesday | Dr. Ambedkar Jayanti / Baisakhi |
| May 01, 2026 | Friday | Maharashtra Day / Labour Day |
| Jun 16, 2026 | Tuesday | Eid-ul-Adha (Bakri Eid) — tentative |
| Aug 15, 2026 | Saturday | Independence Day (already a weekend) |
| Sep 07, 2026 | Monday | Ganesh Chaturthi (tentative) |
| Oct 02, 2026 | Friday | Gandhi Jayanti / Dussehra (tentative) |
| Oct 21, 2026 | Wednesday | Diwali Laxmi Puja (tentative) |
| Oct 22, 2026 | Thursday | Diwali Balipratipada (tentative) |
| Nov 05, 2026 | Thursday | Gurunanak Jayanti (tentative) |
| Dec 25, 2026 | Friday | Christmas |

**Note:** Tentative dates for Muslim and Hindu festivals subject to moon sighting. NSE publishes confirmed holiday list by October each year.
**Source:** https://www.nseindia.com/resources/exchange-communication-holidays

**Diwali Muhurat Trading:**
- NSE conducts a special 1-hour trading session on Diwali evening (~6:00–7:00 PM IST)
- This is a live session with real settlement; not paper trading
- Volume is low; treat as high-spread environment

---

### Seasonality — FII Flows, Expiry Effects, and Macro Patterns

**Union Budget seasonality (February):**
```
Pre-budget rally (Jan 20 – Feb 1):
  - Market typically rallies 2-4% in 2 weeks before budget
  - High volatility: India VIX rises 20-30% above normal
  - Strong sectors: Infrastructure, PSU banks, Defence

Budget day (Feb 1):
  - First 30 minutes: extreme volatility (2-4% swings common)
  - Strategy: avoid trading first 30 min; enter on first pullback after direction is clear
  - India VIX crushes 15-25% on budget day afternoon

Post-budget (Feb 2–15):
  - Reality check: rally or sell-off based on fiscal deficit number
  - Key metric: fiscal deficit as % of GDP vs estimate (>0.2% miss = negative)
```

**Quarterly expiry effects (March, June, September, December):**
```
Week before monthly expiry:
  - FII roll activity visible as futures OI declines and next-month OI rises
  - Rollover cost (basis) indicates market sentiment:
    * Positive basis (futures > spot): bullish carryover
    * Negative basis (futures < spot): bearish carryover

Expiry Thursday pattern:
  - First 30 min: extreme volatility as last-day options writers hedge
  - 11 AM – 2 PM: relatively calm (theta decay working)
  - 2:30 PM onward: pin to max pain; gamma squeeze possible
  - 3:00–3:30 PM: settlement window; DO NOT trade
```

**FII flow patterns by month:**
```
January:      Global rebalancing; FII often net sellers (tax-loss harvesting ends Dec)
February:     Budget uncertainty; FII hold; DII buying dominates
March:        Quarter-end FII outflows (profit booking); last week volatile
April:        FII return after budget clarity; typically positive month for large-caps
May:          "Sell in May" partial effect; reduced activity
June:         Pre-monsoon anxiety; commodity/agri stocks volatile
July-August:  Monsoon progress reports drive FMCG, agrochemicals, utilities
September:    US Fed meeting impact on FII flows (rate decisions = EM flows)
October:      Diwali effect; positive seasonality for equities
November:     FII rebalancing before year-end; SMIDs underperform
December:     Low volume; window dressing by mutual funds; last week illiquid
```

**Monsoon impact on NSE sectors:**
```
Above-normal monsoon (rainfall > 106% LPA):
  + Positive: FMCG (rural demand), Agro-chemicals, Fertilizers, 2-wheelers
  - Negative: Cement (construction delays), Infrastructure stocks (project delays)

Below-normal monsoon (rainfall < 96% LPA):
  + Positive: Irrigation companies, Agro-machinery
  - Negative: FMCG (rural slowdown), Microfinance NBFCs (agri loan stress)

Key monsoon dates for algo feature engineering:
  June 1:  IMD issues first monsoon forecast update
  June 5-8: Kerala landfall (advance vs normal = early signal)
  July 1:  Mid-season review (most impactful for market)
  Sept 30: Official end of monsoon season
  
IMD data URL: https://mausam.imd.gov.in/imd_latest/contents/monsoon_kharif.php
```

**FII activity measurement code:**
```python
def compute_fii_momentum(fii_df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Compute rolling FII net flow momentum.
    fii_df: columns = date, fii_net (positive=buying, negative=selling)
    """
    fii_df = fii_df.sort_values('date').set_index('date')
    rolling_net  = fii_df['fii_net'].rolling(window).sum()
    rolling_std  = fii_df['fii_net'].rolling(window).std()
    fii_z_score  = rolling_net / (rolling_std + 1e-9)
    return fii_z_score  # use as feature in ML model
    # Interpretation: > +1.5 = strong FII buying (bullish), < -1.5 = selling (bearish)
```

---

*End of B3 — Sections 11–15 Deep Expansion*

---

## B4: Sections 16–20 Deep Expansion

---

## SECTION 16 — Tax & Compliance for Algo Traders in India

### 16.1 Complete Charge Schedule (as of FY 2025-26)

#### Securities Transaction Tax (STT)

| Segment | Buy Side | Sell Side | Notes |
|---------|----------|-----------|-------|
| Equity Delivery | 0.1% | 0.1% | Both sides, charged on turnover |
| Equity Intraday | Nil | 0.025% | Sell side only |
| F&O Futures | Nil | 0.0125% | On premium value |
| F&O Options (Sell/Write) | Nil | 0.0625% | On premium received |
| F&O Options (Buy) | Nil | Nil | Buyer pays nothing |

STT is collected by the exchange and remitted to government. Non-deductible as business expense. Cannot be offset against capital gains tax.

#### Exchange Transaction Charges (NSE)

| Segment | Rate | Applied On |
|---------|------|-----------|
| Equity (Cash) | 0.00297% | Turnover both sides |
| Equity F&O (Futures) | 0.0019% | Turnover |
| Equity F&O (Options) | 0.053% | Premium turnover |

BSE rates differ slightly. If routing through BSE for any equity trades:
- BSE Equity: 0.00375% (higher than NSE)
- BSE F&O: Negligible (NSE dominates F&O volume 99%+)

#### SEBI Regulatory Fee

- Rate: 0.0001% on gross turnover (both sides, all segments)
- Revised upward from 0.00005% in 2023. Check SEBI website annually.

#### Stamp Duty (State-wise, post 2020 amendment)

- **Equity Delivery Buy:** 0.015%
- **Equity Intraday Buy:** 0.003%
- **F&O Buy (Futures):** 0.002%
- **F&O Buy (Options):** 0.003%
- Stamp duty applies on **buy side only**, capped per instrument
- Collected by exchange, remitted to investor's state government
- Rate shown is post-2020 uniform national rate; state variations exist for older trades

#### GST (Goods and Services Tax)

- Rate: 18% flat
- Applied on: Brokerage + Exchange Transaction Charges + SEBI charges combined
- Brokerage itself varies: Zero-brokerage brokers (Shoonya, Zerodha Flat) charge ₹0 but still pass through exchange + SEBI charges with 18% GST on those

---

### 16.2 Total Cost Formula — Full Derivation

**Variables:**
- `T` = Trade value (₹)
- `B` = Brokerage (₹)
- `ETC` = Exchange Transaction Charges (₹)
- `SEBI` = SEBI fee (₹)
- `STT` = Securities Transaction Tax (₹)
- `GST` = 18% × (B + ETC + SEBI)
- `SD` = Stamp duty (₹)

**Total Cost = STT + ETC + SEBI + B + GST + SD**

---

#### Worked Example: ₹1,00,000 Intraday Equity Trade (NSE, Zerodha ₹20 flat)

**Trade:** Buy ₹1L, Sell ₹1L (assuming flat price for simplicity)

| Charge | Calculation | Amount |
|--------|-------------|--------|
| STT (sell side only) | 0.025% × ₹1,00,000 | ₹25.00 |
| NSE Transaction (buy) | 0.00297% × ₹1,00,000 | ₹2.97 |
| NSE Transaction (sell) | 0.00297% × ₹1,00,000 | ₹2.97 |
| SEBI fee (buy) | 0.0001% × ₹1,00,000 | ₹0.10 |
| SEBI fee (sell) | 0.0001% × ₹1,00,000 | ₹0.10 |
| Stamp Duty (buy only) | 0.003% × ₹1,00,000 | ₹3.00 |
| Brokerage (buy + sell) | ₹20 × 2 | ₹40.00 |
| GST on (Brokerage + ETC + SEBI) | 18% × (₹40 + ₹5.94 + ₹0.20) | ₹8.31 |
| **TOTAL** | | **₹82.38** |

**Total cost as % of ₹1L trade = 0.0824%** (round trip)
**Breakeven move required: ~0.083%** (before slippage)
**With 1-tick slippage on ₹500 stock: ~0.10% round-trip**

For a 15-min candle strategy on Nifty stocks:
- Average intraday move per candle: 0.1–0.25%
- Net edge required: 0.10–0.15% per trade minimum just to cover costs
- This is why low-turnover strategies outperform in India vs. HFT

---

#### Worked Example: ₹1,00,000 F&O Trade (1 Nifty Futures lot, ~₹5L notional)

Assume: 1 Nifty lot = 25 units × 22,000 = ₹5,50,000 notional, using ₹1L margin

| Charge | Calculation | Amount |
|--------|-------------|--------|
| STT (sell side) | 0.0125% × ₹5,50,000 | ₹68.75 |
| NSE Transaction (buy+sell) | 0.0019% × ₹5,50,000 × 2 | ₹20.90 |
| SEBI fee (both sides) | 0.0001% × ₹5,50,000 × 2 | ₹1.10 |
| Stamp Duty (buy) | 0.002% × ₹5,50,000 | ₹11.00 |
| Brokerage (buy + sell) | ₹20 × 2 | ₹40.00 |
| GST | 18% × (₹40 + ₹20.90 + ₹1.10) | ₹11.16 |
| **TOTAL** | | **₹152.91** |

**As % of notional: 0.0278%** (much lower than intraday equity on notional basis)
**As % of margin deployed: 0.153%** (higher than equity on margin basis)

---

### 16.3 F&O vs Equity — Tax Efficiency for Profitable Algo Traders

**Equity Intraday:**
- Taxed as **speculative business income** (Section 43(5))
- Cannot be offset against F&O profits
- Losses: offset only against other speculative income; carry forward 4 years
- Rate: As per slab (30% for high-income traders)

**Equity Delivery:**
- Short-term (<12 months): 20% flat (revised 2024 budget, was 15%)
- Long-term (>12 months, gains > ₹1.25L): 12.5% flat (revised 2024, was 10%)

**F&O (Futures & Options):**
- Taxed as **non-speculative business income**
- Losses offset against ALL other income (salary, rental, etc.) — except salary in some cases
- Carry forward losses: 8 years
- Rate: As per slab
- **Section 44AB audit**: mandatory if turnover > ₹2 crore (or > ₹1 crore with cash receipts > 5%)
- Presumptive taxation under 44AD: 6% of turnover as deemed profit (only if turnover < ₹2 crore and you want simplicity)

**Bottom Line for Profitable Algo Trader:**
- F&O is superior for loss offsetting and carry-forward
- F&O losses reduce your total taxable income (including salary) — equity intraday losses do not
- If consistently profitable: effective tax rate identical (slab rate) regardless of segment
- Structural advantage of F&O: higher leverage means lower capital deployed for same notional → higher ROC

---

### 16.4 ITR Filing — F&O as Business Income

**Form: ITR-3** (individuals/HUF with business/professional income)
Not ITR-2 (capital gains only), not ITR-1 (salary only).

**Required Schedules in ITR-3:**
- Schedule BP (Business/Profession): declare F&O P&L
- Schedule CYLA: current year loss adjustment
- Schedule BFLA: brought-forward loss adjustment
- Schedule AL: assets and liabilities (if income > ₹50L)

**F&O Turnover Calculation (Critical — misunderstood by most):**
Turnover ≠ notional value traded.

```
F&O Turnover = Sum of |Profit or Loss| per contract settlement

Example:
Trade 1: Nifty Fut Buy 22000, Sell 22150 → Profit ₹3750 (25 × 150) → Add ₹3750
Trade 2: Nifty Fut Buy 22200, Sell 22050 → Loss ₹3750 (25 × 150) → Add ₹3750
Turnover = ₹3750 + ₹3750 = ₹7500

For Options:
Premium received on sale ALSO counts toward turnover
Options Turnover = |Net premium received/paid| + |Settlement P&L|
```

This is per ICAI guidance note on tax audit for derivatives traders.

**Practical implication:** A highly active algo trader churning ₹10 crore notional per month may have F&O turnover of only ₹20–50 lakh if win/loss per trade is small. Audit threshold (₹2 crore) based on this, not notional.

---

### 16.5 Advance Tax Schedule

If total estimated tax liability > ₹10,000 in a financial year, advance tax is mandatory:

| Installment | Due Date | Cumulative % |
|-------------|----------|-------------|
| 1st | June 15 | 15% |
| 2nd | September 15 | 45% |
| 3rd | December 15 | 75% |
| 4th | March 15 | 100% |

**Interest for non-payment:**
- Section 234B: 1% per month if you pay <90% of total tax by March 31
- Section 234C: 1% per month for each installment shortfall

**Practical example:** Algo trader earning ₹15L net from F&O in FY 2025-26:
- Taxable income (assuming no other income deductions): ₹15L
- Tax at slab: ~₹3.0L (under new regime: ₹0 up to ₹3L, then 5%/10%/15%/20%)
- Since tax > ₹10K: must pay advance tax
- Jun 15: ₹45,000 | Sep 15: ₹1,35,000 | Dec 15: ₹2,25,000 | Mar 15: ₹3,00,000

---

### 16.6 Books of Accounts — Requirements

| Turnover | Audit Requirement | Books Required |
|----------|-------------------|----------------|
| < ₹1 crore | No audit | Basic P&L + balance sheet recommended |
| ₹1–2 crore (cash receipts <5%) | No audit (44AD presumptive possible) | Basic books |
| > ₹2 crore | Section 44AB audit mandatory | Full double-entry accounts |
| Any turnover, loss declared | Audit recommended | Full books to substantiate |

**Minimum books for algo trader (regardless of turnover):**
1. Cash book / bank statement
2. Trade ledger (export from broker — all trades)
3. P&L statement (realized + unrealized)
4. Brokerage and charge receipts

**Recommendation:** Export complete trade history from broker platform annually in CSV. Store with contract notes (PDF). Use a CA familiar with derivatives taxation.

---

### 16.7 cost_model.py Implementation

```python
from dataclasses import dataclass

@dataclass
class IndiaCostModel:
    """Production-grade India trade cost model. All rates as of FY 2025-26."""

    # Rates
    stt_intraday_sell: float = 0.00025      # 0.025%
    stt_delivery_both: float = 0.001        # 0.1% each side
    stt_fo_futures_sell: float = 0.000125   # 0.0125%
    stt_fo_options_sell: float = 0.000625   # 0.0625% (writer)

    nse_eq_charge: float = 0.0000297        # 0.00297%
    nse_fo_futures_charge: float = 0.000019 # 0.0019%
    nse_fo_options_charge: float = 0.00053  # 0.053%

    sebi_fee: float = 0.000001              # 0.0001%
    stamp_intraday: float = 0.00003         # 0.003%
    stamp_fo_futures: float = 0.00002       # 0.002%
    stamp_fo_options: float = 0.00003       # 0.003%
    gst_rate: float = 0.18

    brokerage_per_order: float = 0.0        # Shoonya: zero brokerage (set 20.0 for Zerodha/Angel One)

    slippage_ticks: int = 1                 # Conservative: 1 tick per side
    tick_size: float = 0.05                 # NSE equity default

    def compute_intraday_cost(self, trade_value: float, price: float) -> dict:
        """Compute total round-trip cost for one intraday equity trade."""
        stt = self.stt_intraday_sell * trade_value
        etc = self.nse_eq_charge * trade_value * 2
        sebi = self.sebi_fee * trade_value * 2
        stamp = self.stamp_intraday * trade_value
        brokerage = self.brokerage_per_order * 2
        gst = self.gst_rate * (brokerage + etc + sebi)
        slippage = 2 * self.slippage_ticks * self.tick_size * (trade_value / price)

        total = stt + etc + sebi + stamp + brokerage + gst + slippage
        return {
            "stt": round(stt, 2),
            "exchange_charges": round(etc, 2),
            "sebi": round(sebi, 2),
            "stamp": round(stamp, 2),
            "brokerage": round(brokerage, 2),
            "gst": round(gst, 2),
            "slippage": round(slippage, 2),
            "total": round(total, 2),
            "total_pct": round(total / trade_value * 100, 4),
        }
```

---

## SECTION 17 — Free Learning Resources

### 17.1 Free Online Courses (Verified URLs)

#### Machine Learning & Quant Finance
| Course | Provider | URL | Why Relevant |
|--------|----------|-----|-------------|
| Machine Learning for Trading | Coursera (Google Cloud) | https://www.coursera.org/specializations/machine-learning-trading | 4-course specialization. Covers feature engineering, LSTM, portfolio optimization. Free to audit. |
| Practical Deep Learning for Coders | fast.ai | https://course.fast.ai | Jeremy Howard. Tabular models, time series in Part 2. Free, no signup. |
| Microsoft Qlib Tutorials | GitHub | https://github.com/microsoft/qlib/tree/main/examples | Jupyter notebooks for alpha mining, backtest, portfolio. Run locally. |
| Algorithmic Trading & Finance Models | Udemy-style free tier | https://quantra.quantinsti.com/courses | QuantInsti offers free intro courses. India-focused. |
| Introduction to Portfolio Construction | Coursera (EDHEC) | https://www.coursera.org/learn/introduction-portfolio-construction-python | Markowitz, HRP, Black-Litterman in Python. Audit free. |
| Time Series Analysis (Applied) | Kaggle Learn | https://www.kaggle.com/learn/time-series | Practical Python time series. 5 hours. Completely free. |
| FinRL Tutorial Notebooks | GitHub | https://github.com/AI4Finance-Foundation/FinRL/tree/master/tutorials | Step-by-step RL trading. Google Colab compatible. |

---

### 17.2 YouTube Channels (Quality-Ranked)

| Channel | Focus | Why Worth Watching |
|---------|-------|-------------------|
| **QuantProgram** | Python quant, India-focused | NSE backtesting, Angel One API integration, real code |
| **Algo Trader** | NSE/BSE live examples | Zerodha Kite API, options strategies, live P&L demos |
| **Systematic Investor** | Robert Carver content, trend following | Position sizing, diversification, practical Sharpe |
| **Lex Fridman** | ML interviews | Peter Lynch, quant hedge fund interviews. Conceptual. |
| **Two Sigma / Citadel YouTube** | N/A (search by name) | Recorded conference talks from institutional quants |
| **NPTEL Finance** | https://www.youtube.com/@iit | IIT professors on derivatives, financial modeling. Free. |
| **QuantLib Python** | https://www.youtube.com/@qlabschool | Derivatives pricing, Greeks. Relevant for options strategies. |

India-specific search terms that surface quality content:
- "NSE algo trading Python 2024"
- "Zerodha SmartAPI live trading"
- "Indian stock market machine learning backtest"

---

### 17.3 Reddit Communities

| Community | URL | Best Use |
|-----------|-----|----------|
| **r/algotrading** | https://www.reddit.com/r/algotrading | 120K+ members. Strategy discussion, code reviews, broker comparisons. Search "India" for India-specific threads. |
| **r/IndiaInvestments** | https://www.reddit.com/r/IndiaInvestments | Tax questions, broker comparisons, regulatory updates. Wiki has India-specific resources. |
| **r/quantfinance** | https://www.reddit.com/r/quantfinance | Academic quant discussion. Paper recommendations. |
| **Zerodha TradingQnA** | https://tradingqna.com | Official Zerodha Q&A. India-specific. Best for Kite API issues, tax questions, SEBI compliance. Active Zerodha team. |
| **r/learnmachinelearning** | https://www.reddit.com/r/learnmachinelearning | For ML fundamentals questions. Not finance-specific but high quality. |

**High-value r/algotrading threads to search:**
- "India broker API comparison"
- "walk-forward validation India"
- "NSE tick data free"
- "STT optimization F&O"

---

### 17.4 Telegram Groups for NSE Algo Traders

| Group Name | Type | Focus |
|-----------|------|-------|
| **Algo Trading India** | Public | General algo trading, NSE/BSE, strategy sharing |
| **Python for Trading India** | Public | Code-sharing, broker API issues, debugging |
| **NSE Options Flow** | Public | Options chain analysis, OI data, FII/DII data |
| **Zerodha Kite Developers** | Semi-private | Kite API issues, bug reports, workarounds |
| **Angel One SmartAPI** | Official | Angel One API announcements, downtime alerts |
| **Quantitative Finance India** | Public | Academic papers, strategy concepts, portfolio theory |

**Finding groups:** Search Telegram for "NSE Algo", "Indian algo trading", "Python trading India". Vet for spam — genuine groups don't push paid signals.

**Red flags in Telegram groups:** Anyone promising fixed returns, paid subscription for signals, or "proven strategy with 90% win rate" — these are scams.

---

### 17.5 NCFM/NISM Certifications Relevant for Algo Traders

| Certification | Body | URL | Relevance |
|---------------|------|-----|-----------|
| **NISM Series VIII: Equity Derivatives** | NISM | https://certifications.nism.ac.in | Mandatory for equity derivatives trading. Covers F&O mechanics, settlement, risk. |
| **NISM Series I: Currency Derivatives** | NISM | https://certifications.nism.ac.in | For currency F&O (USD/INR pairs) on NSE. |
| **NCFM: Technical Analysis Module** | NSE | https://ncfm.nseindia.com | Charting, indicators. Basic but covers exam-style knowledge. |
| **NCFM: Algorithmic Trading Strategies** | NSE | https://ncfm.nseindia.com | Specifically covers algo trading, co-location, risk management. Direct relevance. |
| **NISM Research Analyst** | NISM | https://certifications.nism.ac.in | If you publish research or advisory. Legally required for paid advisory. |

**Practical note:** NISM Series VIII is required by SEBI if you're managing others' money or providing advisory. For personal algo trading, no certification is legally required as of 2025 — but NCFM Algorithmic Trading module is worth completing for conceptual depth.

**Exam fees:** ₹1,500–2,500 per exam. Study material available free on NISM website.

---

### 17.6 Free Paper Trading Platforms

| Platform | URL | Features | India Support |
|---------|-----|----------|--------------|
| **Zerodha Streak** | https://streak.tech | Strategy builder, auto-backtesting, paper trading mode. Free tier available. | NSE/BSE |
| **TradingView Paper Trading** | https://www.tradingview.com | Built-in paper trading with realistic fills. No Indian broker connection needed. Free. | NSE charts |
| **Sensibull (practice)** | https://sensibull.com | Options strategy builder with virtual trading. Free tier. | NSE Options |
| **Upstox Virtual Trading** | https://upstox.com | Virtual trading account. Real market prices. Free. | NSE/BSE |
| **Angel One Paper Mode** | SmartAPI sandbox | Sandbox environment for API testing. No real orders. | Full NSE API |
| **Backtrader** | https://www.backtrader.com | Python library. Manual paper trading by running with live data feed in simulation. | Any feed |

**For serious paper trading:** Use your actual Python system in paper mode. Replace `place_order()` with a function that logs to SQLite instead of sending to broker. Run for 4+ weeks with real market data. This is the only meaningful test.

---

## SECTION 18 — Institutional Quant Strategies

### 18.1 Cross-Sectional Momentum (Jegadeesh-Titman)

**Original Paper:** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency" — Jegadeesh & Titman, Journal of Finance, 1993.

**Core Formula:**
```
rank_i = close_i[t] / close_i[t - 252]  # 12-month return, skip last 21 bars

# Python implementation:
import pandas as pd

def compute_momentum_rank(prices_df: pd.DataFrame) -> pd.Series:
    """
    prices_df: DataFrame with dates as index, symbols as columns.
    Returns cross-sectional momentum rank (1=strongest, N=weakest).
    """
    lookback = 252   # ~12 months of trading days
    skip = 21        # skip last 1 month (reversal effect)

    if len(prices_df) < lookback + skip:
        raise ValueError("Insufficient history")

    # Price at t-skip (avoid short-term reversal)
    price_now = prices_df.iloc[-(skip + 1)]
    # Price at t-lookback-skip
    price_past = prices_df.iloc[-(lookback + skip)]

    momentum = price_now / price_past - 1
    rank = momentum.rank(ascending=False)  # 1 = top momentum
    return rank
```

**India-specific notes:**
- 252 trading days ≈ 1 year on NSE (NSE typically 250-252 trading days)
- Skip period (21 days) is critical — short-term reversal is stronger in India than US due to lower liquidity
- Works best on Nifty 100 universe (sufficient liquidity)
- Monthly rebalancing standard; weekly increases transaction costs excessively

**Known risks:**
- Momentum crashes: during sharp market reversals (March 2020, Oct 2008), winners become losers abruptly
- Seasonality: January effect reversal can hurt January momentum entry
- India: Budget day (Feb 1) + RBI MPC announcements create momentum disruptions

---

### 18.2 Mean Reversion — Ornstein-Uhlenbeck Process

**Mathematical Definition:**
```
dX_t = θ(μ - X_t)dt + σdW_t

Where:
  X_t = price process (or spread)
  θ   = mean reversion speed (rate of pull toward μ)
  μ   = long-run mean
  σ   = diffusion coefficient (volatility)
  dW_t = Wiener process increment
```

**Half-Life Derivation:**
```
Half-life = ln(2) / θ

Interpretation:
  If θ = 0.1  → half-life = 6.93 days (fast mean reversion)
  If θ = 0.01 → half-life = 69.3 days (slow)
  
Practical threshold: Use if half-life < 20 trading days (otherwise signal decays too slowly)
```

**Calibrating OU Parameters from Price Data:**
```python
import numpy as np
from scipy.optimize import minimize

def calibrate_ou(prices: np.ndarray, dt: float = 1/252) -> dict:
    """
    Fit OU process to log-price series using discrete approximation.
    X_{t+1} = X_t + theta*(mu - X_t)*dt + sigma*sqrt(dt)*eps

    Discrete AR(1): X_{t+1} = a + b*X_t + eps
    a = theta*mu*dt, b = 1 - theta*dt
    """
    X = np.log(prices)
    n = len(X)

    # OLS regression of X_{t+1} on X_t
    X_t = X[:-1]
    X_t1 = X[1:]

    # Analytical OLS
    cov_matrix = np.cov(X_t, X_t1)
    b = cov_matrix[0, 1] / np.var(X_t)    # regression coefficient
    a = np.mean(X_t1) - b * np.mean(X_t)  # intercept

    theta = (1 - b) / dt
    mu = a / (theta * dt)
    residuals = X_t1 - (a + b * X_t)
    sigma = np.std(residuals) / np.sqrt(dt)
    half_life = np.log(2) / theta

    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life_days": half_life}
```

**Practical use:** Apply to pairs spread (Section 18.3). If half-life < 10 trading days → aggressive mean reversion strategy viable. If > 30 days → not suitable for intraday/swing; look for other pairs.

---

### 18.3 Statistical Arbitrage — Pairs Trading (Full Pipeline)

#### Step 1: Universe Selection
Start with stocks in the same sector (HDFC Bank + ICICI Bank, Infosys + TCS, ONGC + BPCL). Same-sector pairs have higher cointegration probability.

#### Step 2: Cointegration Test (Engle-Granger)

```python
from statsmodels.tsa.stattools import coint, adfuller
import pandas as pd
import numpy as np

def find_cointegrated_pairs(prices_df: pd.DataFrame,
                             significance: float = 0.05) -> list:
    """
    Engle-Granger cointegration test on all pairs in prices_df.
    Returns list of (stock1, stock2, p_value, beta) tuples.
    """
    symbols = prices_df.columns.tolist()
    cointegrated = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1 = np.log(prices_df[symbols[i]].dropna())
            s2 = np.log(prices_df[symbols[j]].dropna())

            # Align series
            s1, s2 = s1.align(s2, join='inner')

            score, pvalue, _ = coint(s1, s2)
            if pvalue < significance:
                # Compute hedge ratio beta via OLS
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant
                model = OLS(s1, add_constant(s2)).fit()
                beta = model.params[1]
                cointegrated.append((symbols[i], symbols[j], pvalue, beta))

    return sorted(cointegrated, key=lambda x: x[2])  # sort by p-value
```

#### Step 3: Spread Construction & Z-Score

```python
def compute_spread_zscore(s1: pd.Series, s2: pd.Series,
                           beta: float, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score of the pair spread.
    
    spread = log(S1) - beta * log(S2)
    z = (spread - rolling_mean) / rolling_std
    """
    spread = np.log(s1) - beta * np.log(s2)
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std
    return z_score
```

#### Step 4: Signal Generation

```python
ENTRY_THRESHOLD = 2.0   # |z| > 2.0 → enter
EXIT_THRESHOLD = 0.5    # |z| < 0.5 → exit
STOP_THRESHOLD = 3.5    # |z| > 3.5 → stop-loss (pair broke down)

def generate_pairs_signal(z_score: float, current_position: int) -> int:
    """
    Returns: +1 (long spread), -1 (short spread), 0 (flat)
    Long spread = long S1, short S2 (normalized by beta)
    """
    if current_position == 0:
        if z_score < -ENTRY_THRESHOLD:
            return 1   # spread too low → buy S1, sell S2
        elif z_score > ENTRY_THRESHOLD:
            return -1  # spread too high → sell S1, buy S2
    elif current_position != 0:
        if abs(z_score) < EXIT_THRESHOLD:
            return 0   # exit
        elif abs(z_score) > STOP_THRESHOLD:
            return 0   # stop-loss: spread diverging, not reverting
    return current_position
```

**India-specific considerations:**
- Corporate actions (dividends, splits) break cointegration temporarily → check for events
- NSE F&O expiry week: pairs spread widens due to rollover activity → avoid entry Thu/Fri expiry week
- Use liquid stocks only (avg daily volume > ₹50 crore) to ensure both legs fill simultaneously
- Circuit breakers: if one stock hits 10%/20% circuit, pair trade is stranded

---

### 18.4 Trend Following — ATR Channel & Donchian

#### Donchian Channel

```python
def donchian_channel(high: pd.Series, low: pd.Series,
                     period: int = 20) -> pd.DataFrame:
    """
    Upper = highest high over period
    Lower = lowest low over period
    Mid = (upper + lower) / 2
    
    Signal: Close > Upper → Long | Close < Lower → Short
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})
```

#### ATR Channel Breakout

```python
import pandas_ta as ta

def atr_channel_breakout(df: pd.DataFrame,
                          atr_period: int = 14,
                          atr_mult: float = 2.0,
                          ema_period: int = 20) -> pd.Series:
    """
    Channel: EMA ± atr_mult * ATR
    Entry: Close breaks above upper channel → Long
    Exit: Close breaks below EMA (center)
    """
    atr = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    ema = ta.ema(df['close'], length=ema_period)

    upper_channel = ema + atr_mult * atr
    lower_channel = ema - atr_mult * atr

    signal = pd.Series(0, index=df.index)
    signal[df['close'] > upper_channel] = 1    # Long signal
    signal[df['close'] < lower_channel] = -1   # Short signal

    return signal
```

**NSE-specific tuning:** For Nifty 50 stocks on 15-min candles, ATR(14) with multiplier 1.5–2.5 works better than the textbook 2.0. ATR period 14 captures roughly 3.5 hours of intraday movement. Donchian(20) on 15-min = 5 hours lookback.

---

## SECTION 19 — Portfolio Optimization

### 19.1 Markowitz Mean-Variance Optimization

**Sharpe Ratio:**
```
Sharpe = (E[r_p] - r_f) / σ_p

Where:
  E[r_p] = expected portfolio return
  r_f    = risk-free rate (India: ~6.5% annualized as of 2025, use 91-day T-bill)
  σ_p    = portfolio standard deviation (annualized)
```

**Optimization Problem:**
```
Maximize: Sharpe ratio (or equivalently, for given return, minimize variance)

max w^T μ - λ w^T Σ w   subject to:
  sum(w) = 1
  w_i >= 0 (long-only)
  w_i <= 0.2 (max 20% per stock)
```

**Key practical limitation:** Markowitz requires inverting the covariance matrix. For N stocks, matrix is N×N. With N=50 (Nifty 50) and 2 years of daily data (500 obs), the matrix is unstable (near-singular). Use shrinkage (Ledoit-Wolf) or HRP instead.

---

### 19.2 Kelly Criterion

**Full Kelly Formula:**
```
f* = (p × b - q) / b

Where:
  f* = fraction of capital to bet
  p  = probability of winning
  q  = 1 - p = probability of losing
  b  = win/loss ratio (average win / average loss)

Example:
  p = 0.55 (55% win rate)
  avg_win = ₹1500, avg_loss = ₹1000 → b = 1.5
  f* = (0.55 × 1.5 - 0.45) / 1.5 = (0.825 - 0.45) / 1.5 = 0.25

Full Kelly says: bet 25% of capital on each trade.
Half-Kelly: bet 12.5% — standard practice.
```

**Why Half-Kelly in practice:**
- Full Kelly maximizes geometric growth rate but has extreme drawdowns (50%+ common)
- Half-Kelly achieves ~75% of maximum growth with far less volatility
- Kelly assumes edge estimates are exact — they never are in practice
- Estimation error in p and b → full Kelly systematically overbets

```python
def kelly_fraction(win_rate: float, avg_win: float,
                   avg_loss: float, fraction: float = 0.5) -> float:
    """
    Compute Kelly position size fraction.
    fraction=0.5 → half-Kelly (recommended default).
    """
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    f_full = (win_rate * b - q) / b
    return max(0.0, f_full * fraction)  # never negative
```

**NSE constraint:** Even with Kelly suggesting 25%+ per trade, SEBI/broker margin rules typically force position limits. Hard cap all positions at 5% of capital regardless of Kelly output (per system risk rules in Section 15).

---

### 19.3 Hierarchical Risk Parity (HRP) — Step-by-Step

**Reference:** López de Prado (2016) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678

**Why HRP over Markowitz:**
- No matrix inversion → stable with many assets and short history
- Diversifies risk, not capital
- Works with 20+ assets where Markowitz breaks down

**Algorithm Steps:**

```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def hrp_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Full HRP algorithm.
    returns: DataFrame of asset returns (dates × assets)
    Returns: pd.Series of portfolio weights (sums to 1)
    """
    # Step 1: Compute correlation matrix
    corr = returns.corr()
    cov = returns.cov()

    # Step 2: Distance matrix from correlation
    # Distance = sqrt(0.5 * (1 - corr)) — ensures triangle inequality
    dist = np.sqrt(0.5 * (1 - corr))
    dist_condensed = squareform(dist.values)

    # Step 3: Hierarchical clustering (single linkage)
    link = linkage(dist_condensed, method='single')

    # Step 4: Quasi-diagonalize (sort assets by cluster)
    sorted_items = _get_quasi_diag(link, corr.shape[0])
    sorted_assets = [corr.columns[i] for i in sorted_items]

    # Step 5: Recursive bisection
    weights = _recursive_bisection(cov, sorted_assets)
    return weights


def _get_quasi_diag(link: np.ndarray, N: int) -> list:
    """Sort clustered items into quasi-diagonal order."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= N:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= N]
        i = df0.index
        j = df0.values - N
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _recursive_bisection(cov: pd.DataFrame, sorted_assets: list) -> pd.Series:
    """Assign weights via recursive bisection (risk-parity within each cluster)."""
    weights = pd.Series(1.0, index=sorted_assets)
    cluster_items = [sorted_assets]

    while len(cluster_items) > 0:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for subcluster in range(0, len(cluster_items), 2):
            left = cluster_items[subcluster]
            right = cluster_items[subcluster + 1]

            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)

            alloc_factor = 1 - left_var / (left_var + right_var)
            weights[left] *= alloc_factor
            weights[right] *= 1 - alloc_factor

    return weights


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    """Compute variance of inverse-volatility weighted sub-portfolio."""
    sub_cov = cov.loc[items, items]
    inv_vol = 1 / np.sqrt(np.diag(sub_cov.values))
    w = inv_vol / inv_vol.sum()
    return float(w @ sub_cov.values @ w)
```

---

### 19.4 Black-Litterman Model

**Posterior Return Formula:**
```
E[r | views] = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} × [(τΣ)^{-1}Π + P^T Ω^{-1} Q]

Where:
  Σ  = covariance matrix of returns
  τ  = scaling factor (typically 0.05–0.1; τ=1/T where T = number of observations)
  Π  = implied equilibrium returns = δΣw_mkt (CAPM-implied)
  δ  = market risk aversion coefficient (~2.5 for equity)
  w_mkt = market-cap weights
  P  = pick matrix (views: rows=views, cols=assets)
  Q  = view returns vector
  Ω  = uncertainty of views (diagonal matrix of variance of each view)
```

**Practical Python (PyPortfolioOpt):**
```python
from pypfopt import BlackLittermanModel, expected_returns, risk_models
from pypfopt.black_litterman import market_implied_risk_aversion
import pandas as pd

def black_litterman_nse(prices: pd.DataFrame,
                         mcap_weights: pd.Series,
                         ml_views: dict) -> pd.Series:
    """
    prices: daily closing prices, dates × symbols
    mcap_weights: market-cap weights for each stock (sum=1)
    ml_views: {symbol: expected_return} from ML model
    
    Example ml_views from XGBoost ensemble:
      {"RELIANCE": 0.02, "INFY": -0.01, "HDFCBANK": 0.015}
    """
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Market-implied risk aversion
    delta = market_implied_risk_aversion(prices.iloc[-1])

    # Prior: market equilibrium returns
    market_prior = BlackLittermanModel.market_implied_prior_returns(
        mcap_weights, delta, S
    )

    # View matrix: absolute views (P = identity for viewed assets)
    bl = BlackLittermanModel(
        S,
        pi=market_prior,
        absolute_views=ml_views,  # {asset: expected_return}
        tau=0.05,
    )

    bl_return = bl.bl_returns()
    bl_cov = bl.bl_cov()

    return bl_return, bl_cov
```

**How to generate ML views for BL:**
```python
# If XGBoost predicts +1 (up) for RELIANCE with probability 0.72:
# View = current_price × (0.72 - 0.5) × 2 × daily_vol_estimate
# Simplified: view = (prob - 0.5) * 2 * expected_daily_move

def ml_proba_to_bl_view(symbol: str, up_prob: float,
                          daily_vol: float) -> float:
    """Convert classifier probability to BL expected return view."""
    directional_signal = (up_prob - 0.5) * 2       # scale: -1 to +1
    expected_return = directional_signal * daily_vol  # scale to return
    return expected_return
```

---

### 19.5 PyPortfolioOpt Full Example for NSE Portfolio

```python
"""
Full portfolio optimization pipeline for NSE 10-stock portfolio.
Requires: pip install pyportfolioopt pandas yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models, HRPOpt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# NSE symbols (append .NS for Yahoo Finance)
NSE_SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "ICICIBANK.NS",
    "AXISBANK.NS", "WIPRO.NS", "SBIN.NS", "LT.NS", "BAJFINANCE.NS"
]

def optimize_nse_portfolio(capital: float = 500000) -> dict:
    # Fetch 2 years of daily data
    prices = yf.download(NSE_SYMBOLS, period="2y")["Close"]
    prices.columns = [s.replace(".NS", "") for s in NSE_SYMBOLS]
    prices = prices.dropna()

    # --- Option A: Markowitz Max Sharpe ---
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.25))
    ef.max_sharpe(risk_free_rate=0.065)  # India risk-free ~6.5%
    markowitz_weights = ef.clean_weights()
    perf_m = ef.portfolio_performance(risk_free_rate=0.065)

    # --- Option B: HRP ---
    returns = prices.pct_change().dropna()
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    perf_h = hrp.portfolio_performance(risk_free_rate=0.065)

    # --- Discrete allocation (how many shares to buy) ---
    latest_prices = get_latest_prices(prices)

    da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=capital)
    allocation_hrp, leftover_hrp = da_hrp.lp_portfolio()

    print(f"Markowitz Sharpe: {perf_m[2]:.3f}")
    print(f"HRP Sharpe: {perf_h[2]:.3f}")
    print(f"HRP Allocation: {allocation_hrp}")
    print(f"Leftover cash: ₹{leftover_hrp:.2f}")

    return {
        "markowitz_weights": markowitz_weights,
        "hrp_weights": hrp_weights,
        "hrp_allocation": allocation_hrp,
    }
```

**When to use which optimizer:**
| Method | Use When | Avoid When |
|--------|----------|-----------|
| Markowitz | Strong views on returns, < 15 stocks | > 20 stocks, short history |
| HRP | Many stocks, no strong return views | Need expected return estimates |
| Black-Litterman | Have ML model producing return forecasts | No reliable return signal |
| CVaR (Riskfolio) | Tail risk is the primary concern | Simple portfolio, fast rebalancing |

---

## SECTION 20 — Recommended Implementation Path

### 20.1 Complete Build Sequence with Time Estimates

#### Phase 1: Foundation Infrastructure (Weeks 1–3)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Set up Python 3.11 env, virtual env | conda / venv | 1 day | Working environment |
| Install all dependencies | pip, requirements.txt | 1 day | All imports working |
| Shoonya/Finvasia account + ShoonyaApi-py setup | Shoonya portal + GitHub SDK | 2–3 days | API keys, TOTP secret, vendor code, IMEI |
| Get static IP or AWS Mumbai | ISP / AWS EC2 | 1–2 days | Registered IP on Shoonya |
| Build config.py + .env | Python dataclasses | 0.5 days | All params centralized |
| Build broker_shoonya.py + login test | ShoonyaApi-py | 2 days | Successful login, session token |
| [Optional] Angel One account + SmartAPI registration | Angel One portal | 2–3 days | Dev-reference API keys only — do NOT route live orders here |
| Build candle_store.py (QuestDB) | questdb-client, pandas | 1 day | OHLCV stored and retrieved |
| Download 2yr historical data | jugaad-data / Shoonya historical API | 2–3 days | Full 2yr OHLCV in QuestDB |
| Build websocket_feed.py | Shoonya NorenAPI websocket | 2 days | Live 15-min candles flowing |

**Gate: Live candles appearing in SQLite in real-time before proceeding.**

---

#### Phase 2: Feature Engineering (Weeks 3–5)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Build technical.py (features 1–12, 18–19) | pandas-ta | 2 days | 14 features computing correctly |
| Build statistical.py (HMM, Kalman, GARCH) | hmmlearn, filterpy, arch | 3 days | 4 features, validated outputs |
| Build custom.py (VWAP, rel strength, ToD) | pandas, numpy | 2 days | 5 remaining features |
| Build engine.py (orchestrate all 23) | pandas | 1 day | Single function → 23-column DataFrame |
| Unit test each feature | pytest | 1 day | All features tested with known inputs |

**Common mistake here:** Not testing features on edge cases — market open (first candle), last candle before close, days with low volume. Test all edge cases explicitly.

---

#### Phase 3: Labeling & Validation (Weeks 5–7)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Build triple_barrier.py | numpy, pandas | 2 days | +1/−1/0 labels on all historical data |
| Build walk_forward.py (purge + embargo) | sklearn TimeSeriesSplit | 2 days | Train/test splits, no leakage |
| Validate label distribution | matplotlib | 0.5 days | Check ~40/20/40 label balance |
| Compute feature importance (basic) | XGBoost built-in | 0.5 days | Top 10 features identified |

**Common mistake:** Using future information in labels. Triple-barrier labels look forward — ensure no forward-looking features. Check with `pd.Timestamp` comparisons in tests.

---

#### Phase 4: ML Model Training (Weeks 7–10)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Build ensemble.py (XGB + LGBM + CatBoost) | xgboost, lightgbm, catboost | 3 days | Trained ensemble, prediction function |
| Build tuner.py (Optuna) | optuna | 2 days | 300 trials per model, best params saved |
| Build tracker.py (MLflow) | mlflow | 1 day | All experiments logged, UI accessible |
| Walk-forward backtest | Custom runner | 2 days | Sharpe, Calmar, accuracy across folds |
| Analyze results | matplotlib, seaborn | 1 day | Full performance report |

**Gate before proceeding:** Walk-forward Sharpe > 1.5 across all folds (not cherry-picked).

---

#### Phase 5: Risk & Cost Model (Weeks 10–11)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Build cost_model.py | Python (no extra library) | 1 day | Full India cost per trade |
| Build position_sizer.py (ATR-based) | pandas-ta | 1 day | Position size given capital + ATR |
| Build manager.py (kill switches) | Python threading, logging | 2 days | Daily loss, drawdown kills working |
| Integrate cost model into backtest | runner.py | 1 day | Costs deducted from every trade |

**Common mistake:** Running backtests without costs and only applying costs at the end. Costs must be deducted trade-by-trade so compounding is correct.

---

#### Phase 6: Execution & Paper Trading (Weeks 11–14)

| Task | Tool / Library | Time | Deliverable |
|------|---------------|------|-------------|
| Build order_manager.py | Shoonya NorenAPI (primary) | 2 days | LIMIT orders with 2-candle fallback |
| Build trade_logger.py (SQLite) | sqlite3 | 1 day | All fields logged per trade |
| Build telegram_bot.py | python-telegram-bot | 1 day | Trade alerts + daily P&L |
| Paper trading mode (4+ weeks) | System in paper mode | 4 weeks | P&L log, trade log, review |
| Analyze paper trading results | pandas, matplotlib | 3 days | Slippage analysis, drawdown, win rate |

**Gate:** Paper trading P&L consistently profitable across 4+ weeks with no system crashes.

---

### 20.2 Tools at Each Stage — Specific Library Names

```
STAGE              LIBRARY                      VERSION (as of 2025)
─────────────────────────────────────────────────────────────────────
Broker API         ShoonyaApi-py (PRIMARY)      check GitHub (github.com/Shoonya-Dev/ShoonyaApi-py)
                   smartapi-python (dev ref)    1.3.6+ (dev/reference only, not for live)

Data Storage       sqlite3 (stdlib)             built-in Python 3.11
                   pandas                       2.1+
                   sqlalchemy                   2.0+ (optional ORM)

Historical Data    jugaad-data                  2.0+
                   yfinance                     0.2+ (backup only)

Features           pandas-ta                    0.3.14b
                   hmmlearn                     0.3.0
                   filterpy                     1.4.5
                   arch                         6.0+
                   statsmodels                  0.14+
                   scipy                        1.11+

Labeling           numpy, pandas               (core only)

ML Training        xgboost                      2.0+
                   lightgbm                     4.0+
                   catboost                     1.2+
                   scikit-learn                 1.3+

Hyperparameter     optuna                       3.4+
                   optuna-integration-mlflow    0.3+

Tracking           mlflow                       2.9+

Portfolio Optim    PyPortfolioOpt               1.5+
                   Riskfolio-Lib               5.0+ (optional)
                   scipy.optimize               (built-in)

RL (optional)      stable-baselines3            2.2+
                   gymnasium                    0.29+
                   finrl                        0.3+

Backtesting        backtrader                   1.9+ (optional)
                   Custom runner.py             (preferred)

Alerts             python-telegram-bot          20.7+

Testing            pytest                       7.4+
                   pytest-cov                   4.1+
```

---

### 20.3 Common Mistakes at Each Stage and How to Avoid Them

#### Data Stage Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Survivorship bias in stock universe | Backtest too good, includes delisted stocks | Use NSE historical constituent lists, not current Nifty 50 |
| Look-ahead in features | Feature uses T+1 data at T | Use `.shift(1)` on all label-derived features; test with `assert` on date alignment |
| Time zone mismatch | Orders placed at wrong time | Always use IST (Asia/Kolkata) explicitly: `pd.Timestamp(..., tz='Asia/Kolkata')` |
| Duplicate candles | WebSocket sends duplicate ticks | Deduplicate by timestamp before storing |
| Missing candles on holidays | Model fails on first candle post-holiday | Validate candle sequence; fill gaps explicitly or skip |

#### Feature Engineering Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Feature computed on entire history before split | Leakage — model "knows" future | Compute features inside walk-forward loop, not globally |
| NaN in first N rows (rolling windows) | Model fails at start | Use `min_periods` parameter; explicitly handle NaN in pipeline |
| Scale mismatch | Gradient-boosted trees don't need scaling; but RL/NN does | Apply StandardScaler inside walk-forward loop (fit on train, transform test) |
| Correlated features | Inflated feature importance, unstable weights | Check correlation matrix; drop features with |corr| > 0.95 |

#### Labeling Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Too tight barriers | >50% label 0 (timeout) | Increase ATR multiplier from 1.0 to 1.5–2.0 |
| Too wide barriers | Model never triggers exit | Reduce time barrier or multiplier |
| Label imbalance | Model predicts majority class always | Check class weights parameter in XGBoost; use `scale_pos_weight` |
| No purge/embargo | Inflated walk-forward accuracy | Add 5-bar purge + 2-bar embargo minimum |

#### Model Training Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Overfitting to single split | Out-of-sample Sharpe collapses | Minimum 5 walk-forward folds; report mean ± std |
| Optuna trials too few | Suboptimal hyperparameters | 200+ trials per model; use TPESampler |
| Wrong objective metric | High accuracy but negative Sharpe | Optimize for Sharpe or F1 on minority class, not accuracy |
| Not logging to MLflow | Can't reproduce results | Log every trial: params, metrics, model artifact |

#### Execution Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Market orders in illiquid stocks | Severe slippage, poor fills | Use LIMIT orders; fall back to MARKET only after 2 missed candles |
| Not checking order status | Think order filled, position actually open | Poll order status every 30 seconds until confirmed |
| Multiple instances running | Duplicate orders, doubled positions | Use a lockfile or database flag to prevent concurrent execution |
| Strategy ID missing | SEBI compliance violation (post-2026) | Embed strategy_id in every order call |

---

### 20.4 How to Know When to Move from Paper Trading to Live

**Objective criteria (all must pass, no exceptions):**

| Criterion | Minimum Threshold | How to Measure |
|-----------|------------------|----------------|
| Paper trading duration | 4 weeks minimum (6 preferred) | Calendar date delta |
| Number of paper trades | > 50 completed round-trips | Count in trade log |
| Paper Sharpe (annualized) | > 1.5 | Compute from daily P&L in trade log |
| Max paper drawdown | < 8% | Max drawdown from peak capital |
| Win rate | > 45% (trend following: 40%+) | count(profit > 0) / count(all trades) |
| Avg win / avg loss ratio | > 1.5 | Mean profitable trades / mean losing trades |
| System uptime | > 95% during market hours | Log monitoring |
| Zero critical bugs | No unhandled exceptions | Bug log review |
| Kill switches tested | Both daily loss + drawdown triggers tested | Manual test during paper mode |
| Telegram alerts working | All trade types alerting correctly | Review alert log |

**Qualitative criteria:**

1. You can explain every line of the risk manager without hesitation
2. You have reviewed at least 20 individual losing trades and understand why they lost
3. You have a documented contingency plan for: broker API down, internet outage, system crash mid-trade
4. You have tested the emergency kill script (manually close all positions via broker UI) at least once

---

### 20.5 Minimum Viable Performance Benchmarks Before Going Live

These are the gates. If any single metric below threshold, do NOT go live.

#### Walk-Forward Backtest (Historical, Pre-Paper)

```
Sharpe Ratio (annualized)         : > 1.5
Calmar Ratio (return / max DD)    : > 1.0
Max Drawdown                       : < 15%
Win Rate                           : > 40%
Profit Factor (gross profit/loss)  : > 1.3
Number of trades (per fold)        : > 30 (statistical significance)
Consistency across folds           : Sharpe positive in all 5 folds
```

#### Paper Trading (Live, No Real Money)

```
Sharpe (paper, annualized)         : > 1.5
Paper vs. backtest Sharpe gap      : < 30% degradation
Max drawdown during paper period   : < 10%
Average slippage vs. assumed       : Within 1.5× assumed slippage
Trade execution rate               : > 95% of signals executed (no missed fills)
System crash rate                  : < 1 crash per 20 trading days
```

#### Risk Calibration Checks

```python
# Before going live, verify these programmatically:
def pre_live_checks(config) -> bool:
    checks = {
        "max_position_pct": config.max_position_pct <= 0.05,  # ≤ 5%
        "daily_loss_limit": config.daily_loss_pct <= 0.02,    # ≤ 2%
        "drawdown_limit": config.max_drawdown_pct <= 0.10,    # ≤ 10%
        "kill_switch_active": config.kill_switch_enabled,
        "cost_model_active": config.use_cost_model,
        "telegram_active": config.telegram_alerts_enabled,
        "strategy_id_set": config.strategy_id != "",          # SEBI compliance
        "live_mode_flag": config.mode == "live",
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Pre-live checks failed: {failed}")
    return True
```

#### Capital Sizing for Going Live

```
Start small: ₹1–2 lakh capital maximum for first 4 weeks live
Max position: 5% = ₹5,000–10,000 per trade
After 4 weeks profitable: scale to ₹5 lakh
After 3 months profitable: scale to ₹10–25 lakh
Never bet more than you can afford to lose completely
```

**Scaling rule:** Double capital only after 4 consecutive profitable weeks. Never increase capital after a losing week.

---

### 20.6 Enhancements After First 3 Profitable Live Months

Once you have 3 months of live profitable trading with documented results:

```
Priority 1: HMM regime filter
  → Block trades when HMM detects sideways regime
  → Expected impact: Reduce drawdown 20–30%, reduce trade count 15–20%

Priority 2: Weekly model retraining pipeline
  → Retrain on last 6 months rolling window every Sunday
  → Monitor for model drift: if live Sharpe drops 30% below paper, retrain

Priority 3: Multi-stock expansion
  → Add 3–5 more Nifty 50 stocks
  → Run same system, combine via HRP portfolio allocation

Priority 4: Confirm full Shoonya integration (already primary from Stage 1)
  → ₹0 brokerage on F&O saves ~₹40 per trade vs Zerodha
  → With 10 trades/day = ₹400/day = ₹8,000/month saving vs paid brokers
  → Note: Shoonya should be primary from Day 1 — this milestone is just a verification that Angel One dev-reference code has been fully removed from any live path

Priority 5: RL position sizing
  → Replace fixed Kelly sizing with PPO agent
  → Train with FinRL on historical data before deploying
  → Gate: Paper mode with RL sizer must outperform fixed Kelly for 4 weeks
```

---

*End of B4 — Sections 16–20 Deep Expansion*

---

## B5: Sections 21–30 Deep Expansion

---

## SECTION 21 — Transformers & Foundation Models (Deep Expansion)

### 21.1 Self-Attention Mathematics

The Transformer attention mechanism (Vaswani et al., NeurIPS 2017) computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- Q = query matrix  [seq_len × d_k]
- K = key matrix    [seq_len × d_k]
- V = value matrix  [seq_len × d_v]
- d_k = key dimension (scaling prevents vanishing gradients in softmax)
- sqrt(d_k) scaling: without it, dot products grow large → softmax saturates → gradients vanish

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
```

Each head learns different attention patterns:
- Head 1 might attend to recent candles (short-term momentum)
- Head 2 might attend to same time-of-day in previous sessions
- Head 3 might attend to high-volume bars regardless of recency

**Python implementation of scaled dot-product attention:**
```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, heads, seq_len, d_k]
    Returns: attended output [batch, heads, seq_len, d_v]
    """
    d_k = Q.size(-1)
    # Score: [batch, heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    # Output: [batch, heads, seq_len, d_v]
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        self.W_O = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)
```

---

### 21.2 Temporal Fusion Transformer (TFT) — Full Architecture

**Paper:** Lim et al. (Google, 2021) — ArXiv 1912.09363

TFT has four major components not found in vanilla Transformers:

#### 21.2.1 Gated Residual Network (GRN)

```
GRN(a, c) = LayerNorm(a + GLU(η_1))
η_1 = W_1 * η_2 + b_1
η_2 = ELU(W_2 * a + W_3 * c + b_2)   # c = optional context
GLU(η_1) = σ(W_4 * η_1 + b_4) ⊙ (W_5 * η_1 + b_5)
```

GLU (Gated Linear Unit) controls what information passes through — acts as a learned skip connection.

#### 21.2.2 Variable Selection Network (VSN)

Learns which input features are relevant at each time step:
```
ξ̃_t^(j) = GRN(ξ_t^(j))          # transform each feature
v_χt = Softmax(GRN(ξ_t_flat))    # feature importance weights
˜Ξ_t = Σ_j v_χt^(j) * ξ̃_t^(j)   # weighted combination
```

For your 23 NSE features: VSN automatically learns that RSI matters more in trending markets, OBV matters more in breakout detection, etc.

#### 21.2.3 Static Covariate Encoders

Static features (stock symbol, sector) modulate all temporal processing via 4 context vectors: c_s (selection), c_e (enrichment), c_h (LSTM init h), c_c (LSTM init c).

#### 21.2.4 Temporal Self-Attention with Interpretable Multi-Head Attention

```
Ã(Q, K, V) = [A(q, K, V)]_H
where A(q, K, V) = Softmax(q * K^T / sqrt(d_attn)) * V
MultiHead(Q, K, V) = [Ã(Q,K,V)] * W_H
```

Interpretable because all heads share the same V matrix — attention weights directly show which time steps drove each prediction.

**Python: Load TFT via Darts on NSE data:**
```python
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler

# Load NSE 15-min data
df = pd.read_csv('nse_reliance_15min.csv', parse_dates=['datetime'])
df = df.set_index('datetime').sort_index()

# Create Darts TimeSeries
series = TimeSeries.from_dataframe(df, value_cols='close', freq='15T')
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# Covariates (your 23 features)
covariates = TimeSeries.from_dataframe(
    df[['rsi_14', 'macd', 'bb_pct_b', 'atr_14', 'adx_14',
        'vwap_dev', 'obv', 'hmm_regime', 'garch_vol',
        'tod_sin', 'tod_cos']],
    freq='15T'
)
cov_scaler = Scaler()
covariates_scaled = cov_scaler.fit_transform(covariates)

# Train TFT
model = TFTModel(
    input_chunk_length=96,    # 24 hours of 15-min bars
    output_chunk_length=4,    # predict next 1 hour
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=50,
    add_relative_index=True,
    add_encoders={
        'cyclic': {'future': ['hour', 'dayofweek']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
    }
)

train, val = series_scaled[:-200], series_scaled[-200:]
model.fit(train, future_covariates=covariates_scaled, val_series=val)

# Predict next 4 bars (1 hour)
pred = model.predict(n=4, series=train, future_covariates=covariates_scaled)
pred_unscaled = scaler.inverse_transform(pred)
print(pred_unscaled.values())
```

---

### 21.3 iTransformer — Inverted Attention (ICLR 2024)

**Key insight:** Standard Transformers apply attention across time steps (each token = one time step). iTransformer INVERTS this: each token = one variate (feature) across all time steps.

**Standard attention token:** x_t = [rsi_t, macd_t, atr_t, ...] (all features at time t)
**iTransformer token:** x_i = [feature_i at t1, feature_i at t2, ..., feature_i at T] (one feature across all time)

**Why this works better for multivariate:**
- Standard: attention learns "which past time steps matter" — ignores feature correlations
- Inverted: attention learns "which features correlate" — captures RSI↔MACD↔ADX cross-feature relationships

**iTransformer attention:**
```
H_i = LayerNorm(X_i)                # X_i: variate i across time [T]
Q = H * W^Q,  K = H * W^K,  V = H * W^V   # H: [N_variates × T]
Attn = Softmax(QK^T / sqrt(d)) * V  # attention over variates, not time
Feed-Forward on time dimension: MLP applied per-token (per-variate)
```

**Python: iTransformer inference on NSE:**
```python
# Install: pip install git+https://github.com/thuml/iTransformer
import torch
import numpy as np
import pandas as pd

# Prepare multivariate input: [batch, seq_len, n_variates]
def prepare_itransformer_input(df, features, seq_len=96):
    """
    df: DataFrame with feature columns
    features: list of 23 feature names
    seq_len: lookback window (96 = 24 hours of 15-min bars)
    Returns: tensor [1, seq_len, n_features]
    """
    data = df[features].values[-seq_len:]  # last seq_len bars
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)  # standardize
    return torch.FloatTensor(data).unsqueeze(0)  # [1, 96, 23]

# Minimal iTransformer forward pass (key math):
class InvertedAttention(torch.nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(d_model)
        self.proj = torch.nn.Linear(96, d_model)  # project time dim to d_model
    
    def forward(self, x):
        # x: [B, T, N] → transpose to [B, N, T]
        x = x.transpose(1, 2)  # [B, N_variates, T]
        x = self.proj(x)       # [B, N_variates, d_model]
        # Self-attention over variate dimension
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)
```

---

### 21.4 TimesNet — 2D Period Mapping

**Core algorithm:**
1. Apply FFT to 1D time series → find top-k dominant frequencies
2. Reshape 1D series into 2D matrix using dominant period p: [T/p × p]
3. Apply 2D convolution (Inception block) to capture:
   - Intraperiod variation: patterns within each period (rows)
   - Interperiod variation: same time point across periods (columns)
4. Flatten back to 1D

**FFT period detection:**
```python
import numpy as np
import torch
import torch.nn as nn
import torch.fft

def find_dominant_periods(x, k=3):
    """
    x: [B, T, C] time series
    k: number of periods to extract
    Returns: top-k periods as integers
    """
    # FFT on time dimension
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)  # average over batch and channels
    frequency_list[0] = 0  # zero out DC component
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period_list = x.shape[1] // top_list  # period = T / frequency
    return period_list

def timesnet_reshape(x, period):
    """
    x: [B, T, C] 
    Reshape to 2D: [B, C, T/period, period]
    Pad if T not divisible by period.
    """
    B, T, C = x.shape
    pad_len = (period - T % period) % period
    if pad_len > 0:
        x = torch.cat([x, x[:, :pad_len, :]], dim=1)
    T_pad = x.shape[1]
    x = x.reshape(B, T_pad // period, period, C)
    x = x.permute(0, 3, 1, 2)  # [B, C, T/p, p]
    return x

# NSE application: 15-min bars, trading session = 375 minutes = 25 bars
# Dominant periods expected:
# - Intraday: ~25 bars (1 trading day)
# - Weekly: ~125 bars (5 trading days)
# - Monthly F&O expiry: ~500 bars (20 trading days)
```

---

### 21.5 PatchTST — Patch Tokenization

**Tokenization formula:**
```
Given sequence X ∈ R^(C × T) with C channels and T time steps:
Patch size P, stride S
Number of patches: N = floor((T - P) / S) + 2

Patch i: x_patch^i = X[:, i*S : i*S + P]  ∈ R^(C × P)
Token embedding: z_i = Linear(x_patch^i) + positional_embedding_i
```

**Self-supervised pretraining (masked patch modeling):**
```
Randomly mask 40% of patches → model reconstructs masked patches
Loss = MSE(reconstructed_masked_patches, original_patches)
After pretraining: fine-tune head for forecasting/classification
```

**Python: PatchTST for NSE:**
```python
from darts.models import PatchTSTModel
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# Load data
df = pd.read_csv('nse_15min.csv', parse_dates=['datetime'], index_col='datetime')
series = TimeSeries.from_dataframe(df[['close']], freq='15T')

scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# PatchTST config
model = PatchTSTModel(
    input_chunk_length=192,   # 48 hours lookback
    output_chunk_length=4,    # predict 1 hour ahead
    patch_length=16,          # 4-hour patches
    stride=8,
    d_model=128,
    n_heads=4,
    d_ffn=256,
    num_encoder_layers=3,
    dropout=0.1,
    batch_size=32,
    n_epochs=100,
    likelihood=None,          # point forecast
)

train = series_scaled[:-500]
model.fit(train)
forecast = model.predict(n=4)
forecast_actual = scaler.inverse_transform(forecast)
```

---

### 21.6 Foundation Models: Tokenization Strategies

#### Moirai (Salesforce) — Any-variate patching
```
Input patches: divide each variate into non-overlapping patches of size p
Tokenization: Linear projection of each patch → token embedding
Special tokens: [CLS] for classification, [MASK] for missing values
Mixture output head: predicts full distribution (not just mean)
  → output = Mixture(Normal, Student-T, Negative-Binomial)
  → handles different data distributions without knowing data type
```

**Python: Moirai zero-shot NSE forecast:**
```python
# pip install uni2ts
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import torch
from einops import rearrange

# Load pretrained Moirai-1.0-R-Small (fastest)
model = MoiraiForecast.load_from_checkpoint(
    prediction_length=4,
    context_length=200,
    patch_size=16,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
)

# NSE close prices: shape [1, 1, context_length] = [batch, variates, time]
past_target = torch.FloatTensor(close_prices[-200:]).unsqueeze(0).unsqueeze(0)
past_observed = torch.ones_like(past_target, dtype=torch.bool)

forecast = model(
    past_target=past_target,
    past_observed_target=past_observed,
)
# forecast shape: [n_samples, prediction_length]
median_forecast = forecast.quantile(0.5, dim=0)
print(f"Predicted next 4 closes: {median_forecast}")
```

#### Chronos (Amazon) — Quantization tokenization
```
Step 1: Scale input: x_scaled = x / mean(|x|)  (mean absolute scaling)
Step 2: Quantize to B=4096 bins using uniform quantization on scaled values
  bin_id = floor((x_scaled - x_min) / (x_max - x_min) * (B-1))
Step 3: Treat bin IDs as vocabulary tokens → standard LLM sequence modeling
Step 4: At inference: sample tokens → dequantize → multiply by scale
```

**Python: Chronos NSE forecast:**
```python
# pip install chronos-forecasting
import torch
import numpy as np
import pandas as pd
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # small = fast, large = more accurate
    device_map="cpu",
    torch_dtype=torch.float32,
)

# NSE close prices as numpy array
context = torch.tensor(close_prices[-200:], dtype=torch.float32)

# Zero-shot forecast
forecast = pipeline.predict(
    context=context.unsqueeze(0),   # [1, context_len]
    prediction_length=4,             # 4 bars ahead
    num_samples=20,                  # probabilistic samples
)
# forecast shape: [1, num_samples, prediction_length]
low = np.quantile(forecast[0].numpy(), 0.1, axis=0)
median = np.quantile(forecast[0].numpy(), 0.5, axis=0)
high = np.quantile(forecast[0].numpy(), 0.9, axis=0)
print(f"Forecast next 4 bars: {median}")
print(f"90% CI: [{low}, {high}]")
```

#### TimesFM (Google) — Patched decoder
```
Architecture: decoder-only transformer (like GPT)
Tokenization: non-overlapping patches of length p=32
  patch_token_i = Linear(x[i*32 : (i+1)*32])
Training: 100B time points from Google internal + public datasets
Output: point forecast (quantile regression head for intervals)
```

**Python: TimesFM NSE:**
```python
# pip install timesfm
import timesfm
import numpy as np

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend='cpu',
        per_core_batch_size=32,
        horizon_len=4,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id='google/timesfm-1.0-200m'
    ),
)
tfm.initialize_from_checkpoint()

# NSE close prices
forecast_input = [np.array(close_prices[-512:], dtype=np.float32)]
freq_input = [0]  # 0=high_freq, 1=med_freq, 2=low_freq

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input, freq=freq_input
)
print(f"Next 4 bars: {point_forecast[0]}")
```

---

### 21.7 Practical NSE Deployment: Which Model When

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Quick baseline, no GPU | Chronos-small (CPU) | Zero-shot, 2min inference |
| Best accuracy, 1 stock | PatchTST fine-tuned | Best single-variate SOTA |
| All 23 features together | iTransformer | Designed for multivariate |
| Probabilistic forecast needed | Lag-Llama or Moirai | Full distribution output |
| Multi-period NSE cycles | TimesNet | FFT detects daily/weekly/monthly |
| Interpretable for SEBI audit | TFT | Variable importance scores |
| Production low-latency | DLinear baseline first | If linear beats transformer, use linear |

---

## SECTION 22 — Reinforcement Learning Deep Coverage (20+ Algorithms)

### 22.1 RL Foundations for Trading

**MDP formulation for NSE trading:**
```
State s_t:  23 features + portfolio state (position, unrealized_pnl, cash)
Action a_t: continuous in [-1, 1] → position size multiplier
             OR discrete {-1, 0, +1} → sell/hold/buy signal
Reward r_t: risk-adjusted return (Sharpe component)
Transition: s_{t+1} = market dynamics + executed action
Episode:    one trading day (25 bars of 15-min) or one week
```

---

### 22.2 PPO — Proximal Policy Optimization

**Paper:** Schulman et al. (OpenAI, 2017) — ArXiv 1707.06347

**Clipped Surrogate Objective:**
```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)    # probability ratio

L^CLIP(θ) = E_t[ min(r_t(θ) * Â_t,  clip(r_t(θ), 1-ε, 1+ε) * Â_t) ]

where:
  Â_t = advantage estimate (how much better action was vs baseline)
  ε   = clip range (typically 0.1–0.2)
  The clip prevents too-large policy updates (trust region)
```

**Value function loss:**
```
L^VF(θ) = E_t[ (V_θ(s_t) - V_t^target)^2 ]
V_t^target = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ... (n-step return)
```

**Entropy bonus:**
```
L^S(θ) = E_t[ H(π_θ(· | s_t)) ]  = -E_t[ Σ_a π_θ(a|s) log π_θ(a|s) ]
Encourages exploration — prevents premature convergence to deterministic policy
```

**Full PPO objective:**
```
L^PPO(θ) = L^CLIP(θ) - c_1 * L^VF(θ) + c_2 * L^S(θ)
c_1 ≈ 0.5 (value loss coefficient)
c_2 ≈ 0.01 (entropy coefficient — reduce if policy too random)
```

**PPO update rule:**
```
For each epoch (repeat K=10 times on same data):
  Compute gradient: g = ∇_θ L^PPO(θ)
  Adam update: θ ← θ + α * Adam(g)
  Early stop if KL(π_θ, π_θ_old) > 0.02 (optional)
```

**Python: PPO for NSE position sizing:**
```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# See Section 22.9 for full NSETradingEnv class

env = DummyVecEnv([lambda: NSETradingEnv(df_train, features)])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,       # rollout length before update
    batch_size=64,
    n_epochs=10,        # reuse each rollout 10 times
    gamma=0.99,
    gae_lambda=0.95,    # GAE lambda for advantage estimation
    clip_range=0.2,     # ε in clipped objective
    clip_range_vf=None,
    ent_coef=0.01,      # entropy bonus coefficient
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./ppo_nse_logs/"
)
model.learn(total_timesteps=500_000)
model.save("ppo_nse_position_sizer")
```

---

### 22.3 SAC — Soft Actor-Critic

**Paper:** Haarnoja et al. (Berkeley, 2018) — ArXiv 1801.01290

**Maximum entropy RL objective:**
```
J(π) = Σ_t E_{(s_t,a_t)~ρ_π} [ r(s_t, a_t) + α * H(π(·|s_t)) ]

H(π(·|s_t)) = -E[log π(a_t|s_t)]  = entropy of policy

α = temperature parameter (controls exploration/exploitation trade-off)
```

**Soft Bellman equation:**
```
Q*(s_t, a_t) = r_t + γ * E_{s_{t+1}}[ V*(s_{t+1}) ]
V*(s_t) = E_{a~π}[ Q*(s_t, a) - α * log π(a|s_t) ]
         = E_{a~π}[ Q*(s_t, a) ] + α * H(π(·|s_t))
```

**Twin critics (prevents overestimation):**
```
L_Q(θ_i) = E[ (Q_θi(s_t, a_t) - y_t)^2 ]   for i = 1, 2
y_t = r_t + γ * (min_{i=1,2} Q_θ̄i(s_{t+1}, ã_{t+1}) - α * log π(ã_{t+1}|s_{t+1}))
ã_{t+1} ~ π_φ(·|s_{t+1})
```

**Automatic temperature tuning:**
```
L(α) = E_{a_t~π_t}[ -α * log π_t(a_t|s_t) - α * H̄ ]
H̄ = target entropy = -dim(A)  (e.g., -1 for 1D continuous action)
```

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,    # replay buffer size
    learning_starts=1000,
    batch_size=256,
    tau=0.005,              # soft update coefficient for target networks
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',        # automatic temperature tuning
    target_entropy='auto',  # = -dim(action_space)
    verbose=1,
)
model.learn(total_timesteps=300_000)
```

---

### 22.4 TD3 — Twin Delayed Deep Deterministic Policy Gradient

**Paper:** Fujimoto et al. (McGill, 2018) — ArXiv 1802.09477

Three improvements over DDPG:

**1. Twin critics:**
```
y = r + γ * min(Q_θ̄1(s', ã'), Q_θ̄2(s', ã'))
```

**2. Delayed policy update:**
```
Update critics every step
Update actor only every d=2 steps
```

**3. Target policy smoothing:**
```
ã' = clip(μ_θ̄(s') + clip(ε, -c, c), a_low, a_high)
ε ~ N(0, σ)   (σ=0.2, c=0.5 typical)
```

```python
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

model = TD3(
    "MlpPolicy", env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=100,
    batch_size=100,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    action_noise=action_noise,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
)
```

---

### 22.5 DDPG — Deep Deterministic Policy Gradient

**Deterministic policy gradient theorem:**
```
∇_θ J(μ_θ) = E_{s~ρ^β}[ ∇_a Q^μ(s,a)|_{a=μ_θ(s)} * ∇_θ μ_θ(s) ]
```

**Soft target updates:**
```
θ^{Q'} ← τ * θ^Q + (1-τ) * θ^{Q'}   (τ ≈ 0.001)
θ^{μ'} ← τ * θ^μ + (1-τ) * θ^{μ'}
```

**Weakness:** Overestimation of Q-values. Use TD3 or SAC instead.

---

### 22.6 DQN — Deep Q-Network

**Q-learning update:**
```
Q(s, a) ← Q(s, a) + α * [r + γ * max_{a'} Q(s', a') - Q(s, a)]
```

**DQN loss:**
```
L(θ) = E[(r + γ * max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))^2]
θ^- = target network (frozen, updated every C steps)
```

**When to use DQN:**
- Discrete action space only: {-1, 0, +1} = {sell, hold, buy}
- Good for NSE where you just want a signal, not continuous position size

---

### 22.7 Rainbow DQN — Combines 6 Improvements

**The 6 components:**

**1. Double DQN:**
```
y = r + γ * Q(s', argmax_{a'} Q(s', a'; θ); θ^-)
```

**2. Prioritized Experience Replay:**
```
Priority: p_i = |δ_i| + ε
Sample probability: P(i) = p_i^α / Σ_j p_j^α   (α=0.6)
```

**3. Dueling Networks:**
```
Q(s, a; θ) = V(s; θ) + A(s, a; θ) - mean_{a'} A(s, a'; θ)
```

**4. Multi-step Returns:**
```
R_t^(n) = Σ_{k=0}^{n-1} γ^k * r_{t+k} + γ^n * max_{a'} Q(s_{t+n}, a')
```

**5. Distributional RL (C51):**
```
Model return distribution Z(s,a) with C=51 atoms
Loss: KL divergence between projected target and predicted distribution
```

**6. Noisy Networks:**
```
y = (μ^w + σ^w ⊙ ε^w) x + (μ^b + σ^b ⊙ ε^b)
ε ~ factorized Gaussian noise
```

```python
# Rainbow via Tianshou library
# pip install tianshou
from tianshou.policy import RainbowPolicy
from tianshou.utils.net.discrete import NoisyLinear
```

---

### 22.8 Additional RL Algorithms Reference Table

| Algorithm | Objective Function | Action Space | Key Trait | When to Use |
|-----------|-------------------|-------------|-----------|-------------|
| **A2C** | E[Â_t * log π(a|s)] - c*H | Both | Synchronous actors, fast | Quick baseline |
| **A3C** | Same as A2C | Both | Asynchronous, parallel | Multi-core CPU |
| **TRPO** | E[r_t * Â_t] s.t. KL≤δ | Both | KL constraint, safe updates | Sensitive environments |
| **HER** | Hindsight relabeling | Both | Sparse reward environments | Goal-conditioned |
| **MADDPG** | Σ J(μ_i) joint | Continuous | Multi-agent | Multi-stock portfolio |
| **D4PG** | Distributional DDPG | Continuous | Distribution over returns | Risk-aware trading |
| **MPO** | E-step/M-step EM | Both | Sample efficient, stable | Production systems |
| **IMPALA** | V-trace off-policy | Both | Distributed, scalable | Large data centers |
| **R2D2** | LSTM + prioritized | Discrete | Long-term memory | Regime-dependent trades |
| **MuZero** | Learned model + MCTS | Both | No env model needed | Best for known rules |
| **Dreamer v3** | RSSM world model | Both | Latent imagination | Low interaction budget |
| **Decision Transformer** | Supervised on (R, s, a) | Both | Offline, no Bellman | Pure historical data |
| **CQL** | Q - α*log_sum_exp(Q) | Both | Conservative offline | Prevent extrapolation errors |
| **IQL** | Expectile regression | Both | Stable offline | Most practical offline RL |
| **BCQ** | VAE-constrained Q | Continuous | Near-behavior policy | When data is narrow |
| **BEAR** | MMD kernel constraint | Both | Distributional constraint | OOD-robust offline |
| **MAML** | meta-gradient (2-level) | Both | Fast regime adaptation | Cross-regime transfer |
| **PEARL** | Probabilistic encoder | Continuous | Context inference | Multi-task markets |
| **IRL/GAIL** | Discriminator reward | Both | Imitation of experts | Copy institutional flow |
| **CVaR-RL** | CVaR-constrained MDP | Both | Risk-constrained | Drawdown minimization |

---

### 22.9 Custom NSE Trading Gym Environment (Full Code)

```python
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class NSETradingEnv(gym.Env):
    """
    NSE 15-min trading environment for RL position sizing.
    
    State:  23 technical features + [position, unrealized_pnl, cash_ratio]
    Action: continuous [-1, 1] → position size multiplier
    Reward: Sharpe-adjusted return minus transaction costs
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        max_position_pct: float = 0.95,
        reward_window: int = 20,
        episode_length: int = 100,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.features = features
        self.n_features = len(features)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_pct = max_position_pct
        self.reward_window = reward_window
        self.episode_length = episode_length
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        n_obs = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        
        self.current_step = 0
        self.start_step = 0
        self.capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.returns_history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.df) - self.episode_length - 1
        self.start_step = self.np_random.integers(0, max_start)
        self.current_step = self.start_step
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.returns_history = []
        self.portfolio_value = self.initial_capital
        return self._get_obs(), {}
    
    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        current_price = self.df.loc[self.current_step, 'close']
        next_price = self.df.loc[self.current_step + 1, 'close']
        
        old_position = self.position
        self.position = action * self.max_position_pct
        position_change = abs(self.position - old_position)
        cost = position_change * self.capital * self.transaction_cost_pct
        price_return = (next_price - current_price) / current_price
        pnl = self.position * self.capital * price_return
        self.capital = self.capital + pnl - cost
        self.portfolio_value = self.capital
        
        step_return = (pnl - cost) / self.initial_capital
        self.returns_history.append(step_return)
        reward = self._calculate_reward()
        self.current_step += 1
        
        done = (self.current_step >= self.start_step + self.episode_length)
        truncated = (self.capital <= self.initial_capital * 0.7)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'step_return': step_return,
            'capital': self.capital,
        }
        return self._get_obs(), reward, done, truncated, info
    
    def _get_obs(self):
        features = self.df.loc[self.current_step, self.features].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        portfolio_state = np.array([
            self.position,
            (self.capital - self.initial_capital) / self.initial_capital,
            self.capital / self.initial_capital,
        ], dtype=np.float32)
        return np.concatenate([features, portfolio_state])
    
    def _calculate_reward(self):
        if len(self.returns_history) < 2:
            return 0.0
        window = self.returns_history[-self.reward_window:]
        mean_r = np.mean(window)
        std_r = np.std(window) + 1e-8
        sharpe = mean_r / std_r * np.sqrt(252 * 25)
        return float(sharpe)
    
    def render(self):
        print(f"Step: {self.current_step} | Capital: ₹{self.capital:,.0f} | "
              f"Position: {self.position:.2f} | "
              f"Return: {(self.capital/self.initial_capital - 1)*100:.2f}%")
```

---

### 22.10 RL Reward Engineering for NSE

**Differential Sharpe Ratio (Moody & Saffell, 1998):**
```
A_t = A_{t-1} + η * (R_t - A_{t-1})         # running mean
B_t = B_{t-1} + η * (R_t^2 - B_{t-1})        # running second moment
D_t = (B_t * ΔA_t - 0.5 * A_t * ΔB_t) / (B_t - A_t^2)^{3/2}
η = 0.01 (learning rate for statistics)
```

**India-specific cost-aware reward:**
```python
def nse_reward(action, old_action, price_return, capital):
    trade_value = abs(action - old_action) * capital
    brokerage = min(trade_value * 0.0003, 20.0)
    stt = trade_value * 0.001
    exchange_charges = trade_value * 0.0000345
    sebi_charges = trade_value * 0.000001
    gst = (brokerage + exchange_charges) * 0.18
    stamp_duty = trade_value * 0.00015
    
    total_cost = brokerage + stt + exchange_charges + sebi_charges + gst + stamp_duty
    cost_pct = total_cost / capital
    
    raw_return = action * price_return
    reward = raw_return - cost_pct
    return reward
```

---

## SECTION 23 — Options Research (Deep Expansion)

### 23.1 Black-Scholes Formula

**Call option price:**
```
C = S * N(d1) - K * e^(-rT) * N(d2)

d1 = [ln(S/K) + (r + σ^2/2) * T] / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)

S = spot price
K = strike price
r = risk-free rate (India: 91-day T-bill ≈ 6.5%)
T = time to expiry in years
σ = implied volatility
```

**Put-Call Parity:**
```
C - P = S - K * e^(-rT)
```

**Python: Black-Scholes pricer:**
```python
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
```

---

### 23.2 The Greeks — Full Formulas

```
Delta:  Δ_C = N(d1);   Δ_P = N(d1) - 1
Gamma:  Γ = N'(d1) / (S * σ * sqrt(T))
Theta:  Θ_C = -[S*N'(d1)*σ/(2*sqrt(T))] - r*K*e^(-rT)*N(d2)
Vega:   ν = S * sqrt(T) * N'(d1)
Rho:    ρ_C = K*T*e^(-rT)*N(d2)
```

**Python: All Greeks:**
```python
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 1e-6:
        return {'delta': 1.0 if option_type=='call' else -1.0,
                'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * nd1 / 100
    
    if option_type == 'call':
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {
        'delta': round(delta, 4), 'gamma': round(gamma, 6),
        'theta': round(theta, 4), 'vega':  round(vega, 4),
        'rho':   round(rho, 4),
    }
```

---

### 23.3 Heston Stochastic Volatility Model

**SDE system:**
```
dS_t = μ * S_t * dt + sqrt(v_t) * S_t * dW_t^S
dv_t = κ * (θ - v_t) * dt + ξ * sqrt(v_t) * dW_t^v
Corr(dW_t^S, dW_t^v) = ρ * dt

Parameters:
  κ  = mean reversion speed (typical: 1-5)
  θ  = long-run variance (e.g. 0.04 = 20% LR vol)
  ξ  = vol of vol (typical: 0.3-1.0)
  ρ  = correlation (typical: -0.7 for equities)
```

**Python: Heston via QuantLib:**
```python
import QuantLib as ql

def heston_price(S, K, T_days, r, kappa, theta, sigma_v, rho, v0, option_type='call'):
    today = ql.Date.todaysDate()
    expiry = today + ql.Period(T_days, ql.Days)
    ql.Settings.instance().evaluationDate = today
    
    exercise = ql.EuropeanExercise(expiry)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == 'call' else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed()))
    div_handle  = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    
    heston_process = ql.HestonProcess(
        rate_handle, div_handle, spot_handle, v0, kappa, theta, sigma_v, rho)
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(engine)
    return option.NPV()
```

---

### 23.4 Implied Volatility Solver

```python
from scipy.optimize import brentq

def implied_volatility_newton(C_market, S, K, T, r, option_type='call', tol=1e-6):
    sigma = np.sqrt(2 * np.pi / T) * C_market / S
    sigma = max(0.01, min(sigma, 5.0))
    
    for _ in range(100):
        price = black_scholes(S, K, T, r, sigma, option_type)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        diff = price - C_market
        if abs(diff) < tol:
            return sigma
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))
    
    try:
        return brentq(
            lambda s: black_scholes(S, K, T, r, s, option_type) - C_market,
            0.001, 10.0, xtol=tol)
    except ValueError:
        return np.nan
```

---

### 23.5 NSE Options Structure

```python
NSE_LOT_SIZES = {
    'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 40,
    'MIDCPNIFTY': 75, 'SENSEX': 10,
    'RELIANCE': 250, 'TCS': 150, 'INFY': 300,
    'HDFCBANK': 550, 'ICICIBANK': 700,
    'WIPRO': 1500, 'SBIN': 1500,
    'BAJFINANCE': 125, 'ADANIENT': 500, 'ITC': 3200,
}
```

---

### 23.6 Options Strategies — P&L Formulas

**Iron Condor:**
```
Legs: Sell OTM Put (K1) + Buy further OTM Put (K2) 
    + Sell OTM Call (K3) + Buy further OTM Call (K4)
Max Profit = Net Premium Received
Max Loss = (K1 - K2) - Net_Premium
```

**Straddle:**
```
Long Straddle: Cost = C_K + P_K; P&L = |S - K| - Cost
Short Straddle: Credit = C_K + P_K; P&L = Credit - |S - K|
```

**Bull Call Spread:**
```
Buy Call K1, Sell Call K2 (K1 < K2)
Max Profit = (K2 - K1) - Net_Debit
Max Loss   = Net_Debit
```

```python
def iron_condor_pnl(spot_at_expiry, K1, K2, K3, K4, P_K1, P_K2, C_K3, C_K4):
    net_premium = P_K1 - P_K2 + C_K3 - C_K4
    S = spot_at_expiry
    put_spread = min(K1 - S, K1 - K2) if S < K1 else 0
    put_spread = max(put_spread, -(K1 - K2))
    call_spread = min(S - K3, K4 - K3) if S > K3 else 0
    call_spread = max(call_spread, -(K4 - K3))
    return net_premium - put_spread - call_spread

def straddle_pnl(spot_at_expiry, strike, call_premium, put_premium, position='long'):
    intrinsic = abs(spot_at_expiry - strike)
    total_premium = call_premium + put_premium
    return intrinsic - total_premium if position == 'long' else total_premium - intrinsic
```

---

### 23.7 Options Backtesting

**AlgoTest:** https://algotest.in/ — India-specific options backtester with real NSE premium data.

**Sensibull:** https://sensibull.com/ — Strategy P&L diagrams, Greeks dashboard, IV percentile.

---

## SECTION 24 — Execution Algorithms (Deep Expansion)

### 24.1 VWAP Algorithm

```
VWAP_t = Σ (Price_i * Volume_i) / Σ Volume_i
```

```python
NSE_VOLUME_PROFILE = np.array([
    0.090, 0.060, 0.045, 0.040, 0.038,  # 9:15-10:15 (heavy open)
    0.038, 0.035, 0.033, 0.032, 0.030,  # 10:15-11:30
    0.028, 0.027, 0.026, 0.026, 0.027,  # 11:30-12:45 (lightest)
    0.030, 0.032, 0.035, 0.038, 0.040,  # 12:45-14:00
    0.045, 0.050, 0.055, 0.070, 0.080,  # 14:00-15:15 (institutional)
])

def vwap_execution_schedule(total_qty, start_bar=0,
                             volume_profile=NSE_VOLUME_PROFILE,
                             aggressive_factor=1.0):
    remaining_profile = volume_profile[start_bar:].copy()
    remaining_profile = remaining_profile ** aggressive_factor
    remaining_profile = remaining_profile / remaining_profile.sum()
    schedule = np.floor(total_qty * remaining_profile).astype(int)
    remainder = total_qty - schedule.sum()
    schedule[np.argmax(remaining_profile)] += remainder
    return schedule
```

---

### 24.2 TWAP Algorithm

```python
def twap_execution_schedule(total_qty, n_bars, start_immediately=True):
    base_qty = total_qty // n_bars
    remainder = total_qty % n_bars
    schedule = np.full(n_bars, base_qty, dtype=int)
    if start_immediately:
        schedule[0] += remainder
    else:
        schedule[-1] += remainder
    return schedule
```

---

### 24.3 Implementation Shortfall (Perold 1988)

```
IS = Delay Cost + Market Impact + Timing Risk + Opportunity Cost
```

```python
def implementation_shortfall(decision_price, exec_prices, exec_quantities,
                               total_intended_qty, end_of_day_price):
    total_executed = exec_quantities.sum()
    unexecuted = total_intended_qty - total_executed
    avg_exec_price = (exec_prices * exec_quantities).sum() / total_executed if total_executed > 0 else decision_price
    market_impact = (avg_exec_price - decision_price) / decision_price * 10000
    opp_cost = ((end_of_day_price - decision_price) / decision_price
                * (unexecuted / total_intended_qty) * 10000) if total_intended_qty > 0 else 0
    return {
        'market_impact_bps': round(market_impact, 2),
        'opportunity_cost_bps': round(opp_cost, 2),
        'total_IS_bps': round(market_impact + opp_cost, 2),
        'fill_rate': round(total_executed / total_intended_qty, 3),
    }
```

---

### 24.4 Almgren-Chriss Optimal Execution

**Optimal trajectory:**
```
x_k* = X * sinh(κ*(T-t_k)) / sinh(κ*T)
κ = sqrt(λ*σ^2 / η)
```

```python
def almgren_chriss_schedule(X, T, sigma, eta, gamma, lam):
    tau = 1.0
    kappa = np.sqrt(lam * sigma**2 / eta)
    t_steps = np.arange(0, T + 1) * tau
    
    if kappa * T < 1e-6:
        x_star = X * (1 - t_steps / (T * tau))
    else:
        x_star = X * np.sinh(kappa * (T*tau - t_steps)) / np.sinh(kappa * T * tau)
    
    n_star = np.diff(-x_star)
    n_int = np.floor(n_star).astype(int)
    n_int[0] += X - n_int.sum()
    return n_int
```

---

### 24.5 NSE Order Types Reference

```python
ORDER_TYPES = {
    'MARKET': 'MARKET', 'LIMIT': 'LIMIT',
    'SL': 'SL', 'SL-M': 'SL-M',
    'AMO': 'AMO', 'GTT': 'GTT',
}

VALIDITY = {'DAY': 'DAY', 'IOC': 'IOC', 'EOS': 'EOS'}

PRODUCT_TYPES = {
    'INTRADAY': 'INTRADAY',  # MIS
    'DELIVERY': 'DELIVERY',  # CNC
    'MARGIN':   'MARGIN',    # NRML
}
```

---

### 24.6 Kyle's Lambda — Market Impact

```
ΔP_t = λ * OF_t + ε_t
OF_t = buy_volume_t - sell_volume_t
High λ = illiquid stock; Low λ = liquid
```

```python
def estimate_kyle_lambda(tick_df, window=100):
    lambdas = []
    for i in range(window, len(tick_df)):
        subset = tick_df.iloc[i-window:i]
        X = subset['signed_volume'].values.reshape(-1, 1)
        y = subset['price_change'].values
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            lambdas.append(reg.coef_[0])
        except:
            lambdas.append(np.nan)
    return pd.Series(lambdas, index=tick_df.index[window:])
```

---

## SECTION 25 — Feature Engineering: All 23 Features (Complete Reference)

```python
# All 23 features with formulas and implementations
FEATURE_NAMES_23 = [
    'rsi_14', 'macd', 'bb_pct_b', 'bb_width', 'atr_14',
    'adx_14', 'stoch_k', 'vwap_dev', 'obv_norm', 'supertrend_dir',
    'ema_diff', 'ichimoku_cloud_pos', 'kalman_ratio', 'hmm_regime',
    'garch_vol', 'obi', 'pv_divergence', 'ha_dir', 'pivot_dist',
    'returns_autocorr', 'rel_strength_nifty', 'tod_sin', 'tod_cos',
    'bid_ask_spread'
]
```

**Feature 1 — RSI(14):** `RSI = 100 - 100/(1+AvgGain/AvgLoss)` via Wilder smoothing
```python
df['rsi_14'] = ta.rsi(df['close'], length=14)
```

**Feature 2 — MACD(12,26,9):** `MACD = EMA(12) - EMA(26); Signal = EMA(MACD,9)`
```python
df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])
```

**Feature 3 — BB %B:** `(close - lower) / (upper - lower)`
```python
df['bb_pct_b'], df['bb_upper'], df['bb_lower'], df['bb_mid'] = compute_bb_pct_b(df['close'])
```

**Feature 4 — BB Width:** `4 * StdDev / SMA`
```python
df['bb_width'] = compute_bb_width(df['close'])
```

**Feature 5 — ATR(14):** `Wilder_smooth(max(H-L, |H-Cp|, |L-Cp|), 14)`
```python
df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
```

**Feature 6 — ADX(14):** DI+, DI-, smoothed DX → ADX
```python
adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
df['adx_14'] = adx_df['ADX_14']
```

**Feature 7 — Stoch %K/%D:** `%K = 100*(close-low14)/(high14-low14); %D = SMA(%K,3)`
```python
stoch = ta.stoch(df['high'], df['low'], df['close'])
df['stoch_k'] = stoch['STOCHk_14_3_3']
```

**Feature 8 — VWAP Deviation:** `(close - VWAP) / ATR(14)` (cumulative, reset daily)
```python
df['vwap'] = compute_vwap(df['high'], df['low'], df['close'], df['volume'])
df['vwap_dev'] = (df['close'] - df['vwap']) / (df['atr_14'] + 1e-10)
```

**Feature 9 — OBV:** `cumsum(sign(dClose) * volume)`
```python
df['obv'] = ta.obv(df['close'], df['volume'])
df['obv_norm'] = (df['obv'] - df['obv'].rolling(50).mean()) / (df['obv'].rolling(50).std() + 1e-10)
```

**Feature 10 — Supertrend:** ATR channel with directional ratcheting (+1/-1)
```python
df['supertrend_dir'] = compute_supertrend(df['high'], df['low'], df['close'])
```

**Feature 11 — EMA Cross (9/21):** `ema_diff = EMA(9) - EMA(21)`
```python
df['ema_diff'] = df['close'].ewm(span=9, adjust=False).mean() - df['close'].ewm(span=21, adjust=False).mean()
```

**Feature 12 — Ichimoku Cloud Position:** +1 above cloud, -1 below, 0 inside
```python
df['ichimoku_cloud_pos'] = ichimoku_cloud_pos(df['high'], df['low'], df['close'])
```

**Feature 13 — Kalman Smoothed Price:** Constant-velocity Kalman filter
```python
df['kalman_ratio'] = compute_kalman_price(df['close']) / df['close']
```

**Feature 14 — HMM Regime:** GaussianHMM(k=3) on [log_return, log_volume_ratio]
```python
df['hmm_regime'] = compute_hmm_regime(df['close'], df['volume'])
```

**Feature 15 — GARCH(1,1) Vol:** `σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}`
```python
df['garch_vol'] = compute_garch_vol(df['close'])
```

**Feature 16 — Order Book Imbalance:** `(bid_vol - ask_vol) / (bid_vol + ask_vol)`
```python
df['obi'] = approx_obi(df['close'], df['volume'])
```

**Feature 17 — Price-Volume Divergence:** `rolling_corr(pct_change_price, pct_change_volume, 20)`
```python
df['pv_divergence'] = df['close'].pct_change().rolling(20).corr(df['volume'].pct_change())
```

**Feature 18 — Heikin-Ashi Direction:** `+1 if HA_close > HA_open else -1`
```python
df['ha_dir'] = compute_ha(df['open'], df['high'], df['low'], df['close'])
```

**Feature 19 — Pivot Distance:** `(close - nearest_pivot_level) / ATR(14)`
```python
df['pivot_dist'] = compute_pivot_distance(df)
```

**Feature 20 — Returns Autocorrelation:** `corr(returns[t-20:t], returns[t-25:t-5])`
```python
df['returns_autocorr'] = compute_returns_autocorr(df['close'])
```

**Feature 21 — Relative Strength vs Nifty:** `z_score(stock_ret_20 - nifty_ret_20)`
```python
df['rel_strength_nifty'] = compute_rel_strength(df['close'], nifty_df['close'])
```

**Feature 22 — Time-of-Day (sin/cos):** `sin(2π*t/375), cos(2π*t/375)`
```python
df['tod_sin'], df['tod_cos'] = compute_tod(df.index)
```

**Feature 23 — Bid-Ask Spread:** Corwin-Schultz H/L approximation
```python
df['bid_ask_spread'] = estimate_spread_hl(df['high'], df['low'])
```

---

## SECTION 26 — Free Datasets (Complete URLs and Code)

### 26.1 NSE Bhavcopy

```python
def download_nse_bhavcopy(dt):
    mon = dt.strftime('%b').upper()
    url = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
           f"{dt.year}/{mon}/cm{dt.strftime('%d')}{mon}{dt.year}bhav.csv.zip")
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers, timeout=30)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]))
```

### 26.2 NSE F&O Bhavcopy

```
https://archives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MON}/fo{DD}{MON}{YYYY}bhav.csv.zip
Columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI
```

### 26.3 India VIX Historical

```python
session = requests.Session()
session.get('https://www.nseindia.com', headers={'User-Agent': 'Mozilla/5.0'})
vix_url = 'https://www.nseindia.com/api/historical/vixhistory?from=01-01-2024&to=01-01-2025'
vix_df = pd.DataFrame(session.get(vix_url, headers={'User-Agent': 'Mozilla/5.0'}).json()['data'])
```

### 26.4 FRED — Federal Reserve (India-Relevant)

| Series ID | Description |
|-----------|-------------|
| DEXINUS | USD/INR Exchange Rate (daily) |
| INDCPIALLMINMEI | India CPI |
| VIXCLS | CBOE VIX |
| GS10 | US 10-yr Treasury yield |

```python
from fredapi import Fred
fred = Fred(api_key='your_free_key_from_fred.stlouisfed.org')
usdinr = fred.get_series('DEXINUS', observation_start='2020-01-01')
vix    = fred.get_series('VIXCLS')
```

### 26.5 RBI Data Warehouse

**URL:** https://dbie.rbi.org.in — Repo Rate, CRR, Forex Reserves, Bank Credit, M3.

### 26.6 data.gov.in — Open Government India

```python
def get_gst_data(api_key):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {'api-key': api_key, 'format': 'json', 'limit': 100}
    return pd.DataFrame(requests.get(url, params=params).json()['records'])
```

### 26.7 Kaggle NSE Datasets

| Dataset | URL |
|---------|-----|
| Nifty50 Stock Market Data | https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data |
| Indian Stock Market | https://www.kaggle.com/datasets/abhishek14398/nse-stock-data |

---

## SECTION 27 — Backtesting Frameworks: Deep Comparison

### 27.1 Framework Comparison Matrix

| Feature | Backtrader | VectorBT | PyBroker | NautilusTrader | QuantConnect |
|---------|-----------|---------|---------|----------------|-------------|
| **GitHub Stars** | 21K | 5K | 3.1K | 22.6K | 19K |
| **Speed** | 1x | 500x | 10x | 1000x | 50x |
| **Live Trading** | Via adapters | No | Planned | Yes | Yes (cloud) |
| **ML Integration** | Manual | Manual | Native | Manual | Manual |
| **India Broker** | Custom | None | Custom | Custom | None |

### 27.2 Performance Benchmarks

```
For 10yr daily data + 50 stocks (1M bars):
Backtrader:     ~3 minutes
VectorBT:       ~0.2 seconds
NautilusTrader: ~0.02 seconds
```

### 27.3 NSE Code Examples

```python
# Backtrader NSE feed:
class NSEPandasData(bt.feeds.PandasData):
    params = (('datetime', None), ('open', 'open'), ('high', 'high'),
              ('low', 'low'), ('close', 'close'), ('volume', 'volume'),
              ('openinterest', -1),)

# VectorBT RSI strategy:
rsi = vbt.RSI.run(close, window=14)
entries = rsi.rsi_crossed_below(30)
exits   = rsi.rsi_crossed_above(70)
pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.001)
```

### 27.4 Common Backtesting Mistakes

| Mistake | Fix |
|---------|-----|
| Look-ahead bias | Use `.shift(1)` on all feature columns |
| Survivorship bias | Use NSE historical constituent lists |
| No transaction costs | Apply full India cost model trade-by-trade |
| Wrong signal timing | Signal from bar N enters at bar N+1 open |

---

## SECTION 28 — Advanced Risk Management

### 28.1 Kelly Criterion

```
f* = (p*b - q) / b   where b = avg_win/avg_loss
Half-Kelly: f*/2 (recommended)
```

```python
def kelly_fraction(win_rate, win_loss_ratio, safety_factor=0.5):
    b, p, q = win_loss_ratio, win_rate, 1-win_rate
    f_star = (p * b - q) / b
    return max(0.0, f_star * safety_factor)
```

### 28.2 CVaR

```
CVaR_α = E[L | L > VaR_α] = mean of worst (1-α) fraction of returns
```

```python
def var_cvar(returns, confidence=0.95):
    sorted_ret = np.sort(returns)
    cutoff_idx = int((1 - confidence) * len(sorted_ret))
    var  = -sorted_ret[cutoff_idx]
    cvar = -sorted_ret[:cutoff_idx].mean()
    return var, cvar
```

### 28.3 Drawdown Analytics

```python
def drawdown_metrics(equity):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    annual_ret = equity.pct_change().mean() * 252 * 25
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.inf
    return {
        'max_drawdown_pct': round(max_dd * 100, 2),
        'calmar_ratio':     round(calmar, 3),
    }
```

### 28.4 ATR-Based Stop Loss

```python
def atr_stop_loss(entry_price, atr, direction, multiplier=1.5):
    return entry_price - multiplier * atr if direction == 'long' else entry_price + multiplier * atr

def atr_position_size(capital, risk_pct, entry_price, atr, multiplier=1.5):
    risk_amount = capital * risk_pct
    stop_distance = multiplier * atr
    shares = int(risk_amount / stop_distance)
    max_shares = int(capital * 0.05 / entry_price)
    return min(shares, max_shares)
```

### 28.5 India Cost Model — Full Formula

```python
def india_trade_cost(trade_value, broker='zerodha', product='intraday', option_premium=0.0):
    if broker == 'shoonya':
        brokerage_per_leg = 0.0
    else:
        brokerage_per_leg = min(0.0003 * trade_value, 20.0)
    brokerage_total = brokerage_per_leg * 2
    
    if product == 'delivery':    stt = 0.001 * trade_value
    elif product == 'intraday':  stt = 0.00025 * trade_value
    elif product == 'fo_futures': stt = 0.0001 * trade_value
    elif product == 'fo_options': stt = 0.0005 * option_premium
    else:                         stt = 0.001 * trade_value
    
    exchange_charges = 0.0000345 * trade_value * 2
    sebi_charges = 0.000001 * trade_value * 2
    gst = 0.18 * (brokerage_total + exchange_charges + sebi_charges)
    stamp_duty = 0.00015 * trade_value if product == 'delivery' else 0.00003 * trade_value
    
    total = brokerage_total + stt + exchange_charges + sebi_charges + gst + stamp_duty
    return {
        'brokerage': round(brokerage_total, 2), 'stt': round(stt, 2),
        'exchange_charges': round(exchange_charges, 2),
        'sebi_charges': round(sebi_charges, 4),
        'gst': round(gst, 2), 'stamp_duty': round(stamp_duty, 2),
        'total': round(total, 2), 'cost_pct': round(total / trade_value * 100, 4),
    }
```

---

## SECTION 29 — Alternative Data Sources (India, Free)

### 29.1 Google Trends

```python
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-IN', tz=330, timeout=(10, 25))

def get_stock_search_trend(company, ticker, timeframe='today 3-m'):
    keywords = [f'{company} NSE', f'buy {ticker}', f'{ticker} share price']
    pytrends.build_payload(keywords[:5], geo='IN', timeframe=timeframe)
    import time; time.sleep(1)
    interest = pytrends.interest_over_time()
    if interest.empty: return pd.DataFrame()
    interest['composite'] = interest.mean(axis=1)
    return interest
```

### 29.2 FinBERT Sentiment on Tweets

```python
from transformers import pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert")

def tweet_sentiment_score(tweets):
    scores = []
    for tweet in tweets:
        result = finbert(tweet.text[:512])[0]
        label, conf = result['label'], result['score']
        score = conf if label == 'positive' else (-conf if label == 'negative' else 0)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0
```

### 29.3 NSE Bulk/Block Deals

```python
def get_nse_bulk_deals(from_date, to_date):
    session = requests.Session()
    session.get('https://www.nseindia.com', headers={'User-Agent': 'Mozilla/5.0'})
    url = 'https://www.nseindia.com/api/historical/bulk-deals'
    r = session.get(url, params={'from': from_date, 'to': to_date},
                    headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'})
    return pd.DataFrame(r.json().get('data', []))
```

### 29.4 FII/DII Daily Net Buy/Sell

```python
def get_fii_dii_daily(from_date, to_date):
    session = requests.Session()
    session.get('https://www.nseindia.com', headers={'User-Agent': 'Mozilla/5.0'})
    r = session.get('https://www.nseindia.com/api/fiidiiTradeReact',
                    headers={'User-Agent': 'Mozilla/5.0'},
                    params={'startDate': from_date, 'endDate': to_date})
    df = pd.DataFrame(r.json())
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['fii_net'] = df['fiiBuy'].astype(float) - df['fiiSell'].astype(float)
    df['dii_net'] = df['diiBuy'].astype(float) - df['diiSell'].astype(float)
    return df.set_index('date')[['fii_net', 'dii_net']].sort_index()
```

### 29.5 India Weather (Open-Meteo, Free)

```python
def get_india_weather(city='Mumbai', days=7):
    CITIES = {'Mumbai': (19.08, 72.88), 'Delhi': (28.67, 77.22)}
    lat, lon = CITIES.get(city, CITIES['Mumbai'])
    params = {'latitude': lat, 'longitude': lon,
              'daily': 'precipitation_sum,temperature_2m_max',
              'timezone': 'Asia/Kolkata', 'forecast_days': days}
    return requests.get('https://api.open-meteo.com/v1/forecast', params=params).json()
```

### 29.6 BSE Announcements XML Feed

```python
def get_bse_announcements(days_back=1):
    from datetime import datetime, timedelta
    to_date = datetime.now().strftime('%Y%m%d')
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
    params = {'strCat': '-1', 'strType': 'C', 'strScrip': '', 'strSearch': 'P',
              'strToDate': to_date, 'strFromDate': from_date, 'strOffset': '0'}
    r = requests.get(url, params=params,
                     headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/'})
    return pd.DataFrame(r.json().get('Table', []))
```

### 29.7 NSE Insider Trading Data

```python
def get_nse_insider_trades(symbol):
    session = requests.Session()
    session.get('https://www.nseindia.com', headers={'User-Agent': 'Mozilla/5.0'})
    url = f'https://www.nseindia.com/api/corporates-pit?symbol={symbol}&market=equities'
    r = session.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'})
    return pd.DataFrame(r.json().get('data', []))
```

---

## SECTION 30 — Communities, Certifications & Ecosystem

### 30.1 Professional Certifications

| Certification | Cost | Relevance |
|--------------|------|-----------|
| **QuantInsti EPAT** | ₹2-3L | Highest — algo trading India |
| **CQF** | £4,000 | Very High — ML finance |
| **FRM (GARP)** | ~$1,200 | High — risk models |
| **NSE NCFM** | ₹7,670 | High — India compliance |
| **CFA Level 1** | ~$1,000 | Medium — portfolio theory |

### 30.2 Key Communities

| Community | URL | Use |
|-----------|-----|-----|
| r/algotrading | reddit.com/r/algotrading | Strategy ideas, broker API issues |
| Zerodha TradingQnA | tradingqna.com | India-specific API/compliance questions |
| OpenAlgo Discord | discord.gg/openalgo | Indian broker API community |
| Quantocracy | quantocracy.com | Daily quant blog feed |

### 30.3 Essential GitHub Repositories

```python
REPOS = {
    'FinRL':          'https://github.com/AI4Finance-Foundation/FinRL',
    'NautilusTrader': 'https://github.com/nautechsystems/nautilus_trader',
    'iTransformer':   'https://github.com/thuml/iTransformer',
    'TimesNet':       'https://github.com/thuml/TimesNet',
    'PatchTST':       'https://github.com/yuqinie98/PatchTST',
    'Chronos':        'https://github.com/amazon-science/chronos-forecasting',
    'TimesFM':        'https://github.com/google-research/timesfm',
    'jugaad-data':    'https://github.com/jugaad-py/jugaad-data',
    'ML4T (Jansen)':  'https://github.com/stefan-jansen/machine-learning-for-trading',
    'awesome-quant':  'https://github.com/wilsonfreitas/awesome-quant',
}
```

### 30.4 Academic Paper Sources

```
ArXiv:          https://arxiv.org/search/?query=reinforcement+learning+trading
Papers With Code: https://paperswithcode.com/task/time-series-forecasting
SSRN:           https://www.ssrn.com/index.cfm/en/  (search "NSE India")
Semantic Scholar: https://www.semanticscholar.org/search?q=transformer+time+series+finance
```

### 30.5 Indian Market Calendar — Key Risk Events

```python
NSE_RISK_CALENDAR = {
    'RBI_MPC_MEETINGS':     ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'],
    'UNION_BUDGET':         'First week of February',
    'QUARTERLY_RESULTS':    ['Apr 15-May 15', 'Jul 15-Aug 15', 'Oct 15-Nov 15', 'Jan 15-Feb 15'],
    'NIFTY_MONTHLY_EXPIRY': 'Last Thursday of every month',
    'NIFTY_WEEKLY_EXPIRY':  'Every Thursday',
    'US_FOMC':              '8x per year — high INR volatility',
}

RISK_REDUCTION = {
    'day_before_rbi_policy':   0.5,  # reduce to 50% normal size
    'day_before_union_budget': 0.3,
    'results_week_for_stock':  0.7,
    'monthly_expiry_week':     0.8,
    'us_fomc_day':             0.7,
}
```

---

## APPENDIX — Quick Reference: All Critical Formulas

```
=== TRANSFORMERS ===
Attention:      softmax(QK^T / sqrt(d_k)) * V
TFT GRN:        LayerNorm(a + GLU(ELU(W*a + b)))
PatchTST:       N = floor((T-P)/S) + 2 patches
iTransformer:   attention on variate dimension (not time)

=== RL ALGORITHMS ===
PPO clip:       min(r*A, clip(r, 1-ε, 1+ε)*A) - c1*VF + c2*H
SAC soft-Bellman: Q*(s,a) = r + γ*E[V*(s')] + α*H
TD3 target smooth: ã' = clip(μ(s') + clip(ε,-c,c), lo, hi)
Kelly:          f* = (p*b - q) / b; Half-Kelly: f*/2

=== OPTIONS ===
Black-Scholes:  C = S*N(d1) - K*e^{-rT}*N(d2)
d1:             (ln(S/K) + (r+σ²/2)*T) / (σ*sqrt(T))
Put-Call Parity: C - P = S - K*e^{-rT}
Delta_call:     N(d1); Gamma: N'(d1)/(S*σ*sqrt(T))
Theta:          -(S*N'(d1)*σ)/(2*sqrt(T)) - r*K*e^{-rT}*N(d2)
Vega:           S*sqrt(T)*N'(d1)
Heston:         dv = κ(θ-v)dt + ξ*sqrt(v)*dW_v

=== EXECUTION ===
VWAP:           sum(P*V) / sum(V)
Kyle lambda:    ΔP = λ * order_flow
Almgren-Chriss: x_k* = X * sinh(κ*(T-t_k)) / sinh(κ*T)
IS:             delay + market_impact + timing_risk + opp_cost

=== RISK ===
ATR stop:       entry - (multiplier * ATR)
Position size:  (capital * risk_pct) / (ATR_pts * multiplier)
CVaR_95%:       mean of worst 5% of returns
Drawdown:       (peak - current) / peak
India cost:     ~0.15-0.20% round-trip (Zerodha intraday)

=== FEATURES ===
RSI:            100 - 100/(1 + AvgGain/AvgLoss) [14-period]
MACD:           EMA(12) - EMA(26); Signal=EMA(MACD,9)
BB %B:          (close - lower) / (upper - lower)
ATR:            Wilder_smooth(max(H-L, |H-Cp|, |L-Cp|), 14)
VWAP dev:       (close - VWAP) / ATR
OBV:            cumsum(sign(dClose) * volume)
OBI:            (bid_vol - ask_vol) / (bid_vol + ask_vol)
Time-of-day:    sin(2π*t/375), cos(2π*t/375)
GARCH(1,1):     σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
Kalman update:  K = P*H'*(H*P*H' + R)^{-1}; x = x_pred + K*(z-H*x_pred)
```

---

*End of B5 — Sections 21–30 Deep Expansion*

---

# PART C: FIXES AND NEW ADDITIONS LOG

*Source: DB_FIXES_AND_ADDITIONS.md*

> **Purpose:** This section documents every correction, removal, replacement, and addition
> needed to bring the v2 database into sync with the rated/ranked research.
> All fixes have been applied inline in Part A above. This section preserves the full
> documentation of what was changed and why, for reference and audit purposes.

---

## TABLE OF CONTENTS (Part C)

1. [DEPRECATED TOOLS — Remove or Replace](#1-deprecated-tools--remove-or-replace)
   - 1.1 AWS FinSpace → Replace with QuestDB
   - 1.2 Backtrader → Add Deprecation Warning
   - 1.3 BloombergGPT → Add Honest Caveat
   - 1.4 AutoGPT for Trading → Replace with TradingAgents / CrewAI
   - 1.5 VectorBT Open-Source → Add Caveat
   - 1.6 FinRL — Production-Ready Claim → Correct to Research-Only

2. [RATING CORRECTIONS](#2-rating-corrections)
   - 2.1 NautilusTrader Star Count
   - 2.2 FinRL Rating Downgrade
   - 2.3 QuantConnect Lean Assessment Upgrade
   - 2.4 Qlib — Add RD-Agent Note
   - 2.5 Backtrader in Tier 1 Table

3. [MISSING ADDITIONS](#3-missing-additions)
   - 3.1 PyBroker (Section 1 Tier 2 + Section 27)
   - 3.2 Mamba-SSM (Section 21 + Section 27 reference)
   - 3.3 pykan / KAN Networks (Section 21)
   - 3.4 OpenAlgo (Section 1 Tier 3 + Section 12)
   - 3.5 Polars DataFrame Library (Section 9 Infrastructure)
   - 3.6 ChromaDB (Section 9 Infrastructure — expand existing entry)
   - 3.7 Continual Learning / EWC (Section 22 + Section 28)
   - 3.8 SEBI August 2025 Deadline (Section 14)

---

## 1. DEPRECATED TOOLS — Remove or Replace

---

### 1.1 AWS FinSpace — REPLACE WITH QUESTDB

**Location in database:** Section 1 (GitHub Repositories) is NOT directly affected — AWS FinSpace is not listed there. However, it is mentioned in adjacent infrastructure context. The primary fix is in **Section 9 (Infrastructure)** of AI_Trading_Ideas_Rated_Ranked.txt — the database should reflect this.

**Action:** In any future database section discussing managed financial data platforms or time-series storage, the following replacement applies. Also, a deprecation notice should be added to **Section 9** if/when AWS FinSpace ever appears:

**REMOVE this text (if it appears in any section):**
```
AWS FinSpace — Managed financial data lake for storing, cataloguing, and
analyzing financial data with built-in time-series analytics.
```

**REPLACE WITH:**
```
### AWS FinSpace — DEPRECATED: DO NOT USE

**Status:** SHUTTING DOWN — AWS announced October 2026 end-of-life.
**Rating:** 2/10

Any new project using AWS FinSpace is inheriting a dead-end dependency. AWS
is terminating the service. Migration path for existing users is manual export
only. The CapitalIQ dataset connector (FinSpace Data API) will also break.

**Replacement: QuestDB** (rated 10/10 technical, 8/10 India retail usefulness)

QuestDB is the correct open-source replacement for time-series financial data:
- Write speed: 4.3M rows/second benchmark
- SQL interface with ASOF JOIN (time-based joins) and LATEST ON (latest tick per symbol)
- Denser storage than InfluxDB, faster queries than TimescaleDB
- Free single-node tier — runs on any Linux VPS (AWS Mumbai, DigitalOcean Bangalore)
- No cloud lock-in; self-hosted and fully under your control

Install: `pip install questdb` (Python client)
Docker: `docker pull questdb/questdb`
Docs: https://questdb.io/docs/

India relevance: HIGH — works on ₹500/month VPS, no AWS dependency
```

---

### 1.2 Backtrader — ADD DEPRECATION WARNING

**Location in database:** Multiple locations:
- **Section 1, Tier 1 table** — listed as "21,000+ | Backtrader"
- **Section 27** — has a full dedicated entry with code example

#### Fix A: Section 1, Tier 1 Table

**FIND this row:**
```
| 21,000+ | **Backtrader** | https://github.com/mementum/backtrader | Most popular Python backtesting; event-driven; large community |
```

**REPLACE WITH:**
```
| 21,000+ | **Backtrader** ⚠️ DEPRECATED | https://github.com/mementum/backtrader | LAST RELEASE: May 2019. Star count is legacy from tutorial era. Do NOT start new projects. Use PyBroker (ML), NautilusTrader (production), or VectorBT (vectorized research) instead. |
```

#### Fix B: Section 27, Backtrader entry

**FIND this text block:**
```
### Backtrader (21K Stars)
- **GitHub:** https://github.com/mementum/backtrader
- **Docs:** https://www.backtrader.com/docu/
- **Best For:** Event-driven simulation, options, custom indicators, multiple timeframes
- **Weakness:** Slow on large datasets. Main dev (mementum) largely inactive.
- **NSE Integration:** Use Angel One historical data feed. Custom NSE data feed class needed.
```

**REPLACE WITH:**
```
### Backtrader (21K Stars) — ⚠️ EFFECTIVELY ABANDONED

- **GitHub:** https://github.com/mementum/backtrader
- **Rating:** 4/10
- **CRITICAL WARNING:** Last PyPI release was version 1.9.74.123 on **May 30, 2019**.
  Snyk classifies as Inactive. Creator considers it "complete" (i.e., not maintained).
  The 21,000 stars reflect historical popularity from 2015–2019 tutorials — not current health.
  **Do not start new projects on Backtrader.**

- **Why stars are misleading:** Books and tutorials written during Backtrader's active years
  still rank highly in Google searches. Beginners read those tutorials, star the repo, but
  the codebase has not been updated for Python 3.10+, modern pandas, or any post-2019 library.
  Some features silently produce wrong results with newer NumPy versions.

- **Community forks that survive:** `backtrader2` and `cloudQuant/backtrader` have minor
  patches, but neither has active core development. Use only for maintaining existing code.

- **Replacement decision tree:**
  - Building ML-based strategy and need walk-forward? → **PyBroker** (Section 27)
  - Need fast parameter sweeps / signal screening? → **VectorBT** (Section 27)
  - Building production live trading system? → **NautilusTrader** (Section 27)
  - Need institutional-quality backtesting with options? → **QuantConnect Lean** (Section 27)
```

---

### 1.3 BloombergGPT — ADD HONEST CAVEAT

**Location in database:** **Section 5, "BloombergGPT (Finance Foundation Model)"**

**FIND this text block:**
```
### BloombergGPT (Finance Foundation Model)
- **Title:** BloombergGPT: A Large Language Model for Finance
- **Authors:** Shijie Wu et al. (Bloomberg)
- **ArXiv:** https://arxiv.org/abs/2303.17564
- **Summary:** 50B parameter LLM trained on 363B tokens of financial text. Outperforms general LLMs on financial NLP tasks.
- **Why Critical:** Benchmark reference. Shows what's possible with finance-specific LLM training.
```

**REPLACE WITH:**
```
### BloombergGPT (Finance Foundation Model) — ⚠️ NOT PUBLICLY ACCESSIBLE

- **Title:** BloombergGPT: A Large Language Model for Finance
- **Authors:** Shijie Wu et al. (Bloomberg)
- **ArXiv:** https://arxiv.org/abs/2303.17564
- **Rating:** 3/10 for practical use (hype without access)
- **CRITICAL CAVEAT:** Bloomberg **has never released BloombergGPT publicly** and has
  no announced plans to do so. It is a research paper demonstrating internal capability,
  not a usable tool. As of May 2026, the weights, API, and inference service are all
  Bloomberg-internal only. No public access, no API pricing, no waitlist.
- **Why It Still Matters:** Benchmark reference only. Open-source alternatives (FinBERT,
  FinGPT, FinLLaMA) have now largely caught up.

**Practical Replacements (use these instead):**

1. **FinBERT** (ProsusAI) — Rated 8/10
   - 110M parameters, runs on CPU
   - Classifies text as positive/negative/neutral with 85%+ accuracy on financial text
   - `pip install transformers` + `from transformers import pipeline`
   - Model: `ProsusAI/finbert` on HuggingFace
   - Best for: Fast batch sentiment scoring of NSE company announcements

2. **FinGPT** (AI4Finance Foundation) — Rated 8/10
   - GitHub: https://github.com/AI4Finance-Foundation/FinGPT (14K+ stars, active)
   - Multiple variants: FinGPT-v3.3 (LLaMA-based), FinGPT-Forecaster
   - Fine-tuned on financial news, SEC filings, earnings calls, social media
   - `pip install fingpt` or HuggingFace: `FinGPT/fingpt-sentiment_llama2-13b_lora`
   - Best for: Local deployment, no API costs, India BSE announcements

3. **FinLLaMA** (Meta Llama-based) — Rated 8/10
   - Llama 2/3 fine-tuned on 10-K, earnings calls, analyst reports
   - HuggingFace: various `fin-llama` variants (7B and 13B)
   - Best for: Document summarization, question answering on Indian company filings
```

---

### 1.4 AutoGPT for Trading — REPLACE WITH TRADINGAGENTS / CREWAI

**Location in database:** AutoGPT may be referenced in Section 21 or Section 22.

**FIND any occurrence of:**
```
AutoGPT for trading
```
or
```
AutoGPT trading bot
```

**REPLACE WITH / ADD NOTE:**
```
### AutoGPT for Trading — ⚠️ OBSOLETE APPROACH

- **Rating:** 3/10
- **Status:** The 2023 AutoGPT trading bot craze is over. Agents hallucinated broker APIs,
  made up market data, looped endlessly without executing trades, and could not correctly
  model lot sizes or exchange hours. No independently verified profitable deployments exist.
- **India relevance:** N/A — do not build on this pattern.

**Modern Replacement Approaches (use these instead):**

#### TradingAgents (Multi-Agent Framework) — Rated 7/10
- Paper: "TradingAgents: Multi-Agents LLM Financial Trading Framework" (2024)
- Architecture: Multiple LLM agents with specialized roles (fundamental analyst,
  technical analyst, news analyst, risk manager, trader). Agents debate and reach
  consensus. Built on LangGraph (stateful, not AutoGPT's stateless loops).
- Key improvement over AutoGPT: Structured roles + debate mechanism reduces hallucination.
- Install: `pip install langgraph langchain-openai`

#### CrewAI for Trading Research — Rated 7/10
- GitHub: https://github.com/joaomdmoura/crewAI
- Best use case: Research automation, NOT live signal generation.
  Appropriate tasks: automated earnings call summarization, research report
  generation, strategy hypothesis documentation, market regime commentary.
- CrewAI v0.80+ flows enable proper state management.
- India use: Automate daily BSE filing summaries → feed as features to ML model.
- Install: `pip install crewai crewai-tools`

**Architectural principle:** Use LLMs as research accelerators, not as autonomous trade
executors. The signal quality must come from your quantitative models. LLM adds speed
to research iteration, not alpha to live trading.
```

---

### 1.5 VectorBT Open-Source — ADD CAVEAT

**Location in database:** **Section 1, Tier 2 table** and **Section 27**

#### Fix A: Section 1, Tier 2 Table

**FIND this row:**
```
| 5,000+ | **VectorBT** | https://github.com/polakowo/vectorbt | Blazing-fast vectorized backtesting. Test 1000s of params in seconds |
```

**REPLACE WITH:**
```
| 5,000+ | **VectorBT** ⚠️ OSS Stagnant | https://github.com/polakowo/vectorbt | Vectorized backtesting (signal research). NOTE: Open-source version unmaintained — active dev moved to VectorBT PRO (~$20/month commercial). For ML strategies, prefer PyBroker. |
```

#### Fix B: Section 27, VectorBT entry

**REPLACE WITH:**
```
### VectorBT (5K Stars) — ⚠️ Open-Source Version Effectively Unmaintained

- **GitHub:** https://github.com/polakowo/vectorbt
- **Rating:** 6/10 (open-source) | 7.5/10 (VectorBT PRO)
- **Critical Note:** Active development has migrated to **VectorBT PRO** (closed-source
  commercial product, ~$20/month). The open-source repo has seen minimal commits since
  2022. Known installation issues with Python 3.10+ and recent NumPy/pandas versions.

- **What VectorBT OSS still does well:** Fully vectorized NumPy/Numba backtesting —
  you can run 10,000 parameter combinations in under a minute. Excellent for rapid
  signal screening and parameter sweeps.

- **Known weaknesses (OSS version):**
  - ML model integration is awkward (not designed for sklearn/XGBoost pipelines)
  - No live trading support
  - Memory-intensive for very large parameter grids
  - Python 3.5/3.6 design origins cause compatibility issues with modern environments
  - No active maintainer for bug reports

- **When to use OSS VectorBT:** Signal screening only — rapid evaluation of 1000+
  indicator parameter combinations before committing to a full ML pipeline.

- **Better alternatives for ML strategies:** Use **PyBroker** instead (Section 27 below).
```

---

### 1.6 FinRL — Production-Ready Claim — CORRECT TO RESEARCH-ONLY

**Location in database:** **Section 1, Tier 1 table**, **Section 3 (FinRL Academic Paper)**, and **Section 22 (FinRL Ecosystem)**

#### Fix A: Section 1, Tier 1 Table

**REPLACE ROW WITH:**
```
| 10,000+ | **FinRL** ⚠️ Research Only | https://github.com/AI4Finance-Foundation/FinRL | DRL framework (NeurIPS 2020). DQN, PPO, SAC, A2C, TD3. IMPORTANT: Research/academic use only — not production-ready. Survivorship bias in backtests. No verified live P&L published. |
```

#### Fix B: Section 3, FinRL Academic Paper entry

**REPLACE WITH:**
```
### FinRL (The Standard Reference) — ⚠️ Research Platform, Not Production-Ready

- **Corrected Rating:** 6/10 overall | Research quality: 7/10 | Production readiness: 3/10

- **HONEST CAVEAT — Read Before Using:**
  FinRL's own documentation explicitly states: *"treat FinRL as an experimental platform
  rather than a profit guarantee."* Known production limitations:
  1. **Survivorship bias:** Historical data excludes delisted stocks, inflating backtest results
  2. **Low signal-to-noise ratio:** Financial markets have very low SNR — DRL reward signals
     are extremely noisy, making stable policy learning difficult
  3. **Policy instability:** Small hyperparameter changes → large performance swings.
  4. **No verified live P&L:** No independently verified, real-capital live trading performance
     has been published by the FinRL team as of May 2026.

- **What FinRL IS good for:**
  - Academic sandbox for comparing DRL algorithm variants (PPO vs SAC vs TD3)
  - Learning DRL concepts in a financial context
  - Prototyping RL environments before building custom environments

- **Practical Note:** For RL position sizing, use **Stable-Baselines3 (SB3) directly**
  with a custom NSE environment. Do not use FinRL's environments for production.
```

#### Fix C: Section 22, FinRL Key Results table

**REPLACE WITH:**
```
#### FinRL Reported Results — ⚠️ Treat With Caution

- SPY (S&P 500) buy-and-hold: Sharpe 0.79
- FinRL PPO on DJIA 30: Sharpe 1.27 (60% improvement vs buy-and-hold)
- FinRL ensemble (A2C+PPO+DDPG): Sharpe 1.46

**Caveat:** These results use a specific test period (2019–2020) that includes a market
crash and recovery — conditions particularly favorable to RL-style adaptive agents.
Survivorship bias in DJIA stock selection inflates results. Do not use these as
benchmarks for what to expect from RL on NSE data.
```

---

## 2. RATING CORRECTIONS

---

### 2.1 NautilusTrader — Star Count Correction

The star count in the Tier 1 table is correct at 22,663. Fix needed in **Section 27**:

**REPLACE Section 27 entry header WITH:**
```
### NautilusTrader (~22,000 Stars, May 2026)

- **GitHub:** https://github.com/nautechsystems/nautilus_trader
- **Docs:** https://nautilustrader.io/
- **Rating:** 8/10 overall | Architecture: 10/10 | India readiness: 6/10
- **Star Note:** Stars surged from ~5,000 (2023 estimates) to ~22,000 (2025 actual).

- **Best For:** Production trading — the Rust core eliminates the simulation-to-reality gap.
  Backtest code and live trading code are **identical**. Nanosecond precision in backtests.

- **Confirmed broker integrations (2025):** Interactive Brokers, Binance, Databento, Bybit,
  OKX, dYdX. **No native NSE/Indian broker adapters** — requires custom implementation.

- **Honest caveats:**
  - API breaking changes between releases (check changelog before upgrading)
  - Python 3.11 support ended October 2025 — use Python 3.12+
  - Steep learning curve; not for beginners
  - Building an NSE adapter requires Rust familiarity or careful Python bridge work

- **India Relevance:** HIGH POTENTIAL — best architecture for production deployment
  once an NSE/Zerodha adapter is built. This is the end-goal production platform.
```

---

### 2.2 FinRL — Rating Correction

| Location | Old Rating | New Rating |
|----------|-----------|-----------|
| Section 1, Tier 1 table description | Implicitly "production-ready" | Research only, 6/10 overall |
| Section 3, paper entry | "Why Critical" implied production use | Corrected with honest caveat |
| AI_Trading_Ideas_Rated_Ranked.txt (source) | 9/10 (initial) | 6/10 confirmed |

---

### 2.3 QuantConnect Lean — Assessment Upgrade

**Section 1, Tier 1 Table — REPLACE ROW WITH:**
```
| 19,006 | **QuantConnect Lean** | https://github.com/quantconnect/lean | Cloud + local backtesting. 483K users, 100+ hedge fund clients, $45B+ monthly notional. Self-hostable (LEAN engine, air-gapped). FIX 5.0 SP2 for institutions. Rating: 9/10 |
```

**Section 17, QuantConnect learning resource — REPLACE ROW WITH:**
```
| **QuantConnect** | https://www.quantconnect.com/ | Cloud + local backtesting. **483,000+ users** (updated 2025), 100+ hedge fund clients, $45B+ notional/month, 500K+ backtests/month. Mia V2 AI: writes LEAN algorithm code from natural language. Free tier available. Rating: 9/10 |
```

**Section 27, QuantConnect Lean entry — REPLACE WITH:**
```
### QuantConnect Lean (~19,000 Stars) — Upgraded Assessment: 9/10

- **GitHub:** https://github.com/quantconnect/lean
- **Rating:** 9/10 (research-to-production pipeline)
- **Scale (2025):** 483,000 registered users | 100+ hedge fund clients |
  $45B+ monthly notional trading volume | 500,000+ backtests/month

- **What makes LEAN exceptional:**
  - End-to-end pipeline: research → backtest → paper trading → live trading
  - Self-hostable (air-gapped deployment — no data leakage)
  - FIX 5.0 SP2 for institutional clients
  - Mia V2 AI assistant: writes LEAN algorithm code from natural language

- **Honest weaknesses:**
  - IDE instability noted in ~19% of Trustpilot reviews
  - December 2025 known bug in warm-up period for options strategies
  - NSE/Indian market data: limited direct integration; third-party data providers needed

- **India Relevance:** MEDIUM for direct NSE trading, HIGH as architectural reference.
```

---

### 2.4 Qlib — Add RD-Agent Integration Note

**Section 1, Tier 1 Table — REPLACE ROW WITH:**
```
| 16,600+ | **Microsoft Qlib** | https://github.com/microsoft/qlib | Full AI quant pipeline: alpha seeking → risk → portfolio → execution. **2025 UPDATE:** RD-Agent integration automates full quant R&D loop — NeurIPS 2025 showed 2x annualized returns vs classical factors using 70% fewer factors. Stars corrected to ~16,600 (not 35K). |
```

**ADD this note wherever Qlib is discussed:**
```
### Qlib 2025 Update: RD-Agent Integration

Microsoft released **RD-Agent** (github.com/microsoft/RD-Agent) in 2025, which integrates
with Qlib as a multi-agent AI system that automates the full quantitative R&D loop:

1. **Automated factor hypothesis generation** — LLM agents propose new alpha factors
2. **Automated backtesting** — factors are tested against historical data automatically
3. **Automated factor selection** — agents select the best-performing factors
4. **Iterative refinement** — process runs in a loop, improving factors over time

**NeurIPS 2025 results (Microsoft-published):**
- 2x higher annualized returns vs classical Alpha158 factor library
- 70% fewer factors needed (more interpretable model)

**HONEST CAVEAT:** Self-reported results from the platform's developer. No independent
replication exists as of May 2026. Treat with appropriate skepticism.

**India Application:** RD-Agent could automate factor discovery for NSE Nifty 500 universe.
Requires Qlib custom data loader for NSE. High engineering effort but potentially high reward.
```

---

### 2.5 Backtrader Star Count in Comparison Table (Section 27)

**FIND this row in the comparison table:**
```
| **Stars** | 21K | 5K | 3.1K | 20K | 19K | 22.6K |
```

**REPLACE WITH:**
```
| **Stars** | 21K ⚠️ | 5K ⚠️ | 3.1K ✓ | 20K | 19K ✓ | 22.6K ✓ |
```

And add a note below the table:
```
**Star count notes:** ⚠️ = misleading/stale stars (legacy from inactive era); ✓ = current, actively maintained.
Backtrader (21K) and VectorBT OSS (5K) stars accumulated during active years — both are now
largely unmaintained. PyBroker (3.1K), QuantConnect LEAN (~19K), and NautilusTrader (~22K) are
all actively maintained with recent releases.
```

---

## 3. MISSING ADDITIONS

---

### 3.1 PyBroker — Add to Section 1 (Tier 2) and Section 27

**Section 27, PyBroker entry — REPLACE WITH:**
```
### PyBroker (3.1K Stars) — Recommended for ML Strategy Research

- **GitHub:** https://github.com/edtechre/pybroker
- **Docs:** https://www.pybroker.com/en/latest/
- **Rating:** 7.5/10 for ML backtesting | Production readiness: 4/10
- **Maintenance:** Actively maintained (v1.2.11, December 2025).
- **Install:** `pip install lib-pybroker`

- **Why PyBroker is the RIGHT tool for this project's ML research phase:**
  Walk-forward ML backtesting is a first-class feature — not bolted on.
  You define a training function, PyBroker handles the rolling windows, retraining,
  and evaluation automatically. Native integration with scikit-learn, XGBoost,
  LightGBM, PyTorch, TensorFlow.

- **Key unique feature — BCa Bootstrap:** PyBroker uses Bias-Corrected and Accelerated
  (BCa) bootstrap confidence intervals for Sharpe ratios and returns. This gives you
  statistically meaningful performance intervals, not just point estimates.

- **NSE Integration:** Use custom data source with `jugaad-data` or broker historical API.
  yfinance (`RELIANCE.NS` format) works for prototyping.

- **Weaknesses:**
  - Single-maintainer dependency risk (bus factor = 1)
  - No native live broker integration (research tool only)
  - Limited options backtesting support

- **Code Pattern:**
```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance
from xgboost import XGBClassifier

def train_xgb(train_data, target, hyperparams):
    model = XGBClassifier(**hyperparams)
    model.fit(train_data, target)
    return model

def buy_rule(ctx):
    if ctx.model('xgb').predict(ctx.df.tail(1))[0] == 1:
        ctx.buy_shares = 10

config = StrategyConfig(initial_cash=100_000, fee_type=pybroker.FeeType.PERCENT, fee=0.002)
strategy = Strategy(YFinance(), '2022-01-01', '2024-12-31', config)
strategy.add_execution(buy_rule, ['RELIANCE.NS'],
                       models=[pybroker.ModelSource('xgb', train_xgb, pretrained=False)])
result = strategy.walkforward(windows=5, train_size=0.7)
result.metrics_df  # includes BCa bootstrap Sharpe intervals
```
```

---

### 3.2 Mamba-SSM — Add to Section 21 (Advanced Transformer & Foundation Models)

**ADD this new subsection to Section 21, after the implementation recommendations code block:**

```
### Mamba (State Space Models / SSMs) — HIGH Priority for 2025–2026

- **Key Paper:** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  Gu & Dao (2023) — ArXiv: https://arxiv.org/abs/2312.00752
- **GitHub (official):** https://github.com/state-spaces/mamba
- **Install:** `pip install mamba-ssm` (requires CUDA for fast training)
- **Rating:** Maturity 6/10 | Impact 3Y: 7/10 | Solo Actionability: 7/10 | **HIGH Priority**

**What Mamba does differently from Transformers:**
Standard Transformer attention is O(L²) in sequence length. Mamba uses Selective State
Space Models (S4 variant) that process sequences in O(L) linear time. Critically, Mamba
**selects** which parts of the sequence to remember vs. forget based on the input —
unlike LSTM which has fixed forget gates.

**Finance-specific research results (2024–2025):**
- **CrossMamba:** R² up to 0.963 on 5-day S&P 500 forecasting
- **CMDMamba (2025):** Outperformed Reformer and Transformer on 4 financial datasets
- **CryptoMamba:** Better long-range dependency capture than LSTM/Transformer on crypto data
- **"Can Mamba Replace Transformers in HF Finance?" (2026):** Hybrid Mamba-Transformer
  approaches dominate pure Mamba or pure Transformer on high-frequency data

**Honest Assessment:** Mamba is not a clean knockout over Transformers. Hybrid architectures
(T-Mamba combining Transformer + Mamba blocks) show the strongest results. For standard
15-min candles with 200–500 bar lookbacks, the difference vs Transformer is smaller.

**Why HIGH priority for this project:**
1. Linear time complexity = can afford longer lookbacks (500+ bars) without memory explosion
2. Selective state space = inherently learns which market regimes/patterns to remember
3. Active research (2024–2025 papers on finance applications), worth experimenting now
4. Replace LSTM/GRU blocks in your feature extraction pipeline

**India Application:**
- Replace LSTM in sentiment processing pipeline with Mamba block
- Use for long-sequence volatility regime modeling (>500 bars lookback)
- NIFTY 50 multi-stock portfolio: Mamba handles cross-stock attention more efficiently

**Code Pattern:**
```python
# Install: pip install mamba-ssm causal-conv1d>=1.4.0
# CPU-only version: pip install mamba-ssm --no-cuda (slower but works)

from mamba_ssm import Mamba
import torch

# Drop-in replacement for LSTM/GRU in sequence model
mamba_layer = Mamba(
    d_model=64,      # dimension of your feature vector
    d_state=16,      # SSM state expansion factor
    d_conv=4,        # local convolution width
    expand=2,        # block expansion factor
)

features = torch.randn(32, 256, 64)   # (batch=32, seq_len=256 bars, features=64)
output = mamba_layer(features)         # (32, 256, 64) — same shape, richer representation
# Use output[:, -1, :] as input to classification head for signal generation
```

**Practical note:** Start with mamba-minimal (github.com/johnma2006/mamba-minimal)
for prototyping before committing to the CUDA-dependent official implementation.
```

---

### 3.3 pykan / KAN Networks — Add to Section 21

**ADD this new subsection to Section 21, after the Mamba entry:**

```
### KAN Networks (Kolmogorov-Arnold Networks) — HIGH Priority

- **Key Paper:** "KAN: Kolmogorov-Arnold Networks" — Liu et al. (MIT, 2024)
  ArXiv: https://arxiv.org/abs/2404.19756 | **Venue: ICLR 2025** (major peer-reviewed validation)
- **GitHub:** https://github.com/KindXiaoming/pykan
- **Install:** `pip install pykan`
- **Rating:** Maturity 5/10 | Impact 3Y: 8/10 | Solo Actionability: 7/10 | **HIGH Priority**

**What KANs do differently from MLPs:**
Standard MLPs have fixed activation functions on nodes (ReLU, tanh, etc.) and learnable
weights on edges. KANs invert this: **learnable activation functions on edges** (implemented
as B-splines), fixed summation on nodes. This means:
- Each connection's function is visualizable — you can see what relationship the network learned
- The network can be distilled into mathematical formulas (symbolic regression integration)
- Potentially better at discovering analytical pricing relationships
- Smaller models achieve similar performance to larger MLPs

**Finance-specific research results (2024–2025):**
- **KASPER framework (2025):** R² = 0.89, Sharpe = 12.02, MSE = 0.0001 on Yahoo Finance data
- **GCN-KAN on 100 China A-shares:** 58.26% average accuracy, outperforming GCN, MLP, LSTM
- **LSTM-KAN hybrids:** Significantly reduced prediction errors vs LSTM alone on NIFTY 50 data
- **PIKAN for portfolio optimization:** Improved cumulative returns, Sharpe, and drawdown

**Why especially important for this NSE project:**
1. **Interpretability:** SEBI audit trail requirements. KANs can show WHY each feature
   contributes to the prediction — more interpretable than XGBoost SHAP for neural approaches.
2. **Symbolic extraction:** You can distill a trained KAN into a closed-form mathematical
   formula → discover pricing relationships that become new alpha factors.
3. **Smaller models = less overfitting:** On NSE data with limited history (~10 years),
   smaller models generalize better than large MLPs.
4. **NIFTY 50 specific:** LSTM-KAN hybrid results specifically on NIFTY 50 data are promising.

**Honest assessment:** KAN performance vs deep MLP is **mixed**. The interpretability
advantage is real and valuable for regulatory compliance. The XGBoost/LightGBM ensemble
will likely still outperform KAN on tabular prediction tasks, but KAN is worth testing
as the DL baseline alternative.

**Code Pattern:**
```python
# pip install pykan
import torch
from kan import KAN

# Replace MLP classification head in your DL pipeline
model = KAN(
    width=[23, 64, 32, 3],    # 23 input features → hidden layers → 3 output classes
    grid=5,                    # B-spline grid points (higher = more expressive)
    k=3,                       # B-spline polynomial degree
    seed=42
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    pred = model(X_train_tensor)
    loss = torch.nn.CrossEntropyLoss()(pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualization — see what each connection learned
model.plot()  # shows learned activation functions per edge

# Symbolic regression — extract formula (experimental)
model.auto_symbolic()  # attempts to identify known functions (sin, x², log, etc.)
```
```

---

### 3.4 OpenAlgo — Add to Section 1 Tier 3 (Indian Market Specific) and Section 12

**Section 1, Tier 3 Table — ADD new row:**
```
| 2,000+ | **OpenAlgo** | https://github.com/marketcalls/openalgo | Multi-broker API bridge — Zerodha, Upstox, Shoonya, Fyers, Angel One through one standard interface. Webhook-based order routing. TradingView/AlgoTest → broker. **8/10 — essential for multi-broker NSE deployment** |
```

**Section 12 (Indian Broker APIs) — ADD new subsection before the comparison table:**

```
### OpenAlgo (Multi-Broker Bridge) — Recommended: 8/10

- **GitHub:** https://github.com/marketcalls/openalgo
- **Type:** Open-source broker API gateway / unified interface
- **Rating:** 8/10 | India Relevance: HIGH
- **Install:** Self-hosted (Python/Flask application, Docker available)
- **Docs:** https://docs.openalgo.in/

**What OpenAlgo solves:**
The fundamental problem with Indian algo trading is that every broker has a different
API format, authentication flow, WebSocket protocol, and order schema. Building for
Angel One means Angel-specific code; switching to Shoonya requires rewriting execution
layer. OpenAlgo normalizes all of this into one standard interface.

**Supported brokers (as of 2025):**
Zerodha, Upstox, Shoonya/Finvasia, Fyers, Angel One, Dhan, ICICI Breeze, Kotak Neo.

**How it works:**
```
Your Strategy (Python/TradingView/AlgoTest/Amibroker)
    ↓ REST/Webhook HTTP call (standardized format)
OpenAlgo Gateway (running on your local machine or VPS)
    ↓ Translates to broker-specific API calls
Zerodha / Shoonya / Angel One / Fyers
    ↓
NSE/BSE Exchange
```

**Key features:**
- Webhook-based order routing: TradingView alerts → OpenAlgo → any connected broker
- Standardized REST API for order placement, position query, and order history
- Multi-broker support: switch broker by changing one config parameter
- Paper trading mode for testing without real capital

**Integration with this project:**
If your strategy uses the `BrokerBase` abstract interface from `data/broker_base.py`,
you can add an OpenAlgo adapter that routes through the gateway — giving you flexibility
to switch brokers without code changes.

**India Relevance:** HIGH — reduces integration work across multiple brokers significantly.
```

---

### 3.5 Polars — Add to Section 9 (Infrastructure)

**Section 9 — REPLACE the Apache Arrow / Parquet entry WITH:**
```
11. APACHE ARROW / PARQUET + POLARS
    Rating: 9/10 (Polars) | 8/10 (Arrow/Parquet standalone)
    Type: Column-oriented data format, in-memory analytics, and fast DataFrame library

    APACHE ARROW / PARQUET:
    Arrow: in-memory columnar format (fast ML data loading, zero-copy between libraries).
    Parquet: compressed columnar storage. Together enable fast historical data access.
    Replace pandas CSVs with Parquet for 10x faster I/O on your candle store.

    POLARS — Standalone Recommendation (9/10):
    GitHub: https://github.com/pola-rs/polars | Stars: 32,000+
    Install: pip install polars
    Docs: https://docs.pola.rs/

    Performance benchmarks (real-world):
    - Feature engineering pipeline: 5–10x faster than pandas on typical NSE candle data
    - Group-by operations: 20–50x faster than pandas on large datasets
    - String operations: 10–30x faster (useful for symbol filtering, date parsing)
    - Memory usage: 2–3x lower than pandas for equivalent operations

    Key advantages for this project:
    1. Lazy evaluation: Polars builds a query plan before executing, enabling automatic
       query optimization (push-down predicates, parallel execution).
    2. SIMD vectorization: Uses CPU SIMD instructions for parallel computation.
    3. No GIL: Parallel execution without Python's Global Interpreter Lock.
    4. Expression API: Cleaner, more readable code than pandas chaining.

    Migration from pandas: Partial migration is fine. Use Polars for:
    - Historical data loading (Parquet files)
    - Feature engineering pipeline
    - Large-scale filtering and aggregation
    Keep pandas for: indicators (pandas-ta requires pandas), ML model input

    India Relevance: HIGH — free 5–10x speedup on any existing feature engineering.

    Code Pattern:
```python
import polars as pl

# Load NSE candle data (Parquet is 10x faster than CSV)
df = pl.read_parquet("nse_candles.parquet")

# Feature engineering — all operations in parallel by default
features = df.with_columns([
    pl.col("close").rolling_mean(14).alias("sma_14"),
    pl.col("close").rolling_std(14).alias("std_14"),
    pl.col("volume").rolling_mean(20).alias("vol_ma_20"),
    (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1bar"),
    (pl.col("close") / pl.col("nifty_close")).alias("rel_strength"),
])

# Convert to pandas only for model input (pandas-ta, sklearn)
X = features.to_pandas()

# Save back to Parquet (much faster than CSV)
features.write_parquet("features.parquet", compression="lz4")
```
```

---

### 3.6 ChromaDB — Expand Existing Entry in Section 9

**Section 9, entry #7 — REPLACE WITH:**
```
7. CHROMADB / WEAVIATE / QDRANT (VECTOR DATABASES)
   Rating: ChromaDB 7.5/10 | Qdrant 8/10 | Weaviate 7/10
   Install: pip install chromadb  (simplest, no server required for development)
   Docs: https://docs.trychroma.com/

   What vector databases do for trading:

   USE CASE 1 — REGIME ANALOG SEARCH (most valuable for NSE):
   Embed the current market state as a vector (using your 23 features or a foundation
   model embedding). Query the vector database for historically similar market states.
   Use those historical analogs to estimate the distribution of outcomes.
   This is "non-parametric regime detection" — more flexible than HMM because it
   doesn't force markets into k fixed regimes.

   Implementation sketch:
   - Encode each 15-min bar's 23 features as a 23-dimensional vector
   - Store all historical vectors in ChromaDB with their outcomes (next N-bar return)
   - At inference: find the 50 most similar historical states → look at their outcomes
   - Use outcome distribution as a non-parametric probability estimate

   USE CASE 2 — RAG FOR FINANCIAL DOCUMENTS:
   Store embeddings of BSE company announcements, SEBI circulars, earnings call transcripts.
   At trading time, retrieve relevant documents for the current stock → feed to LLM for
   context-aware sentiment.

   USE CASE 3 — NEWS CLUSTERING:
   Cluster NSE company announcements by semantic similarity. Detect when multiple
   companies in the same sector receive similar news simultaneously → sector-level signal.

   TOOL SELECTION:
   - ChromaDB: Open-source, embedded (no server), simple Python API.
     Best for: prototyping and single-node deployments.
   - Qdrant: Open-source but has production server. Better performance at scale.
   - Weaviate: Richest query language, more complex setup.

   India Relevance: MEDIUM for alternative data pipelines.
   HIGH VALUE if implementing regime analog search on NSE historical data.

   Code Pattern (ChromaDB regime analog search):
```python
import chromadb
import numpy as np

client = chromadb.PersistentClient(path="./market_states")
collection = client.get_or_create_collection("nse_regimes")

# Store historical market states with outcomes
for i, row in features_df.iterrows():
    state_vector = row[feature_cols].values.tolist()
    outcome = float(future_returns.loc[i])  # next N-bar return
    collection.add(
        embeddings=[state_vector],
        metadatas=[{"date": str(i), "outcome": outcome, "symbol": "NIFTY"}],
        ids=[str(i)]
    )

# At inference: find similar historical states
current_state = current_features[feature_cols].values.tolist()
results = collection.query(
    query_embeddings=[current_state],
    n_results=50  # 50 most similar historical states
)
# Analyze outcome distribution of similar states
analog_outcomes = [m["outcome"] for m in results["metadatas"][0]]
expected_return = np.mean(analog_outcomes)
uncertainty = np.std(analog_outcomes)  # regime uncertainty = position sizing input
```
```

---

### 3.7 Continual Learning / EWC — Add to Section 22 (RL) and Section 28 (Risk)

**Section 22, after the Meta-RL section — ADD this new subsection:**

```
### Continual Learning (Catastrophic Forgetting Prevention) — MEDIUM-HIGH Priority

**Maturity:** 5/10 | **Impact 3Y:** 7/10 | **Solo Actionability:** 6/10

**Why this matters for live NSE trading:**
When you retrain your model on recent data to adapt to new market conditions, it tends
to "forget" older patterns — this is **catastrophic forgetting** (Kirkpatrick et al., 2017).
Example: A model trained on 2022–2023 data forgets that 2008-style crashes exist. When a
similar regime returns, the model behaves as if it has never seen it.

This is the **core model drift problem** in production trading systems.

**Elastic Weight Consolidation (EWC) — Primary Recommendation:**
- **Paper:** Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" —
  PNAS 2017, ArXiv: https://arxiv.org/abs/1612.00796
- **Concept:** When adapting to new data (Task B), EWC adds a regularization term to the
  loss function that **penalizes large changes to weights that were important for old data
  (Task A)**. The Fisher information matrix identifies which weights are critical.
- **For trading:** Task A = pre-2022 data (multiple regimes). Task B = adapting to 2024–2025
  data. EWC keeps your model's ability to recognize bull/bear/crisis regimes while adapting.
- **Library:** `pip install avalanche-lib` (ContinualAI research library)

**Experience Replay (Simplest Approach — Start Here):**
Blend recent data with a random sample from all historical data in every training batch.
This is the simplest continual learning method and is surprisingly effective:
```python
# Simple experience replay for model retraining
def create_training_set(recent_data, historical_data, replay_ratio=0.3):
    """
    recent_data: last 30 days of features + labels
    historical_data: full 2-year history
    replay_ratio: fraction of training data from history
    """
    n_recent = len(recent_data)
    n_replay = int(n_recent * replay_ratio / (1 - replay_ratio))
    historical_sample = historical_data.sample(n=n_replay)
    combined = pd.concat([recent_data, historical_sample])
    return combined.sample(frac=1)  # shuffle
```

**Practical path for this project:**
1. **Immediate (use today):** Experience replay in weekly retraining — blend recent 30 days
   with random sample from full 2-year history. 30% replay ratio is a good starting point.
2. **Next step:** EWC via Avalanche library when implementing the RL agent.
3. **Advanced:** Monitor regime distance (ChromaDB analog search) — if current regime is
   novel (low similarity to all historical analogs), trigger model growth.

**Libraries:**
- `pip install avalanche-lib` — ContinualAI: EWC, Progressive Neural Networks, GEM
- `pip install sequoia` — Alternative CL library, research-grade
- Custom experience replay — just implement the function above

**India Relevance:** MEDIUM-HIGH — Indian markets have seen distinct regimes:
COVID crash (2020), infra bull run (2021–2022), rate hike volatility (2022–2023),
bull market (2023–2024). A model trained on any one period will struggle on others.
```

**Section 28 (Advanced Risk Management) — ADD this note:**

```
### Model Drift Detection (Continual Learning Context)

In addition to market regime detection (HMM, BOCPD), production systems must detect
when the **model itself** has drifted from reality. Signs of model drift:
- Sharpe ratio declining week-over-week on paper trading
- Feature SHAP values changing significantly (feature importances shift)
- Prediction confidence (softmax probability) distribution shifting

**Drift detection methods:**
- **Population Stability Index (PSI):** Track distribution of model inputs week-over-week.
  PSI > 0.2 = significant drift, trigger retraining.
- **ADWIN (Adaptive Windowing):** Online algorithm that detects distribution shifts in
  streaming data. `pip install river` → `river.drift.ADWIN`
- **Kolmogorov-Smirnov test:** Weekly KS test comparing current feature distributions
  to training distributions. p < 0.05 = distribution has shifted.

**Response to detected drift:**
1. Minor drift (PSI 0.1–0.2): Retrain model with experience replay
2. Major drift (PSI > 0.2): Full retraining + regime detection audit
3. Unknown regime (no ChromaDB analogs): Reduce position size 50%, escalate to manual review

See Section 22 (Continual Learning) for EWC and experience replay implementations.
```

---

### 3.8 SEBI August 2025 Deadline — Update Section 14

**Section 14, Implementation Timeline table — REPLACE WITH:**
```
### Implementation Timeline

| Milestone | Date | Requirement |
|-----------|------|-------------|
| SEBI Feb 10, 2025 | Feb 10, 2025 | Regulation 16C: SEBI-regulated entities solely responsible for AI output accuracy. **XAI/explainability now a legal obligation.** |
| **⚠️ August 1, 2025** | **Aug 1, 2025** | **EARLIER THAN PREVIOUSLY NOTED: Exchange approval required for ALL algorithmic strategies. Unique Algo ID mandatory for every deployed strategy.** |
| Phase 1 | Oct 1, 2025 | Broker infrastructure ready |
| Phase 2 | Oct 31, 2025 | Static IP registration enabled |
| Phase 3 | Nov 30, 2025 | Strategy approval process live |
| Phase 4 | Jan 3, 2026 | Algorithm categorization complete |
| **Full Mandate** | **Apr 1, 2026** | **All requirements mandatory** |
| 2025–2026 | Ongoing | Harmonized framework: NSE registration auto-shared to BSE/MSE/MCX/NCDEX |

**IMPORTANT:** The August 1, 2025 requirement is **more urgent** than the April 2026 full
mandate. If you are live trading, exchange approval + unique Algo ID was required from
August 2025.
```

**Section 14 — ADD this new subsection:**

```
### XAI (Explainability) Requirements — New Legal Obligation from Feb 2025

SEBI Regulation 16C (effective February 10, 2025) makes SEBI-regulated entities
**solely responsible** for AI output accuracy and regulatory compliance. This creates
a practical XAI requirement: you must be able to explain and audit your model's decisions.

**Practical compliance for this project:**

1. **SHAP for every production model** (Section 25 has implementation code)
   - Log SHAP values for every trade decision (required for audit trail)
   - Feature importance drift monitoring (SHAP value changes over time)

2. **Model cards** for every deployed model version:
   - Training data period
   - Features used
   - Validation methodology (walk-forward Sharpe)
   - Known failure modes (what market conditions break the model)

3. **Version control** for model weights (DVC or MLflow model registry)
   - Every production model version must be reproducible
   - Capability to "replay" any historical trade decision and regenerate the SHAP explanation

4. **Audit trail fields** (update `execution/trade_logger.py`):
   - `model_version` — which MLflow model run generated this signal
   - `top_3_features` — top 3 SHAP features at decision time
   - `signal_confidence` — model probability (not just binary signal)
   - `regime_at_decision` — HMM state at time of trade

5. **AxonFlow** (axonflow.com) — provides SEBI-specific AI compliance documentation tooling.

**India Relevance:** MANDATORY — non-documentation is now a compliance risk, not just
a best practice. Have this infrastructure in place before live deployment.
```

---

## SUMMARY CHECKLIST — APPLY IN ORDER

### Deprecations/Removals:
- [x] Section 1 Tier 1: Add ⚠️ to Backtrader entry, correct star table note
- [x] Section 5: Add honest caveat to BloombergGPT entry with FinBERT/FinGPT replacements
- [x] Section 22: Add AutoGPT deprecation with TradingAgents/CrewAI alternatives
- [x] Section 27: Backtrader — replace with full deprecation warning
- [x] Section 27: VectorBT — replace with caveat about OSS unmaintained status
- [x] Section 1 Tier 1 + Section 3 + Section 22: FinRL — add research-only caveat throughout

### Rating Corrections:
- [x] Section 1 Tier 1: Qlib star count corrected to ~16,600 (not 35,400+)
- [x] Section 1 Tier 1 + Section 27: NautilusTrader star count verified at ~22,000
- [x] Section 3 + Section 22: FinRL corrected to 6/10 overall, research-only
- [x] Section 1 Tier 1 + Section 17 + Section 27: QuantConnect upgraded to 9/10, 483K users
- [x] Section 27 comparison table: Add ⚠️/✓ star quality indicators

### Additions:
- [x] Section 1 Tier 2 + Section 27: PyBroker — expand existing entry with BCa bootstrap, honest caveats
- [x] Section 21: Mamba-SSM — add full entry with finance results and code
- [x] Section 21: KAN networks — add full entry with NIFTY 50 results and code
- [x] Section 1 Tier 3 + Section 12: OpenAlgo — add multi-broker bridge entry
- [x] Section 9: Polars — expand Arrow/Parquet entry with dedicated Polars subsection
- [x] Section 9: ChromaDB — expand existing entry with regime analog search pattern
- [x] Section 22 + Section 28: Continual Learning/EWC — add full entries with code
- [x] Section 14: SEBI August 2025 deadline — add to timeline, add XAI requirements section

---

*End of Part C — Fixes and New Additions Log*
*Total fixes documented: 8 deprecations/caveats | 5 rating corrections | 8 additions*
*All fixes applied inline in Part A (Sections 1–30) of this database.*




---

# PART D: RATED & RANKED REFERENCE — MASTER RANKINGS AND HONEST ASSESSMENTS
## Source: AI_Trading_Ideas_Rated_Ranked.txt (extracted May 2026, unique content only)

---

## D1. Master Ranking: Top 50 AI Trading Ideas
### Weighted: Technical merit 30% + Practical usability 30% + India relevance 20% + Innovation 20%

| Rank | Name | Score | Category |
|------|------|-------|----------|
| 1 | Renaissance Technologies (concept) | 10.0 | Quant Fund |
| 2 | Microsoft Qlib | 9.2 | Big Tech / Open Source |
| 3 | FinRL (Columbia) | 9.0 | Academic |
| 4 | Numerai | 9.0 | Global Startup |
| 5 | QuestDB | 9.0 | Infrastructure |
| 6 | Zerodha Kite + MCP Server | 9.0 | Indian Platform |
| 7 | Two Sigma (methodology) | 8.9 | Quant Fund |
| 8 | D.E. Shaw (methodology) | 8.9 | Quant Fund |
| 9 | Citadel (methodology) | 8.8 | Quant Fund |
| 10 | Man Group AlphaGPT | 8.8 | Agentic / Quant Fund |
| 11 | Optuna (hyperparameter tuning) | 8.8 | Infrastructure |
| 12 | Jane Street (methodology) | 8.7 | Quant Fund |
| 13 | Apache Kafka | 8.7 | Infrastructure |
| 14 | Redis | 8.7 | Infrastructure |
| 15 | Salesforce Moirai | 8.5 | Foundation Model |
| 16 | Amazon Chronos | 8.5 | Foundation Model |
| 17 | Sensibull | 8.5 | Indian Platform |
| 18 | FinBERT | 8.4 | Finance LLM |
| 19 | FinGPT | 8.4 | Finance LLM |
| 20 | AlgoTest | 8.3 | Indian Platform |
| 21 | NVIDIA RAPIDS cuDF | 8.3 | Big Tech / GPU |
| 22 | DVC (Data Version Control) | 8.3 | Infrastructure |
| 23 | OpenBB Copilot | 8.2 | Agentic AI |
| 24 | OpenAlgo | 8.2 | Indian Platform |
| 25 | NautilusTrader | 8.2 | Academic / Open Source |
| 26 | FinLLaMA | 8.2 | Finance LLM |
| 27 | QuantConnect / Mia V2 | 8.1 | Agentic AI / Platform |
| 28 | Apache Arrow / Parquet + Polars | 8.1 | Infrastructure |
| 29 | Oxford Man Institute (research) | 8.0 | Academic |
| 30 | Nasdaq AI Surveillance | 8.0 | Exchange |
| 31 | JPMorgan LLMSuite / IndexGPT | 8.0 | Institutional |
| 32 | Regime-Adaptive AI (HMM+LLM) | 8.0 | Emerging Tech |
| 33 | Multimodal AI for Trading | 7.9 | Emerging Tech |
| 34 | Dhan HQ API | 7.9 | Indian Platform |
| 35 | Angel One SmartAPI | 7.8 | Indian Platform |
| 36 | Shoonya (zero brokerage) | 7.8 | Indian Platform |
| 37 | MLflow | 7.7 | Infrastructure |
| 38 | Google TimesFM | 7.7 | Foundation Model |
| 39 | TimeGPT (Nixtla) | 7.6 | Foundation Model |
| 40 | ChromaDB (vector DB) | 7.6 | Infrastructure |
| 41 | Mamba / SSMs | 7.5 | Emerging Tech |
| 42 | TradingAgents framework | 7.5 | Agentic AI |
| 43 | Streak (NSE-empaneled) | 7.5 | Indian Platform |
| 44 | CrewAI for trading research | 7.5 | Agentic AI |
| 45 | WorldQuant Brain simulator | 7.5 | Academic |
| 46 | MOMENT (CMU foundation model) | 7.4 | Foundation Model |
| 47 | Causal ML (DoWhy/econml) | 7.3 | Emerging Tech |
| 48 | GNN for stock relationship graphs | 7.3 | Emerging Tech |
| 49 | RL Decision Transformer | 7.2 | Emerging Tech |
| 50 | Upstox API | 7.2 | Indian Platform |

**Avoid entirely:** AWS FinSpace (shutting down Oct 2026), AutoGPT for trading, BloombergGPT (not public), Investmint (defunct), Sentient Technologies (defunct — $143M, failed).

**Low priority:** Quantum computing for finance (5-10yr horizon), DeFi AI in India (30% tax + 1% TDS), AutoGen/Magentic-One (ecosystem churn), LangChain for signal generation.

---

## D2. Honest Assessment — What Actually Works vs What Is Overhyped

### What Genuinely Generates Alpha (Evidence-Based)

1. Many weak uncorrelated signals (Renaissance principle) — diversification of signal source > model sophistication.
2. Regime detection (HMM/clustering) before applying any model.
3. Full cost model in backtesting — India: STT + brokerage + GST + slippage. Without this, backtested returns are fictional.
4. Triple-barrier labeling vs naive next-candle labels.
5. Walk-forward validation only — never shuffle financial time-series.
6. Factor-based approaches (momentum, mean reversion, carry) with transaction cost awareness.
7. Statistical arbitrage / pairs trading — proven in Indian NSE context.

### What Is Overhyped

1. "AI predicts stock prices" — ML forecasts return distributions with Sharpe barely above 0.5 in most published research.
2. LLM-generated trading signals — LLMs hallucinate financial data.
3. Fully autonomous AI trading agents — fundamental errors in real trading (wrong lot sizes, ignoring exchange hours).
4. Quantum computing for trading — 5-10 years from practical use.
5. Single "perfect" strategy — markets are non-stationary; strategies decay.

### India-Specific Honest Observations

1. **Transaction costs consume 60-80% of backtested returns** for high-frequency strategies if not modeled correctly. Sub-0.15% edge-per-trade is unprofitable after India's STT + GST + exchange fees + slippage.
2. **SEBI 2026 empanelment requirement raises the bar** — custom-coded strategies must be registered through a broker; non-compliance means order rejection from the exchange after April 2026.
3. **NSE options markets offer better risk-reward** for systematic retail strategies than pure equity intraday. Options strategies survive India's cost structure better.
4. **Data quality is the #1 problem** — undjusted corporate actions (splits, bonuses) make every backtest beyond 6 months unreliable.
5. **The retail edge in India: India-specific alternative data** — regional news sentiment, GST filing data, social media in Hindi/regional languages that global players' English-only pipelines ignore.

### Key Insights from Institutional Funds

- **Renaissance:** Combine many weak uncorrelated signals. Diversification of signal source beats one sophisticated model.
- **Two Sigma:** Trading is a data engineering problem first, then a modeling problem. Data quality > model sophistication.
- **D.E. Shaw:** Regime-aware strategy switching — don't force one model across all market conditions.
- **Man Group:** LLMs as research accelerators, not signal generators. Hypotheses still need human domain judgment.
- **Jane Street:** Every trade is a probabilistic inference problem; live trading is the primary test, not backtesting.

---

## D3. Indian Platform Quick Reference (NSE-Specific, May 2026)

| Platform | Best For | API Cost | SEBI Status |
|----------|----------|----------|-------------|
| Zerodha Kite + MCP | Full algo integration, largest community | Rs.2,000/month | Empaneled |
| Sensibull | Options analytics, Greeks, strategy builder | Subscription | Empaneled (Zerodha) |
| AlgoTest | Options backtesting with real F&O data (2016+) | Free tier | Empaneled (paper) |
| OpenAlgo | Multi-broker API bridge (unified REST) | Free / OSS | N/A (bridge) |
| Dhan (DhanHQ) | Developer-friendly, sandbox environment | Free | Registered |
| Angel One SmartAPI | Best docs, free API, 22M+ users | Free | Registered |
| Shoonya (Finvasia) | Zero brokerage — PRIMARY for this project | Free | Registered |
| Streak | No-code NSE strategies | Subscription | NSE-Empaneled |
| Upstox | Solid API, Ratan Tata-backed | Free | Registered |

Project note: Shoonya is the primary execution broker (zero brokerage, no TOTP expiry, free API). Angel One SmartAPI is dev reference only per CLAUDE.md.
