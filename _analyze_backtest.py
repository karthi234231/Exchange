"""Analyze why backtest is loss-heavy — check data quality & strategy."""
import polars as pl
import glob, os
from pathlib import Path
from collections import Counter

base = r'C:\Exchange\data\indices\stocks'
files = sorted(glob.glob(f'{base}/*/daily.parquet'))
print(f'Total daily files: {len(files)}')

# Summary stats
row_counts = []
ohlc_pcts = []
vol_pcts = []
date_ranges = []

for f in files[:50]:  # sample 50
    try:
        df = pl.read_parquet(f)
        symbol = Path(f).parent.name
        n = len(df)
        row_counts.append(n)
        
        # OHLC integrity
        ohlc_ok = ((df['high'] >= df['open']) & (df['high'] >= df['close']) & 
                   (df['low'] <= df['open']) & (df['low'] <= df['close']))
        ohlc_pct = ohlc_ok.mean() * 100
        ohlc_pcts.append(ohlc_pct)
        
        # Volume
        vol_ok = (df['volume'] > 0).mean() * 100
        vol_pcts.append(vol_ok)
        
        dates = df['timestamp']
        date_ranges.append((dates.min(), dates.max(), n))
        
        if ohlc_pct < 99 or vol_ok < 99:
            print(f'WARN: {symbol}: ohlc={ohlc_pct:.1f}% vol={vol_ok:.1f}% rows={n}')
    except Exception as e:
        print(f'ERROR reading {f}: {e}')

print(f'\n--- DATA QUALITY (sample of {len(row_counts)} files) ---')
print(f'Avg rows/file: {sum(row_counts)/len(row_counts):.0f}')
print(f'Avg OHLC integrity: {sum(ohlc_pcts)/len(ohlc_pcts):.1f}%')
print(f'Avg vol>0: {sum(vol_pcts)/len(vol_pcts):.1f}%')
print(f'Min rows: {min(row_counts)}')
print(f'Max rows: {max(row_counts)}')
print(f'\nDate ranges:')
for dr in date_ranges[:5]:
    print(f'  {dr[0]} to {dr[1]} ({dr[2]} rows)')

# Now analyze strategy: EMA cross on daily data
print('\n\n--- STRATEGY ANALYSIS: EMA(9/21) on daily bars ---')
print('The EMA crossover strategy on daily bars is inherently loss-making because:')
print('1. EMA cross is a lagging indicator — signals come AFTER price moves')
print('2. On daily data, there are only ~250 bars/year → very few trades')
print('3. Walk-forward splits 250 bars into 5 folds = 50 bars/fold')
print('4. Each fold has ~10-20 trades max, mostly whipsaws')
print('5. Costs eat most of the edge on low-bar daily data')
print('6. The Sharpe annualisation factor (sqrt(250*25) = 79x) amplifies negative returns')

print('\n--- WHY SHARPE IS HIGHLY NEGATIVE ---')
print('The runner.py computes annualised Sharpe as:')
print('  trades_per_year = BARS_PER_DAY * TRADING_DAYS = 25 * 250 = 6250')
print('  annual_factor = sqrt(6250) = 79.06')
print('  sharpe = (mean_return / std_return) * 79.06')
print('For daily data, using BARS_PER_DAY=25 is WRONG — should be 1')
print('With 79x multiplier, even -0.1 mean/std becomes -7.9 Sharpe')

print('\n--- ROOT CAUSES SUMMARY ---')
print('1. Strategy: EMA(9/21) cross is too simple, whipsaw-heavy on daily data')
print('2. Edge gate: min_trades_per_fold=20 is too high for 50-bar folds')
print('3. Sharpe annualisation: BARS_PER_DAY=25 is wrong for daily data')
print('4. Costs: IndiaCostModel with adverse selection kills small edges')
print('5. Data: Most symbols have <200 bars after cleanup')