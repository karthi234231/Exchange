"""Run backtest on ALL available NSE intraday data."""
import os
import sys
import math
import statistics
from decimal import Decimal
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl

DATA_DIR = r"c:\Exchange\data\nifty-banknifty-intraday-data-main"

FAST_EMA = 9
SLOW_EMA = 21
INITIAL_CAPITAL = Decimal("500000")
NO_TRADE_BEFORE = (9, 30)
SQUARE_OFF_TIME = (15, 15)

CORE_COLS = ['symbol','date','time','open','high','low','close','volume','dt']

def parse_file(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip()
    n_cols = len(first.split(','))
    
    if n_cols == 8:
        df = pl.read_csv(fp, has_header=False, separator=',', infer_schema_length=10000,
            schema_overrides={
                'column_1': pl.Utf8, 'column_2': pl.Int64, 'column_3': pl.Utf8,
                'column_4': pl.Float64, 'column_5': pl.Float64, 'column_6': pl.Float64,
                'column_7': pl.Float64, 'column_8': pl.Int64,
            })
        df = df.rename({'column_1':'symbol','column_2':'date','column_3':'time',
                        'column_4':'open','column_5':'high','column_6':'low',
                        'column_7':'close','column_8':'volume'})
    else:
        df = pl.read_csv(fp, has_header=False, separator=',', infer_schema_length=10000,
            schema_overrides={
                'column_1': pl.Utf8, 'column_2': pl.Int64, 'column_3': pl.Utf8,
                'column_4': pl.Float64, 'column_5': pl.Float64, 'column_6': pl.Float64,
                'column_7': pl.Float64,
            })
        df = df.rename({'column_1':'symbol','column_2':'date','column_3':'time',
                        'column_4':'open','column_5':'high','column_6':'low',
                        'column_7':'close'})
        df = df.with_columns(pl.lit(0).alias('volume'))
    
    df = df.with_columns(
        pl.format('{} {}', pl.col('date').cast(pl.Utf8), pl.col('time')).alias('_dt_str')
    ).with_columns(
        pl.col('_dt_str').str.to_datetime(format='%Y%m%d %H:%M').alias('dt')
    ).drop('_dt_str')
    df = df.filter(
        (pl.col('time') >= '09:15') & (pl.col('time') < '15:30')
    )
    # Always return exactly CORE_COLS
    out = []
    for c in CORE_COLS:
        if c in df.columns:
            out.append(df[c])
        else:
            out.append(pl.lit(0).alias(c))
    return pl.DataFrame(out).sort('dt')


def resample_15min(df):
    df = df.with_columns(
        (pl.col('dt').dt.timestamp('ms') / (15*60*1000)).floor().cast(pl.Int64).alias('bin')
    )
    resampled = df.group_by('bin').agg([
        pl.col('dt').first().alias('ts'),
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'),
        pl.col('close').last().alias('close'),
        pl.col('volume').sum().alias('volume'),
    ]).sort('ts')
    resampled = resampled.with_columns(
        pl.from_epoch(pl.col('bin') * (15*60), time_unit='s').alias('ts')
    )
    return resampled.select(['ts','open','high','low','close','volume'])


def ema_signals(closes, fast, slow):
    n = len(closes)
    sigs = [0]*n
    af = 2.0/(fast+1)
    as_ = 2.0/(slow+1)
    ef = closes[0]
    es = closes[0]
    for i in range(n):
        pf, ps = ef, es
        ef = af*closes[i] + (1-af)*ef
        es = as_*closes[i] + (1-as_)*es
        if i < slow: continue
        if pf <= ps and ef > es: sigs[i] = 1
        elif pf >= ps and ef < es: sigs[i] = -1
    return sigs


def atr(closes, highs, lows, period=14):
    n = len(closes)
    atrs = [closes[0]*0.005]*n
    if n < 2: return atrs
    trs = []
    for i in range(1, n):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
        if len(trs) < period:
            atrs[i] = sum(trs)/len(trs)
        else:
            atrs[i] = (atrs[i-1]*(period-1) + tr)/period
    return atrs


def run_backtest(df, symbol):
    closes = df['close'].to_list()
    opens = df['open'].to_list()
    highs = df['high'].to_list()
    lows = df['low'].to_list()
    timestamps = df['ts'].to_list()
    n = len(df)
    
    signals = ema_signals(closes, FAST_EMA, SLOW_EMA)
    atrs = atr(closes, highs, lows)
    
    trades = []
    equity = [INITIAL_CAPITAL]
    no_trade_min = 9*60+30
    sqoff_min = 15*60+15
    
    i = 0
    while i < n-1:
        sig = signals[i]
        if sig == 0:
            i += 1
            continue
        
        ts = timestamps[i]
        bar_min = ts.hour*60 + ts.minute
        if bar_min < no_trade_min or bar_min >= sqoff_min:
            i += 1
            continue
        
        close_i = closes[i]
        atr_i = atrs[i] if not math.isnan(atrs[i]) else close_i*0.005
        
        # 1-bar fill delay
        fill_bar = i+1
        fill_open = opens[fill_bar]
        
        # Stale signal check (>50 bps)
        if close_i > 0:
            move_bps = abs(fill_open - close_i)/close_i*10000
            if move_bps > 50:
                i += 1
                continue
        
        entry = Decimal(str(fill_open))
        qty = max(1, int(INITIAL_CAPITAL*Decimal('0.05')/entry))
        atr_d = Decimal(str(round(atr_i, 4)))
        
        if sig > 0:
            stop = entry - atr_d*Decimal('1.5')
        else:
            stop = entry + atr_d*Decimal('1.5')
        
        # Seek exit
        exit_bar = fill_bar + 1
        exit_reason = 'force_close'
        exit_price = Decimal(str(closes[min(fill_bar, n-1)]))
        
        while exit_bar < n:
            ts_e = timestamps[exit_bar]
            em = ts_e.hour*60 + ts_e.minute
            if em >= sqoff_min:
                exit_price = Decimal(str(opens[exit_bar]))
                exit_reason = 'time_exit'
                break
            
            ch, cl, co = highs[exit_bar], lows[exit_bar], opens[exit_bar]
            if sig > 0 and cl <= float(stop):
                exit_price, exit_reason = Decimal(str(co)), 'stop'
                break
            if sig < 0 and ch >= float(stop):
                exit_price, exit_reason = Decimal(str(co)), 'stop'
                break
            
            if exit_bar - fill_bar >= 20:
                exit_price = Decimal(str(co))
                exit_reason = 'force_close'
                break
            exit_bar += 1
        else:
            exit_price = Decimal(str(closes[n-1]))
            exit_reason = 'force_close'
            exit_bar = n-1
        
        gross = sig * (exit_price - entry) * Decimal(str(qty))
        cost = (entry*Decimal(str(qty)) + exit_price*Decimal(str(qty))) * Decimal('0.001')
        net = gross - cost
        
        trades.append({
            'symbol': symbol, 'dir': sig, 'entry': float(entry), 'exit': float(exit_price),
            'qty': qty, 'gross': float(gross), 'net': float(net),
            'reason': exit_reason, 'bars': exit_bar - fill_bar,
        })
        equity.append(equity[-1] + net)
        i = exit_bar + 1
    
    if not trades:
        return {'symbol': symbol, 'trades': 0}
    
    nets = [t['net'] for t in trades]
    wins = sum(1 for p in nets if p > 0)
    n = len(trades)
    wr = wins/n
    gp = sum(p for p in nets if p > 0)
    gl = abs(sum(p for p in nets if p < 0))
    pf = gp/gl if gl > 0 else float('inf')
    
    rets = [p/float(INITIAL_CAPITAL) for p in nets]
    mu = statistics.mean(rets) if len(rets)>1 else 0
    sigma = statistics.stdev(rets) if len(rets)>1 else 0
    sharpe = (mu/sigma)*math.sqrt(25*250) if sigma > 0 else 0
    
    peak = INITIAL_CAPITAL
    max_dd = 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq)/peak
        if dd > max_dd: max_dd = dd
    
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['reason']] += 1
    
    return {
        'symbol': symbol,
        'trades': n,
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 4),
        'sharpe': round(sharpe, 4),
        'max_dd': round(max_dd, 4),
        'net_pnl': round(sum(nets), 2),
        'passes': all([sharpe>=1.0, pf>=1.2, wr>=0.4, n>=200]),
        'reasons': dict(reasons),
    }


def find_files():
    files = []
    for root, dirs, fnames in os.walk(DATA_DIR):
        for fn in fnames:
            if fn.lower().endswith(('.txt', '.csv')):
                files.append(os.path.join(root, fn))
    return files


def main():
    print("=" * 80)
    print("  NSE Full Backtest — All Available Intraday Data")
    print("=" * 80)
    
    files = find_files()
    if not files:
        print("No data files found.")
        return
    
    print(f"\nFound {len(files)} files")
    
    by_symbol = defaultdict(list)
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        by_symbol[stem.upper()].append(fp)
    
    print(f"Found {len(by_symbol)} symbols")
    print("Running backtests...")
    
    results = []
    tested = 0
    for sym, fps in sorted(by_symbol.items()):
        frames = []
        skipped = 0
        for fp in fps:
            try:
                raw = parse_file(fp)
                if raw.shape[0] > 0:
                    frames.append(raw)
            except Exception:
                skipped += 1
        if not frames:
            continue
        try:
            df = pl.concat(frames, how='vertical_relaxed').sort('dt')
            df15 = resample_15min(df)
            n_bars = len(df15)
            if n_bars < 100:
                continue
            res = run_backtest(df15, sym)
            tested += 1
            if res['trades'] == 0:
                continue
            results.append(res)
        except Exception:
            continue
    
    print()
    print("=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print()
    
    if not results:
        print("  No results generated.")
        return
    
    total_trades = sum(r['trades'] for r in results)
    total_pnl = float(sum(r['net_pnl'] for r in results))
    passing = sum(1 for r in results if r['passes'])
    
    print(f"{'Symbol':<20} {'Trades':<8} {'Win%':<8} {'Sharpe':<10} {'PF':<10} {'MaxDD':<10} {'PnL':<12} {'Gate'}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x['sharpe']):
        print(f"{r['symbol']:<20} {r['trades']:<8} {r['win_rate']:.1%}  {r['sharpe']:<8.2f}  {r['profit_factor']:<8.2f}  {r['max_dd']:<8.1%}  INR{r['net_pnl']:<+10,.0f}  {'PASS' if r['passes'] else 'FAIL'}")
    
    print("-" * 85)
    print(f"{'AGGREGATE':<20} {total_trades:<8} {'':<8} {'':<10} {'':<10} {'':<10}  INR{total_pnl:<+10,.0f}  {passing}/{len(results)} symbols passed")
    print()
    
    print(f"EDGE GATE:")
    print(f"  Symbols tested: {tested}")
    print(f"  Results generated: {len(results)}")
    print(f"  Symbols passing: {passing}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total PnL: INR {total_pnl:+,.2f}")
    print()
    
    if passing > 0:
        print("SUCCESS: At least one symbol passed all edge gates (Sharpe>=1.0, PF>=1.2, WR>=40%, Trades>=200)")
    else:
        print("NOTE: No symbol passed all gates. Strategy may need parameter tuning or more data.")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()