import json
import shutil
import os
import time

src = r'c:\Exchange\results\full_backtest_results.json'
tmp = r'c:\Exchange\results\_tmp_view.json'

d = None
for attempt in range(20):
    try:
        shutil.copyfile(src, tmp)
        with open(tmp, encoding='utf-8') as f:
            d = json.load(f)
        break
    except Exception:
        time.sleep(0.3)
        continue
finally:
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass

if d is None:
    print("Could not read results file after retries (still mid-write). Try again shortly.")
    raise SystemExit

r = d['results']
tt = sum(x['trades'] for x in r)
pnl = sum(x['net_pnl'] for x in r)
pa = sum(1 for x in r if x['passes'])

print("=" * 95)
print("  INTERIM FULL BACKTEST RESULTS (read-only view)")
print("=" * 95)
print(f"Symbols tested (approx): {d.get('tested')}")
print(f"Results so far: {len(r)}")
print(f"Total trades: {tt}")
print(f"Symbols passing: {pa}")
print(f"Total PnL: INR {pnl:+,.2f}")
print()

top = sorted(r, key=lambda y: -y['sharpe'])[:15]
print(f"{'Symbol':<20}{'Trades':>8}{'Win%':>9}{'Sharpe':>10}{'PF':>10}{'MaxDD':>10}{'PnL':>14}{'Gate':>7}")
print("-" * 95)
for x in top:
    print(f"{x['symbol']:<20}{x['trades']:>8}{x['win_rate']:>8.1%}{x['sharpe']:>10.2f}{x['profit_factor']:>10.2f}{x['max_dd']:>10.1%}{x['net_pnl']:>14,.0f}{'PASS' if x['passes'] else 'FAIL':>7}")

print()
print("Worst 5 by Sharpe:")
print(f"{'Symbol':<20}{'Trades':>8}{'Win%':>9}{'Sharpe':>10}{'PF':>10}{'MaxDD':>10}{'PnL':>14}{'Gate':>7}")
print("-" * 95)
for x in sorted(r, key=lambda y: y['sharpe'])[:5]:
    print(f"{x['symbol']:<20}{x['trades']:>8}{x['win_rate']:>8.1%}{x['sharpe']:>10.2f}{x['profit_factor']:>10.2f}{x['max_dd']:>10.1%}{x['net_pnl']:>14,.0f}{'PASS' if x['passes'] else 'FAIL':>7}")
print("=" * 95)

</parameter>
<task_progress>
- [x] Fix parsing bugs in run_quick_backtest.py
- [x] Execute run_quick_backtest.py
- [x] Diagnose 0-bar parsing issue
- [x] Fix datetime construction and rerun
- [x] Verify final results printed without error
- [ ] Run backtest on all available data in chunks
- [ ] Report final results
</task_progress>
</write_to_file>