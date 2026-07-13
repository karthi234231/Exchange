import json

src = r'c:\Exchange\results\full_backtest_results.json'

txt = open(src, encoding='utf-8', errors='ignore').read()

# Extract individual result objects by balanced braces starting from "symbol"
recs = []
i = 0
n = len(txt)
while True:
    idx = txt.find('"symbol"', i)
    if idx == -1:
        break
    start = txt.rfind('{', 0, idx)
    if start == -1:
        break
    depth = 0
    j = start
    end = -1
    while j < n:
        c = txt[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = j
                break
        j += 1
    if end == -1:
        break
    chunk = txt[start:end + 1]
    try:
        rec = json.loads(chunk)
        if 'symbol' in rec and 'trades' in rec:
            recs.append(rec)
    except Exception:
        pass
    i = end + 1

if not recs:
    print("No readable result objects found yet (file may be mid-write).")
else:
    r = recs
    tt = sum(x.get('trades', 0) for x in r)
    pnl = sum(x.get('net_pnl', 0) for x in r)
    pa = sum(1 for x in r if x.get('passes'))
    print("Interim results parsed:", len(r))
    print("Total trades:", tt)
    print("Symbols passing:", pa)
    print("Total PnL: INR {:+,.2f}".format(pnl))
    print()
    print("Top 15 by Sharpe:")
    for x in sorted(r, key=lambda y: -y.get('sharpe', 0))[:15]:
        print("{:<20}{:>8}{:>8.1%}{:>10.2f}{:>10.2f}{:>10.1%}{:>14,.0f}{:>7}".format(
            x['symbol'], x['trades'], x['win_rate'], x['sharpe'], x['profit_factor'],
            x['max_dd'], x['net_pnl'], 'PASS' if x['passes'] else 'FAIL'))
    print()
    print("Worst 5 by Sharpe:")
    for x in sorted(r, key=lambda y: y.get('sharpe', 0))[:5]:
        print("{:<20}{:>8}{:>8.1%}{:>10.2f}{:>10.2f}{:>10.1%}{:>14,.0f}{:>7}".format(
            x['symbol'], x['trades'], x['win_rate'], x['sharpe'], x['profit_factor'],
            x['max_dd'], x['net_pnl'], 'PASS' if x['passes'] else 'FAIL'))
