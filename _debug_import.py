import sys, os
sys.path.insert(0, os.path.abspath('.'))
from run_quick_backtest import parse_file

fp = r'c:\Exchange\data\nifty-banknifty-intraday-data-main\2012\JAN_OCT_2012\NIFTY\NIFTY.txt'
raw = parse_file(fp)
print('raw shape', raw.shape)
print('columns', raw.columns)
print(raw.head())