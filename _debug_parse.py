import sys, os
sys.path.insert(0, os.path.abspath('.'))
import polars as pl

fp = r'c:\Exchange\data\nifty-banknifty-intraday-data-main\2012\JAN_OCT_2012\NIFTY\NIFTY.txt'
with open(fp,'r') as f:
    first = f.readline().strip()
print('first', first)
schema = {
    'column_1': pl.Utf8, 'column_2': pl.Int64, 'column_3': pl.Utf8,
    'column_4': pl.Float64, 'column_5': pl.Float64, 'column_6': pl.Float64, 'column_7': pl.Float64,
}
df = pl.read_csv(fp, has_header=False, separator=',', schema_overrides=schema)
print('after read', df.shape)
df = df.rename({
    'column_1':'symbol','column_2':'date','column_3':'time',
    'column_4':'open','column_5':'high','column_6':'low','column_7':'close'
})
print('after rename', df.shape)
print(df.head())