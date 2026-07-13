import sys, os
sys.path.insert(0, os.path.abspath('.'))
import polars as pl

fp = r'c:\Exchange\data\nifty-banknifty-intraday-data-main\2012\JAN_OCT_2012\NIFTY\NIFTY.txt'
schema = {
    'column_1': pl.Utf8, 'column_2': pl.Int64, 'column_3': pl.Utf8,
    'column_4': pl.Float64, 'column_5': pl.Float64, 'column_6': pl.Float64, 'column_7': pl.Float64,
}
df = pl.read_csv(fp, has_header=False, separator=',', schema_overrides=schema)
print('read', df.shape)

df = df.rename({
    'column_1':'symbol','column_2':'date','column_3':'time',
    'column_4':'open','column_5':'high','column_6':'low','column_7':'close'
})
print('rename', df.shape)

df = df.with_columns(
    pl.format('{} {}', pl.col('date').cast(pl.Utf8), pl.col('time')).alias('_dt_str')
)
print('after format', df.shape)
print(df.select('_dt_str').head())

df = df.with_columns(pl.col('_dt_str').str.to_datetime(format='%Y%m%d %H:%M').alias('dt'))
print('after to_datetime', df.shape)
print(df.select('dt').head())

df = df.drop('_dt_str')
df = df.filter(
    (pl.col('dt').dt.hour()*60 + pl.col('dt').dt.minute()) >= (9*60+15)
).filter(
    (pl.col('dt').dt.hour()*60 + pl.col('dt').dt.minute()) < (15*60+30)
)
print('after filter', df.shape)