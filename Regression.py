import pandas as pd
import quandl



df = quandl.get("FINRA/FNYX_GOOGL", authtoken="RjdpHYQufwxdKd36Z5NW")
df = df[['ShortVolume','ShortExemptVolume','TotalVolume']]
df['HL_PCT'] =  (df['TotalVolume']  - df['ShortExemptVolume'])
print(df)
