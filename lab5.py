import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 0. TAO DU LIEU GIA LAP
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
df_main = pd.DataFrame({
    'Date': dates,
    'Revenue': np.random.randint(100, 500, size=100),
    'Close': np.cumsum(np.random.randn(100)) + 100,
    'Production': np.random.randint(50, 150, size=100)
})

dates_h = pd.date_range(start='2024-01-01', periods=48, freq='h')
df_web = pd.DataFrame({'Traffic': np.random.randint(10, 100, size=48)}, index=dates_h)

# BAI 1: DOANH THU SIEU THI
print("Dang xu ly Bai 1...")
df1 = df_main[['Date', 'Revenue']].copy()
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
df1['Revenue'] = df1['Revenue'].ffill()

df1['Year'] = df1.index.year
df1['Month'] = df1.index.month
df1['Quarter'] = df1.index.quarter
df1['DayOfWeek'] = df1.index.dayofweek
df1['IsWeekend'] = df1.index.dayofweek >= 5

df1.resample('ME')['Revenue'].sum().plot(title='Doanh thu theo thang')
plt.show()

# BAI 2: LUU LUONG WEBSITE
print("Dang xu ly Bai 2...")
df2 = df_web.copy()
df2 = df2.asfreq('h').interpolate(method='linear')
df2['Hour'] = df2.index.hour
df2.groupby('Hour')['Traffic'].mean().plot(kind='bar', title='Traffic theo gio')
plt.show()

# BAI 3: GIA CO PHIEU
print("Dang xu ly Bai 3...")
df3 = df_main[['Date', 'Close']].copy()
df3.set_index('Date', inplace=True)
df3 = df3.ffill()
df3['MA7'] = df3['Close'].rolling(window=7).mean()
df3['MA30'] = df3['Close'].rolling(window=30).mean()
df3[['Close', 'MA7', 'MA30']].plot(title='Trend Co Phieu')
plt.show()

# BAI 4: SAN XUAT CONG NGHIEP
print("Dang xu ly Bai 4...")
df4 = df_main[['Date', 'Production']].copy()
df4.set_index('Date', inplace=True)
df4['Week'] = df4.index.isocalendar().week
df4['Quarter'] = df4.index.quarter

result = seasonal_decompose(df4['Production'], model='additive', period=7)
result.plot()
plt.show()

print("Hoan thanh Lab 5!")