import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('data.csv')  

print("Missing values:")
print(df.isnull().sum())

print("\nThống kê mô tả:")
print(df.describe())



df.fillna(df.mean(numeric_only=True), inplace=True)


for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram - {col}')

    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')

    plt.show()

minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df.select_dtypes(include=['int64','float64'])),
                         columns=df.select_dtypes(include=['int64','float64']).columns)

# Z-score
zscore = StandardScaler()
df_zscore = pd.DataFrame(zscore.fit_transform(df.select_dtypes(include=['int64','float64'])),
                         columns=df.select_dtypes(include=['int64','float64']).columns)


for col in df_minmax.columns:
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Gốc - {col}')

    plt.subplot(1,3,2)
    sns.histplot(df_minmax[col], kde=True)
    plt.title(f'MinMax - {col}')

    plt.subplot(1,3,3)
    sns.histplot(df_zscore[col], kde=True)
    plt.title(f'Z-score - {col}')

    plt.show()