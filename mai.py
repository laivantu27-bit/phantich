import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ten_file = 'data.csv' 

try:
    df = pd.read_csv(ten_file)
    print("--- Nạp dữ liệu thành công ---")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{ten_file}'.")
    exit()


df = df.fillna(df.mean(numeric_only=True))


scaler_mm = MinMaxScaler()
scaler_z = StandardScaler()


cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
df_minmax = pd.DataFrame(scaler_mm.fit_transform(df[cols_to_scale]), columns=cols_to_scale)
df_zscore = pd.DataFrame(scaler_z.fit_transform(df[cols_to_scale]), columns=cols_to_scale)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['Price'], kde=True).set_title("Gốc (Bài 1/2/4)")
plt.subplot(1, 3, 2)
sns.histplot(df_minmax['Price'], kde=True, color='g').set_title("Min-Max")
plt.subplot(1, 3, 3)
sns.histplot(df_zscore['Price'], kde=True, color='r').set_title("Z-Score")
plt.show()


if 'Price' in df.columns and 'StockQuantity' in df.columns:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['Price'], y=df['StockQuantity']).set_title("Trước chuẩn hóa (Bài 3)")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df_zscore['Price'], y=df_zscore['StockQuantity'], color='r').set_title("Sau Z-Score")
    plt.show()


print("\n--- KẾT QUẢ THẢO LUẬN LAB 3 ---")
print("1. Bài 2: Biến Price/Huyết áp bị ảnh hưởng nhiều bởi ngoại lệ.")
print("2. Bài 3: Dữ liệu có ngoại lệ lớn (công ty cực lớn), Z-Score phù hợp hơn Min-Max.")
print("3. Bài 4: Với người chơi 'cày cuốc', Min-Max sẽ làm hẹp phân phối của đa số người chơi.")
print("4. Kết luận: Z-Score ổn định hơn khi chuẩn bị dữ liệu cho mô hình KNN/Clustering.")
