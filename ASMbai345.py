import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'description': [
        'Nhà đẹp quận 1, diện tích 50m2, giá rẻ',
        'Căn hộ chung cư cao cấp, view biển, 2PN',
        'Nhà đẹp Q1, dt 50m2, giá cực tốt',  # Giả lập duplicate
        'Biệt thự sân vườn, diện tích 200m2',
        'Nhà cấp 4, hẻm xe hơi, diện tích 40m2',
        'Dinh thự xa hoa bậc nhất thành phố'    # Outlier về giá
    ],
    'price': [5000, 12000, 5100, 45000, 3500, 1000000], 
    'area': [50, 70, 50, 200, 40, 60],
    'type': ['Phố', 'Chung cư', 'Phố', 'Biệt thự', 'Cấp 4', 'Phố']
}

df = pd.DataFrame(data)
print("--- Dữ liệu gốc ---")
print(df)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


df['price_cleaned'] = np.clip(df['price'], lower_bound, upper_bound)


df['price_zscore'] = np.abs(stats.zscore(df['price']))

scaler_mm = MinMaxScaler()
df['price_minmax'] = scaler_mm.fit_transform(df[['price_cleaned']])

scaler_std = StandardScaler()
df['area_std'] = scaler_std.fit_transform(df[['area']])

le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = cosine_similarity(tfidf_matrix)

threshold = 0.7
to_drop = set()

print("\n--- Phân tích trùng lặp văn bản ---")
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            print(f"Dòng {i} và Dòng {j} giống nhau {cosine_sim[i,j]*100:.1f}%. Gợi ý: MERGE.")
            to_drop.add(j)

df_final = df.drop(index=list(to_drop)).reset_index(drop=True)

print("\n--- Dữ liệu cuối cùng sau khi xử lý (Đã loại trùng) ---")

columns_show = ['description', 'price_cleaned', 'price_minmax', 'area_std', 'type_encoded']
print(df_final[columns_show])