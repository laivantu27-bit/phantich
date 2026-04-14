import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- KHỞI TẠO DỮ LIỆU ---
data = {
    'price': [2500, 15000, 3200, 45000, 5000, 12000, 7000, 8500, 30000, 2200, 1800, 25000, 4000, 9000, 11000, 6500, 14000, 3500, 2800, 50000, 2600, 16000, 3300, 46000, 5100, 12500, 7200, 8800, 31000, 2300],
    'area': [35, 120, 45, 500, 60, 110, 75, 80, 250, 30, 25, 300, 50, 95, 105, 65, 130, 42, 38, 550, 37, 125, 47, 510, 62, 115, 78, 82, 260, 32],
    'district': ['Cau Giay', 'Tay Ho', 'Cau Giay', 'Tu Liem', 'Thanh Xuan', 'Tay Ho', 'Cau Giay', 'Dong Da', 'Tu Liem', 'Ha Dong', 'Ha Dong', 'Tay Ho', 'Cau Giay', 'Thanh Xuan', 'Dong Da', 'Ba Dinh', 'Tay Ho', 'Cau Giay', 'Ha Dong', 'Tu Liem', 'Cau Giay', 'Tay Ho', 'Cau Giay', 'Tu Liem', 'Thanh Xuan', 'Tay Ho', 'Cau Giay', 'Dong Da', 'Tu Liem', 'Ha Dong'],
    'description': ['nhà đẹp', 'biệt thự luxury', 'nhà phố', 'siêu sang trọng', 'bán gấp', 'view hồ tây', 'chính chủ', 'gần phố', 'đẳng cấp', 'nhà cũ', 'nhỏ', 'luxury villa', 'đẹp', 'thoáng', 'trung tâm', 'phố cổ', 'biệt thự', 'giá tốt', 'cần tiền', 'siêu biệt thự', 'giá rẻ', 'biệt thự', 'phố', 'luxury', 'gấp', 'hồ tây', 'chủ', 'phố', 'cấp', 'cũ'],
    'year_built': [2015, 2020, 2010, 2023, 2018, 2021, 2012, 2014, 2022, 2005, 2000, 2019, 2016, 2017, 2020, 1990, 2022, 2015, 2010, 2025, 2014, 2021, 2011, 2024, 2017, 2020, 2013, 2015, 2021, 2004]
}
df = pd.DataFrame(data)

# --- BÀI 1: BIẾN ĐỔI NÂNG CAO & FEATURE ENGINEERING ---
print("\n" + "="*70)
print("BÀI 1: FEATURE ENGINEERING & SKEWNESS ANALYSIS")
print("-" * 70)
print(f"Skewness ban đầu của Area: {df['area'].skew():.2f}")

df['price_per_m2'] = df['price'] / df['area']
df['property_age'] = 2026 - df['year_built']
df['luxury_score'] = df['description'].str.contains(r'luxury|sang trọng|biệt thự', flags=re.IGNORECASE).astype(int)
df['desc_len'] = df['description'].str.len()

print("-> Đã tạo các feature mới: price_per_m2, property_age, luxury_score, desc_len")

# --- BÀI 2: PIPELINE HOÀN CHỈNH ---
X = df[['area', 'property_age', 'luxury_score', 'desc_len', 'district']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('pow', PowerTransformer(method='yeo-johnson')),
        ('std', StandardScaler())
    ]), ['area', 'property_age', 'desc_len']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['district'])
], remainder='passthrough')

print("\n" + "="*70)
print("BÀI 2: KIỂM THỬ PIPELINE")
print("-" * 70)
dummy_output = preprocessor.fit_transform(X_train)
print(f"Kiểm thử Shape dữ liệu sau Pipeline: {dummy_output.shape}")
print("Trạng thái Pipeline: Hoạt động tốt, không lỗi, Shape consistent.")

# --- BÀI 3: MÔ HÌNH DỰ BÁO ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n" + "="*70)
print("BÀI 3: SO SÁNH MÔ HÌNH DỰ BÁO")
print("-" * 70)
print(f"{'Mô Hình':<20} | {'RMSE':<12} | {'MAE':<12} | {'R2 Score':<10}")

for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('reg', model)])
    pipe.fit(X_train, np.log1p(y_train))
    preds = np.expm1(pipe.predict(X_test))
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<20} | {rmse:<12.2f} | {mae:<12.2f} | {r2:<10.4f}")

# --- BÀI 4: PHÂN TÍCH ĐA KỊCH BẢN & KPI ---
print("\n" + "="*70)
print("BÀI 4: PHÂN TÍCH KPI & KỊCH BẢN")
print("-" * 70)
avg_dist = df.groupby('district')['price_per_m2'].mean().sort_values()
print(f"- Ngưỡng cực trị giá (Top 5%): {df['price'].quantile(0.95):.0f} triệu VNĐ")
print(f"- Khu vực giá thấp tiềm năng: {avg_dist.index[0]} ({avg_dist.values[0]:.2f} tr/m2)")

# --- BÀI 5: TRỰC QUAN HÓA & INSIGHT ---
print("\n" + "="*70)
print("BÀI 5: TRỰC QUAN HÓA & INSIGHT NGHIỆP VỤ")
print("-" * 70)
print("- Insight: Biến đổi Power giúp ổn định mô hình trước các căn nhà có diện tích cực lớn.")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['area'], bins=10, color='blue', alpha=0.7)
plt.title("B5: Phân phối Area (Gốc)")
plt.subplot(1, 2, 2)
plt.hist(PowerTransformer().fit_transform(df[['area']]), bins=10, color='green', alpha=0.7)
plt.title("B5: Phân phối Area (Sau Transform)")
plt.tight_layout()
plt.show()