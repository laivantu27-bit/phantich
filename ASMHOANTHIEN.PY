import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Giả lập tập dữ liệu để chạy demo (Bạn có thể thay bằng pd.read_csv)
data = {
    'dien_tich': [50, 80, 120, np.nan], # Có dữ liệu thiếu 
    'so_phong': [2, 3, 4, 3],
    'quan': ['Cau Giay', 'Nam Tu Liem', 'Unknown_Dist', 'Cau Giay'], # Có category mới 
    'mo_ta': ['nha dep thoang', 'gan mat duong', 'gia re bat ngo', 'sang trong'],
    'gia_nha': [2500, 3200, 4500, 2800] # Target để train mô hình
}
df = pd.DataFrame(data)
X = df.drop('gia_nha', axis=1)
y = df['gia_nha']

print("=== GIAI ĐOẠN HOÀN THIỆN ===\n")

# --- MỤC 1 & 2: PIPELINE TỰ ĐỘNG & XỬ LÝ UNSEEN/MISSING --- [cite: 82, 83]
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Xử lý unseen categories
])

# Tự động phát hiện cột theo kiểu dữ liệu 
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, make_column_selector(dtype_include=['number'])),
    ('cat', cat_pipeline, make_column_selector(dtype_include=['object']))
])

print("1 & 2. Pipeline đã tự động phát hiện và xử lý missing/unseen categories.")

# --- MỤC 3: FEATURE INTERACTION --- 
# Thử tạo interaction giữa diện tích x số phòng
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df[['dien_tich', 'so_phong']].fillna(0))

print(f"3. Feature Interaction thành công. Số đặc trưng sau tương tác: {interaction_features.shape[1]}")

# --- MỤC 4: SO SÁNH MÔ HÌNH ĐA PHƯƠNG THỨC --- 

# Mô hình A: Chỉ dùng Numerical (Diện tích, Số phòng)
X_num = df[['dien_tich', 'so_phong']].fillna(0)
model_a = XGBRegressor().fit(X_num, y)
pred_a = model_a.predict(X_num)
rmse_a = np.sqrt(mean_squared_error(y, pred_a))

# Mô hình B: Đa phương thức (Numerical + Text từ mô tả)
# Gộp Pipeline xử lý số/hạng mục và Pipeline xử lý văn bản
full_pipeline = FeatureUnion([
    ('num_cat', preprocessor),
    ('text', TfidfVectorizer(max_features=10)) 
])

# Lưu ý: TF-IDF áp dụng cho cột 'mo_ta'
X_full = full_pipeline.fit_transform(X)
model_b = XGBRegressor().fit(X_full, y)
pred_b = model_b.predict(X_full)
rmse_b = np.sqrt(mean_squared_error(y, pred_b))

print("\n4. KẾT QUẢ SO SÁNH MÔ HÌNH:")
print(f" > Mô hình chỉ dùng số (Numerical): RMSE = {rmse_a:.2f}")
print(f" > Mô hình Đa phương thức (Numerical + Text): RMSE = {rmse_b:.2f}")

if rmse_b < rmse_a:
    print("\n=> KẾT LUẬN: Mô hình đa phương thức cho kết quả tốt hơn (RMSE thấp hơn)!")
else:
    print("\n=> KẾT LUẬN: Cần thêm dữ liệu ảnh hoặc tinh chỉnh để cải thiện mô hình.")