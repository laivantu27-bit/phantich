import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

# --- BÀI 1: XÂY DỰNG PIPELINE TỔNG QUÁT ---
data = {
    'price': [5000, 12000, 5100, 45000, 3500, 1000000, 7500, 8200, 9000, 11000],
    'area': [50, 70, 50, 200, 40, 60, 55, 65, 80, 45],
    'type': ['Phố', 'Chung cư', 'Phố', 'Biệt thự', 'Cấp 4', 'Phố', 'Chung cư', 'Phố', 'Biệt thự', 'Cấp 4'],
    'description': ['nhà đẹp', 'căn hộ cao cấp', 'giá rẻ', 'sân vườn', 'hẻm', 'dinh thự', 'view biển', 'gần chợ', 'mặt tiền', 'chính chủ'],
    'date_posted': ['2026-01-01', '2026-01-15', '2026-02-01', '2026-02-10', '2026-03-01', '2026-03-15', '2026-04-01', '2026-04-10', '2026-04-20', '2026-04-25']
}
df = pd.DataFrame(data)
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['month'] = df['date_posted'].dt.month

num_features = ['area', 'month']
cat_features = ['type']
text_feature = 'description'

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features),
    ('text', TfidfVectorizer(max_features=10), text_feature)
])

processed_data = preprocessor.fit_transform(df)
cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features).tolist()
text_names = preprocessor.named_transformers_['text'].get_feature_names_out().tolist()
final_schema = num_features + cat_names + text_names

print("--- BÀI 1: SCHEMA CUỐI CÙNG ---")
print(final_schema)
print(f"Số lượng cột: {len(final_schema)}")

# --- BÀI 2: KIỂM THỬ PIPELINE ---
test_cases = {
    "Full Data": pd.DataFrame({'area': [55], 'month': [4], 'type': ['Phố'], 'description': ['mặt tiền']}),
    "Missing Data": pd.DataFrame({'area': [np.nan], 'month': [5], 'type': [None], 'description': [np.nan]}),
    "Skewed Data": pd.DataFrame({'area': [999999], 'month': [12], 'type': ['Phố'], 'description': ['siêu biệt thự']}),
    "Unseen Category": pd.DataFrame({'area': [60], 'month': [6], 'type': ['Penthouse'], 'description': ['view mây']}),
    "Wrong Format": pd.DataFrame({'area': ['80'], 'month': [7], 'type': ['Phố'], 'description': ['nhà mới']})
}

print("\n--- BÀI 2: KẾT QUẢ KIỂM THỬ ---")
for name, case in test_cases.items():
    try:
        if name == "Wrong Format": case['area'] = pd.to_numeric(case['area'])
        out = preprocessor.transform(case)
        print(f"{name:<20} | Success | Shape: {out.shape}")
    except Exception as e:
        print(f"{name:<20} | FAILED  | Error: {str(e)[:30]}")

# --- BÀI 3: TÍCH HỢP MÔ HÌNH & CROSS-VALIDATION ---
if not os.path.exists('train.csv'):
    np.random.seed(42)
    df_cv = pd.DataFrame({
        'SalePrice': np.random.exponential(200000, 500),
        'LotArea': np.random.exponential(10000, 500),
        'month': np.random.randint(1, 13, 500),
        'type': np.random.choice(['Phố', 'Chung cư', 'Biệt thự'], 500),
        'description': np.random.choice(['nhà đẹp', 'giá rẻ', 'hẻm', 'mặt tiền'], 500)
    })
else:
    df_cv = pd.read_csv('train.csv')

X_cv = df_cv.drop('SalePrice', axis=1)
y_cv = df_cv['SalePrice']

# Cập nhật preprocessor cho bài 3 (tên cột khác bài 1)
pre_cv = ColumnTransformer(transformers=[
    ('num', num_transformer, ['LotArea', 'month']),
    ('cat', cat_transformer, ['type']),
    ('text', TfidfVectorizer(max_features=5), 'description')
])

models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)}
results = []

print("\n--- BÀI 3: SO SÁNH MÔ HÌNH ---")
for name, model in models.items():
    pipe = Pipeline([('pre', pre_cv), ('model', model)])
    cv = cross_validate(pipe, X_cv, y_cv, cv=5, scoring=['neg_root_mean_squared_error', 'r2'])
    results.append({'Mô hình': name, 'RMSE': -cv['test_neg_root_mean_squared_error'].mean(), 'R2': cv['test_r2'].mean()})
print(pd.DataFrame(results))

# --- BÀI 4: TRIỂN KHAI SẢN PHẨM (INFERENCE) ---
pipeline_prod = Pipeline([('prep', pre_cv), ('model', RandomForestRegressor())])
pipeline_prod.fit(X_cv, y_cv)
joblib.dump(pipeline_prod, 'house_price_model.pkl')

def predict_price(csv_path):
    new_data = pd.read_csv(csv_path)
    model = joblib.load('house_price_model.pkl')
    new_data['Predicted_Price'] = model.predict(new_data)
    return new_data

test_df = pd.DataFrame({
    'LotArea': [6000, 20000], 'month': [12, 1], 'type': ['Phố', 'Penthouse'], 'description': ['nhà mặt tiền', 'căn hộ view biển']
})
test_df.to_csv('new_data_test.csv', index=False)

print("\n--- BÀI 4: DỰ BÁO DỮ LIỆU MỚI ---")
print(predict_price('new_data_test.csv')[['type', 'Predicted_Price']])

# Vẽ biểu đồ Before/After
skewed_sample = pd.DataFrame({'area': np.random.exponential(100, 1000), 'month': [1]*1000, 'type': ['Phố']*1000, 'description': ['n']*1000})
transformed_sample = preprocessor.transform(skewed_sample)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); sns.histplot(skewed_sample['area'], kde=True).set_title("Trước Pipeline")
plt.subplot(1, 2, 2); sns.histplot(transformed_sample[:, 0], kde=True, color='green').set_title("Sau Pipeline")
plt.tight_layout(); plt.show()