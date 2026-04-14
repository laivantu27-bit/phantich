import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

if not os.path.exists('train.csv'):
    np.random.seed(42)
    data = {
        'SalePrice': np.random.exponential(200000, 1000),
        'LotArea': np.random.exponential(10000, 1000),
        'GrLivArea': np.random.normal(1500, 500, 1000),
        'OverallQual': np.random.randint(1, 11, 1000),
        'GarageCars': np.random.randint(0, 5, 1000),
        'MiscVal': np.random.normal(0, 100, 1000)
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv('train.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns
skew_values = df[numeric_cols].skew().sort_values(ascending=False)
print("Top 10 Skewed Columns:\n", skew_values.head(10))

top_3_cols = skew_values.index[:3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(top_3_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Dist of {col}")
plt.show()

col_pos1, col_pos2 = 'LotArea', 'SalePrice'
col_neg = 'MiscVal'

df['log_val'] = np.log1p(df[col_pos1].clip(lower=0))
df['boxcox_val'], _ = stats.boxcox(df[col_pos2].clip(lower=0) + 1)
pt = PowerTransformer(method='yeo-johnson')
df['power_val'] = pt.fit_transform(df[[col_neg]])

X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_a = LinearRegression().fit(X_train, y_train)
pred_a = model_a.predict(X_test)

y_train_log = np.log1p(y_train.clip(lower=0))
model_b = LinearRegression().fit(X_train, y_train_log)
pred_b = np.expm1(model_b.predict(X_test))

pt_x = PowerTransformer()
X_train_pt = pt_x.fit_transform(X_train)
X_test_pt = pt_x.transform(X_test)
model_c = LinearRegression().fit(X_train_pt, y_train)
pred_c = model_c.predict(X_test_pt)

def eval_m(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

eval_m(y_test, pred_a, "Version A (Original)")
eval_m(y_test, pred_b, "Version B (Log Target)")
eval_m(y_test, pred_c, "Version C (Power Features)")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True).set_title('Raw SalePrice')
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['SalePrice'].clip(lower=0)), kde=True).set_title('Transformed SalePrice')
plt.show()