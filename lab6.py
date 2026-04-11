import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 0. TAO DU LIEU GIA LAP DE CHAY CODE
# Buoc nay giúp ban co du lieu de thuc hanh ngay
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'Ngay': dates,
    'DoanhThu': np.random.randint(200, 1000, size=100),
    'ChiPhi': np.random.randint(100, 600, size=100),
    'DanhMuc': np.random.choice(['Dien tu', 'Gia dung', 'Thuc pham', 'Thoi trang'], size=100),
    'LoiNhuan': np.random.randint(-50, 200, size=100)
})

# BAI 1: TRUC QUAN HOA XU HUONG (2D)
# Ve bieu do duong (Line plot) so sanh Doanh thu va Chi phi theo thoi gian
print("Dang thuc hien Bai 1...")
plt.figure(figsize=(12, 6))
plt.plot(df['Ngay'], df['DoanhThu'], label='Doanh Thu', color='blue', linewidth=2)
plt.plot(df['Ngay'], df['ChiPhi'], label='Chi Phi', color='red', linestyle='--')
plt.title('Xu huong Doanh thu va Chi phi theo thoi gian')
plt.xlabel('Ngay')
plt.ylabel('Gia tri')
plt.legend()
plt.grid(True)
plt.show()

# BAI 2: TRUC QUAN HOA PHAN PHOI (2D)
# Su dung Histogram va Boxplot de kiem tra phan phoi cua Loi nhuan
print("Dang thuc hien Bai 2...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
sns.histplot(df['LoiNhuan'], kde=True, ax=axes[0], color='green')
axes[0].set_title('Phan phoi Loi nhuan (Histogram)')

# Boxplot
sns.boxplot(x='DanhMuc', y='LoiNhuan', data=df, ax=axes[1])
axes[1].set_title('So sanh Loi nhuan theo Danh muc (Boxplot)')
plt.show()

# BAI 3: TRUC QUAN HOA MOI QUAN HE (2D)
# Ve bieu do tan xa (Scatter plot) giua Doanh thu va Loi nhuan
print("Dang thuc hien Bai 3...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='DoanhThu', y='LoiNhuan', hue='DanhMuc', size='ChiPhi', data=df)
plt.title('Moi quan he giua Doanh thu va Loi nhuan')
plt.show()

# BAI 4: TRUC QUAN HOA TY TRONG (2D)
# Ve bieu do tron (Pie chart) ty le doanh thu theo danh muc
print("Dang thuc hien Bai 4...")
category_revenue = df.groupby('DanhMuc')['DoanhThu'].sum()
plt.figure(figsize=(8, 8))
plt.pie(category_revenue, labels=category_revenue.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Ty le Doanh thu theo Danh muc')
plt.show()

# BAI 5: GIANG VIEN MO RONG - HEATMAP (2D)
# Ve bieu do nhiet (Heatmap) the hien tuong quan giua cac bien so
print("Dang thuc hien Bai 5...")
plt.figure(figsize=(8, 6))
corr = df[['DoanhThu', 'ChiPhi', 'LoiNhuan']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma tran tuong quan giua cac bien')
plt.show()

print("Hoan thanh Lab 6!")