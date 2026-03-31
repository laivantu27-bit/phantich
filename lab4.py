import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [2, 4, 5, 6, 7, 8, 8.5, 9]
}
df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

X = df[["Hours"]] 
y = df["Score"]   

model = LinearRegression()

model.fit(X, y)  

new_hours = [[6]]
predicted_score = model.predict(new_hours)
print("Predicted score cho 6 giờ:", predicted_score)

new_data = [[4], [6], [9]]
predictions = model.predict(new_data)
print(predictions)

plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy (dự đoán)')
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.show()

from sklearn.metrics import r2_score
y_pred = model.predict(X)
score = r2_score(y, y_pred)
print("\nR2 Score:", score)