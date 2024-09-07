import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Đọc dữ liệu từ iris.csv
data = pd.read_csv('iris.csv')
print(data)
X = data.drop('variety', axis=1)
y = data['variety']

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình với Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(model, 'model_iris.pkl')

print("Model training complete and saved to model_iris.pkl")
