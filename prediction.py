import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('heart.csv')

scaler = StandardScaler()
columns_to_scale = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_train, y_train))

joblib.dump(model, "heart_disease_prediction.pkl")
joblib.dump(scaler, "standard_scaler.pkl")