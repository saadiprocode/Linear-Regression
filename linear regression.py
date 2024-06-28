import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

tips = sns.load_dataset('tips')
print(tips.head())
print(tips.isnull().sum())
tips.dropna(inplace=True)

Q1 = tips['total_bill'].quantile(0.25)
Q3 = tips['total_bill'].quantile(0.75)
IQR = Q3 - Q1
outliers = tips[(tips['total_bill'] < (Q1 - 1.5 * IQR)) | (tips['total_bill'] > (Q3 + 1.5 * IQR))]
print(outliers)
tips = tips[~tips.index.isin(outliers.index)]

plt.figure(figsize=(10, 6))
plt.scatter(tips['total_bill'], tips['tip'], alpha=0.5)
plt.title('Total Bill vs Tip: Relationship Analysis')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()

X = tips[['total_bill']]
y = tips['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Tips')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.legend()
plt.show()

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Poly MSE: {mse}')
print(f'Poly R^2: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Tips (Poly Features)')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.legend()
plt.show()
