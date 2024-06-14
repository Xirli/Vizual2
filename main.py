import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Створення синтетичних даних
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Випадкові дані для X в діапазоні від 0 до 10
y = 4.222151077447231 + 2.968467510701019 * X + np.random.randn(100, 1) * 2  # Лінійна залежність з деяким шумом

# Створення моделі лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Отримання параметрів моделі
intercept = model.intercept_[0]
slope = model.coef_[0][0]
print(f'Зсув (intercept): {intercept}')
print(f'Коефіцієнт нахилу (slope): {slope}')

# Прогноз для наступного періоду (наприклад, для X = 10)
X_new = np.array([[10]])
y_pred = model.predict(X_new)
print(f'Прогноз для X = 10: {y_pred[0][0]}')

# Обчислення середньої квадратичної помилки та R^2 оцінки
y_train_pred = model.predict(X)
mse = mean_squared_error(y, y_train_pred)
r2 = r2_score(y, y_train_pred)
correlation = np.corrcoef(X.T, y.T)[0, 1]

print(f'Середня квадратична помилка: {mse}')
print(f'R^2 Оцінка: {r2}')
print(f'Коефіцієнт кореляції: {correlation}')

# Візуалізація результатів
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, model.predict(X), color='red', label='Лінійна регресія')
plt.scatter(X_new, y_pred, color='green', label='Прогноз для X = 10')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.legend()
plt.show()

if r2 > 0.75:
    print("Модель має високий рівень адекватності.")
else:
    print("Модель має низький рівень адекватності.")
