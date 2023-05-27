from sklearn.metrics import mean_squared_error
import numpy as np

# Загрузка предобработанных тестовых данных
test_x = np.loadtxt('test_processed_x.txt')
test_y = np.loadtxt('test_processed_y.txt')
# Загрузка обученной модели
# Предполагается, что модель была сохранена в файле 'trained_model.joblib'
import joblib
model = joblib.load('trained_model.joblib')

# Прогнозирование на тестовых данных
predictions = model.predict(test_x)

# Вычисление среднеквадратичной ошибки (MSE)
mse = mean_squared_error(test_y, predictions)
print('Mean Squared Error:', mse)