from sklearn.linear_model import LinearRegression
import numpy as np

# Загрузка предобработанных тренировочных данных
train_x = np.loadtxt('train_processed_x.txt')
train_y = np.loadtxt('train_processed_y.txt')

# Создание и обучение модели
model = LinearRegression()
model.fit(train_x, train_y)

# Сохранение обученной модели
# Для примера используется сохранение в файл с расширением .joblib
import joblib
joblib.dump(model, 'trained_model.joblib')