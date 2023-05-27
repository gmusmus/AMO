import os
import random
import numpy as np

def generate_data(size, anomalies=False):
    x = np.arange(size)
    noise = np.random.normal(0, 0.5, size=size)
    if anomalies:
        anomalies_indices = random.sample(range(size), int(size * 0.1))
        noise[anomalies_indices] += np.random.normal(5, 2, size=len(anomalies_indices))
    y = 2 * x + 5 + noise
    return x, y

def save_data(x, y, folder):
    os.makedirs(folder, exist_ok=True)
    np.savetxt(os.path.join(folder, 'x.txt'), x)
    np.savetxt(os.path.join(folder, 'y.txt'), y)

# Создание тренировочных данных
train_x, train_y = generate_data(1000, anomalies=True)
save_data(train_x, train_y, 'train')

# Создание тестовых данных
test_x, test_y = generate_data(200)
save_data(test_x, test_y, 'test')