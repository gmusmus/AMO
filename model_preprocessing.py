from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(folder):
    x = np.loadtxt(os.path.join(folder, 'x.txt'))
    y = np.loadtxt(os.path.join(folder, 'y.txt'))
    return x, y

# Загрузка тренировочных данных
train_x, train_y = load_data('train')

# Загрузка тестовых данных
test_x, test_y = load_data('test')

# Применение предобработки данных
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x.reshape(-1, 1))
test_x = scaler.transform(test_x.reshape(-1, 1))

# Сохранение предобработанных данных
np.savetxt('train_processed_x.txt', train_x)
np.savetxt('train_processed_y.txt', train_y)
np.savetxt('test_processed_x.txt', test_x)
np.savetxt('test_processed_y.txt', test_y)