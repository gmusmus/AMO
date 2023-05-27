from sklearn.metrics import mean_squared_error
import numpy as np

# �������� ���������������� �������� ������
test_x = np.loadtxt('test_processed_x.txt')
test_y = np.loadtxt('test_processed_y.txt')
# �������� ��������� ������
# ��������������, ��� ������ ���� ��������� � ����� 'trained_model.joblib'
import joblib
model = joblib.load('trained_model.joblib')

# ��������������� �� �������� ������
predictions = model.predict(test_x)

# ���������� ������������������ ������ (MSE)
mse = mean_squared_error(test_y, predictions)
print('Mean Squared Error:', mse)