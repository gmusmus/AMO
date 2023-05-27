from sklearn.linear_model import LinearRegression
import numpy as np

# �������� ���������������� ������������� ������
train_x = np.loadtxt('train_processed_x.txt')
train_y = np.loadtxt('train_processed_y.txt')

# �������� � �������� ������
model = LinearRegression()
model.fit(train_x, train_y)

# ���������� ��������� ������
# ��� ������� ������������ ���������� � ���� � ����������� .joblib
import joblib
joblib.dump(model, 'trained_model.joblib')