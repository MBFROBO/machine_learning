import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2

candy_data_frame = pd.read_csv('linear_regression\candy-data.csv', delimiter=',', index_col= 'competitorname')

# определим тренировочный набор данных
candy_data_train = candy_data_frame.drop(['Chiclets','Fruit Chews'],axis=0)
print(candy_data_train)

X = pd.DataFrame(candy_data_train.drop(['winpercent', 'Y'], axis=1))
y = pd.DataFrame(candy_data_train['winpercent'])

# оубчаем и предсказываем
reg = LinearRegression().fit(X,y)
pred = reg.predict([[0,0,1,0,1,0,1,0,1,0.607,0.254]])
# предсказываем значение конфеты Chiclets
Chiclets_pred_data = candy_data_frame.loc['Chiclets',:].to_frame().T
pred_Chiclets = reg.predict(Chiclets_pred_data.drop(['winpercent', 'Y'], axis = 1))
# предсказываем значение конфеты Fruit Cows
Fruit_pred_data = candy_data_frame.loc['Fruit Chews',:].to_frame().T
pred_Fruit = reg.predict(Fruit_pred_data.drop(['winpercent', 'Y'], axis = 1))

print('Предсказанное значение по данным: %.3f' % pred[0])
print('Предсказанно значение по Chiclets: %.3f' % pred_Chiclets)
print('Предсказанно значение по Fruit Chews: %.3f' % pred_Fruit)