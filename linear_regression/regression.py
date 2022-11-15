import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2

df = pd.read_csv('new.csv', 
                delimiter=';',
                encoding='utf-8',
                index_col='id')

print(df)

awerage_X = df['X'].mean()
awerage_Y = df['Y'].mean()

req = LinearRegression().fit(df.drop(['Y'], axis = 1), df['Y'])
pred= req.predict(df.drop(['Y'], axis = 1))
determination = r2(df.drop(['X'],axis=1),pred)

print('Awerage X =',awerage_X,'  |  ','Awerage Y =',awerage_Y,'  |  ','Teta_0 = %.2f' %req.intercept_, '  |  ','Teta_1 = %.2f' %req.coef_[0],'  |  ','Detrmination = %.2f' % determination)

plt.figure(1)
plt.plot(df['X'],df['Y'], '.')
plt.plot(df['X'],(req.intercept_ + req.coef_[0]*df['X']))
plt.plot(df['X'], pred, '.')

plt.show()


