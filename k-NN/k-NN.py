import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# С помощью sklearn обучим модель по Евклидовому (р = 2) и Манхеттенскому методу (р =1) и поределим расстояние до случйного объекта NewObject
# После чего опредеим, к какому классу относится новый объект

train_data = pd.read_csv("Task_data.csv", delimiter=',', index_col='id')

# предикторы - Х и У 
X = pd.DataFrame(train_data.drop(['Class'], axis=1))
# откликом выбираем стоблец классов
y = pd.DataFrame(train_data['Class']).values.ravel()

neigh_Euc = KNeighborsClassifier(n_neighbors=3, p=2)
neigh_Manx = KNeighborsClassifier(n_neighbors=3, p=1)
neigh_Euc.fit(X, y)
neigh_Manx.fit(X,y)
# инициализируем новый объект
NewObject = [52, 87]
pred_Euc = neigh_Euc.predict([NewObject])
pred_Manx = neigh_Manx.predict([NewObject])

print(pred_Euc, pred_Manx)

pred_probe_Euc = neigh_Euc.predict_proba([NewObject])
pred_probe_Manx = neigh_Manx.predict_proba([NewObject])
print(pred_probe_Euc, pred_probe_Manx)

k_Euc_dist = neigh_Euc.kneighbors([NewObject])
k_Manx_dist = neigh_Manx.kneighbors([NewObject])

print(k_Euc_dist,k_Manx_dist)

# Евклидовый метод вручную

euklid_X = [(x - NewObject[0])**2 for x in train_data['X']]
euklid_Y = [(y - NewObject[1])**2 for y in train_data['Y']]
print(euklid_X)
print(euklid_Y)
Euclidan = [np.sqrt(X + Y) for X,Y in zip(euklid_X,euklid_Y)]
print(Euclidan)
train_data["Euclidan"] = Euclidan
sort_Euc = train_data.sort_values(by = 'Euclidan')
print(sort_Euc)

# Манхеттенский метод вручную

manx_x = [abs(x - NewObject[0]) for x in train_data['X']]
manx_y = [abs(y - NewObject[1]) for y in train_data['Y']]
manx = [x + y for x,y in zip(manx_x ,manx_y)]
train_data['Manhattan'] = manx
sort_Manx = train_data.sort_values(by = 'Manhattan')
print(sort_Manx)