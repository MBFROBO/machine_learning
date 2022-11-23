import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import graphviz

df = pd.read_csv('binary three/diabetes.csv')
print(df.head())

# По заданию обираем первые 610 строк
task_data = df.head(610)

#Выявляем, сколько пациентов из 610 больны диабетом (класс 1)

class_1 = len(task_data[task_data['Outcome'] == 1])
print("Diabetic:", class_1)

#Разделим данные 80/20 (80 - учебные, 20 - валидационные)

train_data = task_data.head(int(len(task_data) * 0.8))
test_data = task_data.tail(int(len(task_data)* 0.2))

print("Train data (length):", len(train_data))
print("Test data (length):", len(test_data))

#Выделим предикторы и отклики: x - предиктор, у - отклик

features = list(train_data.columns[:8])
x = train_data[features]
y = train_data['Outcome']

tree = DecisionTreeClassifier(criterion='entropy', #критерий разделения
                              min_samples_leaf=10,  #минимальное число объектов в листе
                              max_leaf_nodes=10,    #максимальное число листьев
                              random_state=2020)
clf=tree.fit(x, y)

columns = list(x.columns)


export_graphviz(clf, out_file='tree.dot', 
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True, label='all')

print("Depth",clf.tree_.max_depth)

# Предскажем значения отклика валидационной выборки с помощью графов

features = list(test_data.columns[:8])
x = test_data[features]
y_true = test_data['Outcome']
y_pred = clf.predict(x)

print("Accurancy: %.2f" % accuracy_score(y_true, y_pred))
print("F1: %.2f" %f1_score(y_true, y_pred, average="macro"))

features = list(df.columns[:8])
x = df[features]
y_pred = clf.predict(x)
df["Predicted"] = y_pred


print("Predicted 710", (df.iloc[710])['Predicted'])
print("Predicted 742", (df.iloc[742])['Predicted'])
print("Predicted 741", (df.iloc[741])['Predicted'])
print("Predicted 727", (df.iloc[727])['Predicted'])

