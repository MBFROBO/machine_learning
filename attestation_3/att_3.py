import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


def data_red():
    df = pd.read_csv('attestation_3/att_3.csv')
    return df


def Normilze_MIP(Data_Frame):
    Data_Frame_Normed = (Data_Frame - Data_Frame.min())/(Data_Frame.max() - Data_Frame.min())
    return Data_Frame_Normed

def averages(Data_Frame, norm_data):

    average = Data_Frame['MIP'].mean()
    norm_average = norm_data['MIP'].mean()
    print('MIP Average: %.3f' % average)
    print('Normalize MIP data: %.3f' %norm_average)


def diagrams(predictor):

    fig = plt.figure()
    
    ax_1 = fig.add_subplot(2, 4, 1)
    ax_2 = fig.add_subplot(2, 4, 2)
    ax_3 = fig.add_subplot(2, 4, 3)
    ax_4 = fig.add_subplot(2, 4, 4)
    ax_5 = fig.add_subplot(2, 4, 5)
    ax_6 = fig.add_subplot(2, 4, 6)
    ax_7 = fig.add_subplot(2, 4, 7)
    ax_8 = fig.add_subplot(2, 4, 8)

    ax_1.plot(predictor['MIP'], '.')
    ax_2.plot(predictor['STDIP'], '.')
    ax_3.plot(predictor['EKIP'], '.')
    ax_4.plot(predictor['SIP'], '.')
    ax_5.plot(predictor['MC'], '.')
    ax_6.plot(predictor['STDC'], '.')
    ax_7.plot(predictor['EKC'], '.')
    ax_8.plot(predictor['SC'], '.')
    plt.show()
    
def LogicalRegressionModel(df,Normilize_data):

    predictors = Normilize_data.drop(['TARGET'], axis = 1)
    print(predictors)

    callbacks = pd.DataFrame(df['TARGET'])
    print(callbacks)
    New_Object = [0.598, 0.748, 0.809, 0.913, 0.667, 0.608, 0.473, 0.731]
    predict_model = LogisticRegression(random_state=2019, solver='lbfgs').fit(X = predictors.values, y=callbacks.values.ravel())
    prd = predict_model.predict_proba([New_Object])

    print("Prediction class of star:", prd[0])

    return predictors, callbacks, New_Object

def K_NN_method(train_data, NewObject):
    euklid_A = [(a - NewObject[0])**2 for a in train_data['MIP']]
    euklid_B = [(b - NewObject[1])**2 for b in train_data['STDIP']]
    euklid_C = [(c - NewObject[1])**2 for c in train_data['EKIP']]
    euklid_D = [(d - NewObject[1])**2 for d in train_data['SIP']]
    euklid_E = [(e - NewObject[1])**2 for e in train_data['MC']]
    euklid_F = [(f - NewObject[1])**2 for f in train_data['STDC']]
    euklid_G = [(g - NewObject[1])**2 for g in train_data['EKC']]
    euklid_H = [(h - NewObject[1])**2 for h in train_data['SC']]

    Euclidan = [np.sqrt(A+B+C+D+E+F+G+H) for A,B,C,D,E,F,G,H in zip(euklid_A,euklid_B,euklid_C,euklid_D,euklid_E,euklid_F,euklid_G,euklid_H)]

    train_data["Euclidan"] = Euclidan
    sort_Euc = train_data.sort_values(by = 'Euclidan')
    Euc_min = train_data["Euclidan"].min()
    print("Minimal Euclidan distance(tested): %.3f" %Euc_min)

    neigh_Euc = KNeighborsClassifier()
    X = train_data.drop(['TARGET','Euclidan'], axis = 1)
    print(X)
    y = train_data['TARGET']
    neigh_Euc.fit(X, y)
    k_Euc_dist = neigh_Euc.kneighbors([NewObject])
    print("Minimal Euclidan distance(LIB): %.3f" %k_Euc_dist[0].min())


def main():
    df  = data_red()
    Norm_data = Normilze_MIP(df)
    averages(df, Norm_data)
    predictors, callbacks, New_Object = LogicalRegressionModel(df,Norm_data)
    diagrams(predictors)
    K_NN_method(Norm_data,New_Object)
    


if __name__ == '__main__':
    main()