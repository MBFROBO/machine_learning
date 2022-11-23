from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Image
from IPython import display
from matplotlib import pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.metrics import f1_score

def ext_hist(img, bins=(8, 8, 8)):
    """
        >> HistCalc is a calculated histogram function. 
        >> [img] - list of researched images
        >> [0,1,2] - BRG (blue, red, green) channels
        >> None - mask (optional)
        >> bins - size in each dimension
        >> [0, 256, 0, 256, 0, 256] - Array of the dims arrays of the histogram bin boundaries in each dimension
        >> return float32 arrays of points
    """

    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def enum():
    imagePaths = sorted(list(paths.list_images('linear_regression/train')))
    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = ext_hist(image)
        data.append(hist)
        labels.append(label)
    
    return imagePaths, labels ,data

def test():
    """
        По тесту можно понять, что в классе 0 содержатся коты (кошки),
        тогда пусть в классе 1 содержатся собаки
    """
    imagePaths,labels, data = enum()
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print(labels[0])
    filename = str(imagePaths[0])
    img = plt.imread(filename)
    plt.imshow(img)
    plt.show()
    return labels

def train(labels, data):
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), 
                                                                        labels, 
                                                                        test_size=0.25, 
                                                                        random_state=3)
    
    model = LinearSVC(random_state = 3, C = 1.05)
    model.fit(trainData, trainLabels)
    return trainData, testData, trainLabels, testLabels, model                                                                    

def predictions(model, testData, testLabels):
    predictions = model.predict(testData)
    predict = f1_score(testLabels, predictions, average='macro')
    print('F1: %.2f' % predict)
    teta = model.coef_
    intercept = model.intercept_
    return predict, teta, intercept

def classification(model):
    """
        Определим класс изображений по обученному классификатору
    """
    singleImage_1 = cv2.imread('linear_regression/test/dog.1043.jpg')
    singleImage_2 = cv2.imread('linear_regression/test/dog.1017.jpg')
    singleImage_3 = cv2.imread('linear_regression/test/cat.1006.jpg')
    singleImage_4 = cv2.imread('linear_regression/test/dog.1014.jpg')
    singleImage_5 = cv2.imread('linear_regression/test/cat.jpg')

    _Images = (singleImage_1,singleImage_2,singleImage_3,singleImage_4, singleImage_5)

    hisst_1 = ext_hist(_Images[0])
    hisst_2 = ext_hist(_Images[1])
    hisst_3 = ext_hist(_Images[2])
    hisst_4 = ext_hist(_Images[3])
    hisst_5 = ext_hist(_Images[4])

    hist_reshape_1 = hisst_1.reshape(1,-1)
    hist_reshape_2 = hisst_2.reshape(1,-1) 
    hist_reshape_3 = hisst_3.reshape(1,-1) 
    hist_reshape_4 = hisst_4.reshape(1,-1)
    hist_reshape_5 = hisst_5.reshape(1,-1)

    predict_class_1 = model.predict(hist_reshape_1)
    predict_class_2 = model.predict(hist_reshape_2)  
    predict_class_3 = model.predict(hist_reshape_3)  
    predict_class_4 = model.predict(hist_reshape_4)    
    predict_class_5 = model.predict(hist_reshape_5) 

    return predict_class_1, predict_class_2, predict_class_3, predict_class_4, predict_class_5

def main():
    imagePaths,labels, data = enum()
    labels = test()
    trainData, testData, trainLabels, testLabels, model = train(labels,data)
    predict, teta, intercept = predictions(model, testData, testLabels)
    
    pc1,pc2,pc3,pc4,pc5 = classification(model)
    print('Predictions: %.2f' %predict,'   |')
    print('Teta_131:    %.2f' %teta[0][130],'  |')
    print('Teta_292:    %.2f' %teta[0][291],'  |')
    print('Teta_38:     %.2f' %teta[0][37],'  |')
    print('Intercept:   %.2f' %intercept,'  |')
    print('--------------------------------------')
    print(f'Predict classes: dog.1043: {pc1[0]}; dog.1017: {pc2[0]}; cat.1006: {pc3[0]}; dog.1014: {pc4[0]}; TEST_CAT: {pc5[0]}')


if __name__ == '__main__':
    main()

