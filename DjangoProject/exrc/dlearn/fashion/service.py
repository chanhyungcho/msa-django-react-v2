import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

class FashionService(object):
    def __init__(self):
        global class_names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#, i, predictions_array, true_label, img
    def service_model(self,i) -> int:
        model = load_model(r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fashion\fashion_model.h5')
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img=predictions[i], test_labels[i], test_images[i] #***
        # plt.grid(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(img, cmap=plt.cm.binary)
        result = np.argmax(predictions_array)
        print(f"예측한 답 : {result}")
       # predicted_label = np.argmax(predictions_array) #***
        # print(f'예측한 답: {predicted_label}')
        # if predicted_label == true_label:
        #     color = 'blue'
        # else:
        #     color = 'red'
        # plt.xlabel('{} {:2.0f}% ({})'.format(
        #     class_names[true_label],
        #     100 * np.max(predictions_array),
        #     class_names[true_label]
        # ), color = color)
        # plt.show()
        if result == 0:
            resp = 'T-shirt/top'
        elif result == 1:
            resp = 'Trouser'
        elif result == 2:
            resp = 'Pullover'
        elif result == 3:
            resp = 'Dress'
        elif result == 4:
            resp = 'Coat'
        elif result == 5:
            resp = 'Sandal'
        elif result == 6:
            resp = 'Shirt'
        elif result == 7:
            resp = 'Sneaker'
        elif result == 8:
            resp = 'Bag'
        elif result == 9:
            resp = 'Ankle boot'
        return resp


fashion_menu = ["Exit", #0
                "service_model"] #1
fashion_lambda = {
    "1" : lambda x: x.service_model(),
}
if __name__ == '__main__':
    service = FashionService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(fashion_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                fashion_lambda[menu](service)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")