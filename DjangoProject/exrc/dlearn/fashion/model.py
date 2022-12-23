import os

import keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
from django.db import models

import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder



class FashionModel(object):

    def __init__(self):
        global model
        model = None


    def create_model(self):

        model = Sequential([ #케라스의 모델 불러옴
            keras.layers.Flatten(input_shape=(28,28)), #픽셀사이즈
            keras.layers.Dense(128, activation='relu'), # 밀도
                keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        # plt.figure()
        # plt.imshow(train_images[10])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

        model.fit(train_images,train_labels,epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test Accuracy is {test_acc}')
        file_name = os.path.join(os.path.abspath("../../../admin/save"), "fashion_model.h5")
        print(f'저장경로 {file_name}')
        model.save(file_name)







model_menu = ["Exit", # 0
              "create_model", # 1
              "spec", #2
               ]

model_lambda = {
    "1": lambda x: x.create_model(),
    #"2": lambda x: x.spec(),

}
if __name__ == '__main__':
    model = FashionModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(model_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                model_lambda[menu](model)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")
