from django.db import models

import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

'''
Iris Species
Classify iris plants into three species in this classic dataset #답이 3종류고 분류기를 써야한다. = 이처럼 방향성을 잡고 시작해야.
'''



class Iris(object):
    def __init__(self):
        self.iris = datasets.load_iris()
        print(f'type {type(self.iris)}') #type <class 'sklearn.utils._bunch.Bunch'>
        self._X = self.iris.data
        self._Y = self.iris.target


    def hook(self):
        self.spec()
        #self.create_model()



    def spec(self):
        print(f'{self.iris.feature_names}')

    '''
    Shape (150, 6)
    columns ['Id', 'Sepal꽃받침LengthCm', 'SepalWidthCm', 'Petal꽃잎LengthCm', 'PetalWidthCm', 'Species']
    '''

    def create_model(self):
        X = self._X
        Y = self._Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1,1)).toarray()
        model = Sequential() #<텐서플로 도구로 만든 모델
        model.add(Dense(4, input_dim=4, activation='relu')) #확률 밀도 함수 > 분포를 따짐. 타깃과 id를 뺀 피처
        model.add(Dense(3, activation='softmax')) #타깃의 클래스 개수, 필터.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed')

        file_name = r'C:\Users\AIA\djangoProject\admin\save\iris_model.h5' #인코딩 된것
        model.save(file_name)
        print(f'Model Saved in {file_name}')









iris_menu = ["Exit",  # 0
               "Hook",  # 1
             "spec" #2
               ]
iris_lambda = {
    "1": lambda x: x.hook(),
    #"2": lambda x: x.spec(),

}
if __name__ == '__main__':
    iris = Iris()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")
