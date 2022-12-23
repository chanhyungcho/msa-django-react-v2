import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

class FruitsService:
    def __init__(self):
        global class_names, trainpath, testpath, Apple_Braeburn_Test, Apple_Crimson_Snow_Test, Apple_Golden_1_Test, \
            Apple_Golden_2_Test, Apple_Golden_3_Test,Apple_Braeburn_Train,Apple_Crimson_Snow_Train,Apple_Golden_1_Train,\
            Apple_Golden_2_Train,Apple_Golden_3_Train,batch_size,img_height,img_width

        class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3']
        trainpath =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test'
        testpath =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training'
        Apple_Braeburn_Test =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test\Apple Braeburn'
        Apple_Crimson_Snow_Test =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test\Apple Crimson Snow'
        Apple_Golden_1_Test =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test\Apple Golden 1'
        Apple_Golden_2_Test =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test\Apple Golden 2'
        Apple_Golden_3_Test =r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Test\Apple Golden 3'
        Apple_Braeburn_Train = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training\Apple Crimson Snow'
        Apple_Crimson_Snow_Train = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training\Apple Braeburn'
        Apple_Golden_1_Train = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training\Apple Golden 1'
        Apple_Golden_2_Train = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training\Apple Golden 2'
        Apple_Golden_3_Train = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\fruits-360-5\Training\Apple Golden 3'
        batch_size =32
        img_height=100
        img_width=100

    def hook(self):
        #self.show_apple()
        # self.load_train_data()
        # self.divide_train_data()
        # self.load_test_data()
        # self.make_test_data()
        # self.concatenate()
        # self.show_image_with_info()
        self.modify_ds_to_prefetch()
        self.create_model()


    def show_apple(self):
        img = tf.keras.preprocessing.image.load_img \
            (f'{Apple_Golden_3_Train}//0_100.jpg')
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def load_train_data(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        return train_ds
    def divide_train_data(self):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        return val_ds


    def load_test_data(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        return test_ds

    def make_test_data(self):
        test_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False)
        return test_ds1

    def concatenate(self):
        test_ds=self.load_test_data()
        test_ds1=self.make_test_data()
        y = np.concatenate([y for x, y in test_ds], axis=0)
        print(y)
        y = np.concatenate([y for x, y in test_ds1], axis=0)
        print(y)
        x = np.concatenate([x for x, y in test_ds1], axis=0)
        print(x[0])
        return x,y

    def show_image_with_info(self):
        x,y=self.concatenate()
        plt.figure(figsize=(3, 3))
        plt.imshow(x[0].astype("uint8"))
        plt.title(class_names[y[0]])
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(3, 3))
        plt.imshow(x[-1].astype("uint8"))
        plt.title(class_names[y[-1]])
        plt.axis("off")
        plt.show()

    def modify_ds_to_prefetch(self):
        train_ds= self.load_train_data()
        val_ds=self.divide_train_data()
        test_ds=self.load_test_data()
        BUFFER_SIZE = 10000
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds,val_ds,test_ds

    def create_model(self):
        train_ds,val_ds,test_ds = self.modify_ds_to_prefetch()
        test_ds1 = self.make_test_data()
        num_classes = 5
        model = tf.keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(.50),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(.50),
            layers.Flatten(),
            layers.Dense(500, activation='relu'),
            layers.Dropout(.50),
            layers.Dense(num_classes, activation='softmax')
        ])
        print(f'model.summary():',model.summary())

        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        checkpointer = ModelCheckpoint('CNNClassifier.h5', save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy',
                                                          restore_best_weights=True)
        epochs = 20

        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpointer, early_stopping_cb]
        )
        print(f'history.history["val_accuracy"]:',len(history.history['val_accuracy']))
        acc = history.history['accuracy']  # 모델의 학습 정확도를 변수 acc에 저장
        val_acc = history.history['val_accuracy']  # 모델의 검증 정확도를 변수 val_acc에 저장

        loss = history.history['loss']  # 모델의 학습 손실을 변수 loss에 저장
        val_loss = history.history['val_loss']  # 모델의 검증 손실을 변수 val_loss에 저장

        # epochs가 14회가 아닌 다른 결과(예:10회)로 나오면 아래 줄 14를 해당 숫자인 10로 바꿔주야 함에 유의
        epochs_range = range(1, len(loss)+1)  # epochs가 14회까지만 수행된 것을 반영

        # 학습 정확도와 검증 정확도를 그리기
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # 학습 손실와 검증 손실을 그리기
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        model.load_weights('CNNClassifier.h5')
        test_loss, test_acc = model.evaluate(test_ds)

        print("test loss: ", test_loss)
        print()
        print("test accuracy: ", test_acc)

        predictions = model.predict(test_ds1)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        score = tf.nn.softmax(predictions[-1])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )


if __name__ == '__main__':
    FruitsService().hook()
