import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from keras.saving.save import load_model
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

class FruitsModel:
    def __init__(self):
        global class_names, trainpath, testpath, Apple_Braeburn_Test, Apple_Crimson_Snow_Test, Apple_Golden_1_Test, \
            Apple_Golden_2_Test, Apple_Golden_3_Test,Apple_Braeburn_Train,Apple_Crimson_Snow_Train,Apple_Golden_1_Train,\
            Apple_Golden_2_Train,Apple_Golden_3_Train,batch_size,img_height,img_width, num_classes, CNNClassifier

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
        num_classes = 5  # ????????? ?????? ??????, ??? ?????? ????????? ??????
        CNNClassifier = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\CNNClassifier.h5'

    def hook(self):
        # self.show_apple()
        train_ds = self.create_train_ds()
        class_names = train_ds.class_names
        train_ds = self.change_to_prefetch_with_shuffle(train_ds)
        val_ds = self.create_validation_ds()
        val_ds = self.change_to_prefetch(val_ds)
        test_ds = self.create_test_ds()
        test_ds1 = self.create_test_ds1()
        self.merge_image_label_tester(test_ds=test_ds,
                                      test_ds1=test_ds1,
                                      class_names=class_names,
                                      index=-1)  # -1 ??? ?????? ????????? ?????????
        if not os.path.isfile(CNNClassifier):
            model = self.create_model(train_ds=train_ds, val_ds=val_ds)
        else:
            model = load_model(CNNClassifier)
        print(f"model.summary(): \n{model.summary()}")

    def show_apple(self):
        img = tf.keras.preprocessing.image.load_img \
            (f'{Apple_Golden_3_Train}\\0_100.jpg')
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def create_train_ds(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    def create_validation_ds(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    def create_test_ds(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        print(f"*** Type of test_ds is {type(test_ds)} ***")
        # Type of test_ds is tensorflow.python.data.ops.dataset_ops.BatchDataset
        return test_ds

    def create_test_ds1(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False)  # shuffle=False ???????????? test_ds1??? ????????? ??????

    def extract_label_from_ds(self, ds):
        return np.concatenate([y for x, y in ds], axis=0)

    def extract_image_info(self, ds):
        return np.concatenate([x for x, y in ds], axis=0)

    def merge_image_label_tester(self, **kwargs):
        test_ds = kwargs["test_ds"]
        test_ds1 = kwargs["test_ds1"]
        class_names = kwargs["class_names"]
        y = self.extract_label_from_ds(test_ds)
        print(f"test_ds?????? ????????? ????????? ???????????? y??? ??????\n"
              f"????????? ????????? y ????????? ?????????: {y}\n")
        y = self.extract_label_from_ds(test_ds1)
        print(f"Shuffle=False ????????? ?????? ????????? test_ds1 ????????? y??? ??????\n"
              f"????????? ????????? y ????????? ????????? ??????: {y}\n")
        x = self.extract_image_info(kwargs["test_ds1"])
        print(f"test_ds1?????? ????????? ????????? ???????????? x??? ?????? : {x[0]}\n")
        plt.figure(figsize=(3, 3))
        plt.imshow(x[0].astype("uint8"))
        plt.title(class_names[y[0]])
        plt.axis("off")
        plt.show()

    def merge_image_label(self, **kwargs):
        dataset = kwargs["dataset"]
        class_names = kwargs["class_names"]
        i = kwargs["index"]
        x = self.extract_image_info(dataset)
        y = self.extract_label_from_ds(dataset)
        plt.figure(figsize=(3, 3))
        plt.imshow(x[i].astype("uint8"))
        plt.title(class_names[y[i]])
        plt.axis("off")
        plt.show()

    def change_to_prefetch(self, dataset):
        # train_ds ??????????????? Prefetch ?????????????????? ??????
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        prefetch_dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return prefetch_dataset

    def change_to_prefetch_with_shuffle(self, dataset):
        # test_ds, val_ds ??????????????? Prefetch ?????????????????? ??????
        BUFFER_SIZE = 10000
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        prefetch_dataset = dataset.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
        type(f"Type of train_ds is {prefetch_dataset}")
        '''Type of train_ds is tensorflow.python.data.ops.dataset_ops.PrefetchDataset'''
        return prefetch_dataset

    def create_model(self, **kwargs):
        train_ds = kwargs["train_ds"]
        val_ds = kwargs["val_ds"]
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
        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        checkpointer = ModelCheckpoint(CNNClassifier, save_best_only=True)
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
        print(f"len(history.history['val_accuracy']) is {len(history.history['val_accuracy'])}")
        # ??? epochs?????? ?????? ???????????? ????????? ???????????? ?????????

        acc = history.history['accuracy']  # ????????? ?????? ???????????? ?????? acc??? ??????
        val_acc = history.history['val_accuracy']  # ????????? ?????? ???????????? ?????? val_acc??? ??????

        loss = history.history['loss']  # ????????? ?????? ????????? ?????? loss??? ??????
        val_loss = history.history['val_loss']  # ????????? ?????? ????????? ?????? val_loss??? ??????

        # epochs??? 14?????? ?????? ?????? ??????(???:10???)??? ????????? ?????? ??? 14??? ?????? ????????? 10??? ???????????? ?????? ??????
        epochs_range = range(1, len(loss) + 1)  # epochs??? 14???????????? ????????? ?????? ??????
        # len(history.history['val_accuracy']) is 11

        # ?????? ???????????? ?????? ???????????? ?????????
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # ?????? ????????? ?????? ????????? ?????????
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        return model

    if __name__ == '__main__':
        FruitsModel().hook()
