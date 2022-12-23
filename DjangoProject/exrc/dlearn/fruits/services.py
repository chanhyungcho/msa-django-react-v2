import os

import numpy as np
import tensorflow as tf
from keras.saving.save import load_model
from exrc.dlearn.fruits.model import FruitsModel


class FruitsService():
    def __init__(self):
        global CNNClassifier
        CNNClassifier = r'C:\Users\AIA\MsaProject\DjangoProject\exrc\dlearn\fruits\CNNClassifier.h5'

    def find_fruits(self, id):
        # id = -1 리액트에서 입력한 값
        fm = FruitsModel()
        train_ds = fm.create_train_ds()
        class_names = train_ds.class_names
        train_ds = fm.change_to_prefetch_with_shuffle(train_ds)
        val_ds = fm.create_validation_ds()
        val_ds = fm.change_to_prefetch(val_ds)
        if not os.path.isfile(CNNClassifier):
            model = fm.create_model(train_ds=train_ds,val_ds=val_ds)
        else:
            model = load_model(CNNClassifier)
        test_ds = fm.create_test_ds()
        test_ds = fm.change_to_prefetch(test_ds)
        model.load_weights(CNNClassifier)
        test_loss, test_acc = model.evaluate(test_ds)
        print("test loss: ", test_loss)
        print("test accuracy: ", test_acc)
        predictions = model.predict(test_ds)
        score = tf.nn.softmax(predictions[id])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

if __name__ == '__main__':
    FruitsService().find_fruits()