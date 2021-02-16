import os
import numpy as np
import json

try:
    import matplotlib.pyplot as plt
except:
    pass
import random
import pathlib
from PIL import Image
import glob
import tensorflow as tf
from collections import Counter
import matplotlib.patches as patches
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Flatten,
    Activation,
    Dense,
    Dropout,
    Layer,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.applications import vgg16, VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from pipeline import MyImageDataGenerator
from tensorflow import keras

MODEL_SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "D:/image_classification/rock_paper_scissors_image_classification/models/",
)
MODEL_SAVE_PATH = MODEL_SAVE_DIR + "rock_paper_scissors.h5"


class MyLenetArchitecture(object):
    def __init__(self, width, height, image_channel, number_of_classes):
        self.width = width
        self.height = height
        self.image_channel = image_channel
        self.number_of_classes = number_of_classes
        super(MyLenetArchitecture, self).__init__()

    def build_lenet_model(self):

        model = Sequential()

        # Convolution
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                strides=(2, 2),
                input_shape=(self.width, self.height, self.image_channel),
            )
        )

        # ReLU
        model.add(Activation("relu"))

        # Pooling
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Convolution
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))

        # ReLU
        model.add(Activation("relu"))

        # Pooling
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(100))

        model.add(Activation("relu"))

        model.add(Dense(self.number_of_classes, activation="softmax"))

        return model

    def model_train(self, model, epoch):
        self.model = model
        self.epoch = epoch
        img_pipeline = MyImageDataGenerator()
        (
            train_generator,
            validation_generator,
            train_samples,
            validation_samples,
            batch_size,
        ) = img_pipeline.image_pipeline()

        self.model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples // batch_size,
            epochs=self.epoch,
            validation_data=validation_generator,
            validation_steps=validation_samples // batch_size,
        )
        self.model.save_weights(MODEL_SAVE_PATH)


class MyKerasModels(object):
    def __init__(self, width, height, image_channel, number_of_classes):
        self.width = width
        self.height = height
        self.image_channel = image_channel
        super(MyKerasModels, self).__init__()

    def build_naive_model(self, number_of_classes: int):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(self.width, self.height, self.image_channel),
                ),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dense(6, activation=tf.nn.softmax),
            ]
        )

    def build_fcnn_from_vgg16(self, number_of_classes: int):
        base_model = vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.width, self.height, self.image_channel),
        )
        fcnn_model = Sequential()
        for layer in base_model.layers:
            fcnn_model.add(layer)
        fcnn_model.add(
            Conv2D(filters=4096, kernel_size=(7, 7), name="fc1", activation="relu")
        )
        fcnn_model.add(
            Conv2D(filters=4096, kernel_size=(1, 1), name="fc2", activation="relu")
        )
        fcnn_model.add(Conv2D(filters=1000, kernel_size=(1, 1), name="predictions"))
        fcnn_model.add(Softmax4D(axis=-1, name="softmax"))
        vgg_top = vgg16.VGG16(
            weights="imagenet",
            include_top=True,
            input_shape=(self.width, self.height, self.image_channels),
        )
        for layer in fcnn_model.layers:
            if layer.name.startswith("fc") or layer.name.startswith("pred"):
                orig_layer = vgg_top.get_layer(layer.name)
                W, b = orig_layer.get_weights()
                ax1, ax2, previous_filter, n_filter = layer.get_weights()[0].shape
                new_W = W.reshape(ax1, ax2, -1, n_filter)
                layer.set_weights([new_W, b])
        del base_model
        del vgg_top
        return fcnn_model


def main():
    model = MyLenetArchitecture(224, 224, 3, 3)
    return_model = model.build_lenet_model()
    model.model_train(return_model, 10)


if __name__ == "__main__":
    # main()
    print("my name is kandel deepak")
