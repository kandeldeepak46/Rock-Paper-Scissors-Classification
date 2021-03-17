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


MODEL_SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "D:/computer_vision/rock_paper_scissors_image_classification/models/",
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


class MyPretrainedModel(object):

    """
    Boilerplate class for building the model for training the model using Pre-trained models of Deep Neural Network.

    InceptionV3 and ResnNet50V2 are specificallly used in this case
    ...
    Attributes
    ----------
    batch_size : int
        number of images, to pass to network in each iteration
    img_height : int
        specified height of the training image
    img_width : int
        specified width of training image
    epochs: int
        number of training iterations
    learning_rate : float
        training rate for neural network

    """

    def __init__(self, batch_size, img_height, img_width, epochs, learning_rate):

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.img_shape = (img_height, img_width, 3)
        super(MyPretrainedModel, self).__init__()

    def define_model(self, n_layers=150, BASE_MODEL="ResNet50V2"):
        if BASE_MODEL == "ResNet50V2":

            # Pre-trained model with MobileNetV2
            base_model = ResNet50V2(
                input_shape=self.img_shape, include_top=False, weights="imagenet"
            )
            head_model = base_model
            for layers in base_model.layers[:n_layers]:
                layers.trainable = False
            head_model = head_model.output
            head_model = tf.keras.layers.GlobalMaxPooling2D()(head_model)
            head_model = tf.keras.layers.Flatten(name="Flatten")(head_model)
            head_model = tf.keras.layers.Dense(1024, activation="relu")(head_model)
            head_model = tf.keras.layers.Dropout(0.2)(head_model)
            prediction_layer = tf.keras.layers.Dense(
                len(CLASS_NAMES), activation="softmax"
            )(head_model)
            model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)

        if BASE_MODEL == "InceptionV3":
            base_model = InceptionV3(
                input_shape=self.img_shape, include_top=False, weights="imagenet"
            )
            head_model = base_model
            for layers in base_model.layers[:n_layers]:
                layers.trainable = False

            head_model = head_model.output
            head_model = tf.keras.layers.GlobalMaxPooling2D()(head_model)
            head_model = tf.keras.layers.Flatten(name="Flatten")(head_model)
            head_model = tf.keras.layers.Dense(1024, activation="relu")(head_model)
            head_model = tf.keras.layers.Dropout(0.5)(head_model)
            prediction_layer = tf.keras.layers.Dense(
                len(CLASS_NAMES), activation="softmax"
            )(head_model)
            model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)
        return model

    def train_model(self):

        model = self.define_model(BASE_MODEL="InceptionV3")
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=0.01),
            metrics=["accuracy"],
        )
        history = model.fit(
            train_generator, epochs=5, verbose=1, validation_data=validation_generator
        )
        model.save(MODEL_SAVE_PATH)
        # model.save(f"./saved_models/trained_model.h5")
        return history

    def show_metrics(history):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.hostory["loss"]
        val_loss = history.history["val_loss"]

        epochs = range(len(acc))

        plt.plot(epochs, acc, "r", label="Training Accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend(loc=0)
        plt.figure()
        plt.show()


def main():
    model = MyLenetArchitecture(224, 224, 3, 3)
    return_model = model.build_lenet_model()
    model.model_train(return_model, 10)


if __name__ == "__main__":
    main()
    # print("my name is kandel deepak")
