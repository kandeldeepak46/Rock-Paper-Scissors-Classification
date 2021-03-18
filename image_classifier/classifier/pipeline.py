import os
from pathlib import Path
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

TRAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    "D:\Computer Vision\Rock-Paper-Scissors-Classification\data\data\\test",
)

TEST_PATH = os.path.join(
    os.path.dirname(__file__),
    "D:\Computer Vision\Rock-Paper-Scissors-Classification\data\data\\train",
)

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32


class MyImageDataGenerator(object):
    def __init__(self, batch_size=BATCH_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT)):

        self.batch_size = batch_size
        self.input_shape = input_shape
        super(MyImageDataGenerator, self).__init__()

    def image_pipeline(self):

        data_gen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            zca_epsilon=1e-3,
            rotation_range=15,
            shear_range=0.1,
            zoom_range=0.2,
            fill_mode="nearest",
            horizontal_flip=True,
            vertical_flip=False,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )

        train_generator = data_gen.flow_from_directory(
            TRAIN_PATH,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        validation_generator = data_gen.flow_from_directory(
            TEST_PATH,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode="categorical",
        )
        train_samples = train_generator.samples
        validation_samples = validation_generator.samples
        batch_size = train_generator.batch_size

        return (
            train_generator,
            validation_generator,
            train_samples,
            validation_samples,
            batch_size,
        )


def main():
    images = MyImageDataGenerator()
    images.image_pipeline()


if __name__ == "__main__":
    main()
