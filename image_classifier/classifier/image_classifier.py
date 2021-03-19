from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

try:
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np
import sys
import os

MODLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "D:\Computer Vision\Rock-Paper-Scissors-Classification\models\\rock_paper_scissors.h5",
)

if not os.path.isfile(MODLE_PATH):
    raise FileNotFoundError("please check the input model or train the network")


def load_image(img_path, show=False):
    img = image.load_image(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0

    if show:
        plt.imshow(img_tensor[0])
        plt.axis("off")
        plt.show()

    return img_tensor


if __name__ == "__main__":
    # loading the keras h5 model
    model = load_model(MODLE_PATH)
    img_path = sys.argv[1]

    if not os.path.isfile(img_path):
        raise FileNotFoundError("please give the proper image path")
    new_image = load_image(img_path)

    pred = model.predit(new_image)

    print(pred)
