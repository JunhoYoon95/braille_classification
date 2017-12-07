import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

def get_braille(get_size=None):
    braille_data = []
    braille_data_index = []


    if get_size == True:
        return 520

    file_index = 191
    # index = 64
    index = 0
    for j in range(1, 27):
        one_hot_map = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0.]

        for k in range(1, 21):
            file_path = "Braille/character{0}/{1}_{2}.png".format(str(j).zfill(2),
                                                                  str(file_index + j).zfill(4),
                                                                  str(k).zfill(2))
            image = cv2.imread(file_path, 0)
            image_norm = np.full((28, 28), 255)
            image = np.divide(image, image_norm)
            braille_data.append(image)
            # one_hot_map = list(pad([0], 26, 0))
            one_hot_map[index] = 1
            braille_data_index.append(one_hot_map)

        index += 1
    # print(len(braille_data) == len(braille_data_index))
    # print(np.array(braille_data).shape)
    # print(np.array(braille_data_index).shape)
    return np.array(braille_data), np.array(braille_data_index)

def get_test_braille(get_image=False, get_label=False):
    _x, _y = get_braille()
    if get_image ^ get_label:
        if get_image == True:
            return _x[random.randint(0, 520)]
        elif get_label == True:
            return _y[random.randint(0, 520)]
    else:
        raise InputError("Only one parameter should be true")

    return _x[random.randint(0, 520)], _y[random.randint(0, 520)]


_x, _y = get_braille()
plt.imshow(_x[110])
plt.show()
print(_x[110])

