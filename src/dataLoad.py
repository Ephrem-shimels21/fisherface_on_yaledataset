import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import errno
import sys

BASE_DIRECTORY = "faces"


def process_data():
    image_data = []
    targets = []

    for i, folder in enumerate(os.listdir(BASE_DIRECTORY)):
        folder_path = os.path.join(BASE_DIRECTORY, folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path) as img:
                img = img.resize((150, 150))
                img_array = np.array(img)
                image_data.append(img_array)
                targets.append(i)

    image_data = np.array(image_data)
    targets = np.array(targets)
    # modified_data = image_data.reshape(165, 150 * 150)
    train_data, test_data, train_label, test_label = [], [], [], []

    for person in range(15):
        person_indices = np.where(targets == person)[0]
        train_indices = person_indices[3:11]
        test_indices = person_indices[:3]
        train_data.extend(image_data[train_indices])
        train_label.extend(targets[train_indices])
        test_data.extend(image_data[test_indices])
        test_label.extend(targets[test_indices])

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label


print(process_data())


def process_data_2(path, sz=None):
    c = 0
    x, y = [], []
    print("I am in")
    i = 0
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filname in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filname))

                    if sz is not None:
                        im = im.resize(sz)
                    x.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error({0}): {1}".format(errno, os.strerror))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1

    return x, y


x, y = process_data_2("faces", (150, 150))


a, b, c, d = process_data()

print(len(a[0][0]))
