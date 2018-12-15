from constants import *

from sklearn.model_selection import train_test_split

import numpy as np
import os, csv, cv2

# Only possible to generate if the full dataset has been already generated

# Load full dataset
data_images = np.load(SAVE_DATASET_FER_2013_IMAGES_FILENAME)
data_labels = np.load(SAVE_DATASET_FER_2013_LABELS_FILENAME)
test_images = np.load(SAVE_DATASET_FER_2013_IMAGES_TEST_FILENAME)
test_labels = np.load(SAVE_DATASET_FER_2013_LABELS_TEST_FILENAME)

# Get only half of dataset
x_train, _, y_train, _ = train_test_split(data_images, data_labels, test_size=0.5, random_state=2, shuffle=True)
_, x_test, _, y_test = train_test_split(test_images, test_labels, test_size=0.5, random_state=2, shuffle=True)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

np.save(SAVE_DATASET_FER_2013_IMAGES_MEDIUM_FILENAME, x_train)
np.save(SAVE_DATASET_FER_2013_LABELS_MEDIUM_FILENAME, y_train)
np.save(SAVE_DATASET_FER_2013_IMAGES_TEST_MEDIUM_FILENAME, x_test)
np.save(SAVE_DATASET_FER_2013_LABELS_TEST_MEDIUM_FILENAME, y_test)     

print ('[+] FERPlus medium dataset and labels saved!')