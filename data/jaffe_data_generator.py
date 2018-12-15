from constants import *

from sklearn.model_selection import train_test_split

from mtcnn.mtcnn import MTCNN

import numpy as np
import os, re, cv2

# Instantiate face detector
detector = MTCNN()

# Prepare label
def emotion_to_vec(x):
    d = np.zeros(len(JAFFE_EMOTIONS))
    d[x] = 1.0
    return d

# Convert label to number
def convert_label(label):
    if label == 'AN':
        return 0
    elif label == 'DI':
        return 1
    elif label == 'FE':
        return 2
    elif label == 'HA':
        return 3
    elif label == 'SA':
        return 4
    elif label == 'SU':
        return 5
    else:
        return 6

# Detect and return only the face in the image
def crop_face(image):    
    result = detector.detect_faces(image)
    if result:
        bbox = result[0]['box']
        for i in range(len(bbox)): # Avoid getting negative coordinates via MTCNN
            if bbox[i] < 0:
                bbox[i] = 0
        image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:
                      bbox[0] + bbox[2], :]
        return image
    else:
        return []

data_labels = []
data_images = []

for img in os.listdir(JAFFE_DATASET_PATH):
    # Labels
    input_label = re.sub('[0-9]', '', img.split('.')[1])
    input_label = emotion_to_vec(convert_label(input_label))
    # Images
    print ('[+] Current file: ', img)
    if img.split('.')[3] == 'tiff': # Avoid formatting files that are not images
        input_img = cv2.imread(JAFFE_DATASET_PATH + img, cv2.IMREAD_COLOR) # Convert to RGB to detect faces via MTCNN
        input_img = crop_face(input_img)
        if input_img != []: # If face was detected
            input_img = cv2.resize(input_img, (112, 150), cv2.INTER_LINEAR)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            data_labels.append(input_label)
            data_images.append(input_img)

x_train, x_test, y_train, y_test = train_test_split(data_images, data_labels, test_size=0.20, random_state=2, shuffle=True)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

np.save(SAVE_DATASET_JAFFE_IMAGES_FILENAME, x_train)
np.save(SAVE_DATASET_JAFFE_LABELS_FILENAME, y_train)
np.save(SAVE_DATASET_JAFFE_IMAGES_TEST_FILENAME, x_test)
np.save(SAVE_DATASET_JAFFE_LABELS_TEST_FILENAME, y_test)

print ('[+] JAFFE dataset and labels saved!')