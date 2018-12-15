from constants import *

from sklearn.model_selection import train_test_split

from mtcnn.mtcnn import MTCNN

import numpy as np
import os, cv2

# Instantiate face detector
detector = MTCNN()

# Prepare label
def emotion_to_vec(x):
    d = np.zeros(len(CK_EXTENDED_EMOTIONS))
    d[x] = 1.0
    return d

# Return label according to file if it exists
def get_label(filepath):
    if os.path.exists(filepath) and os.listdir(filepath):
        g = open(filepath + str(os.listdir(filepath)[0]), 'r')
        label = g.readline().split('.')[0].replace(" ", "")
        return int(label)
    else:
        return -1

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

# Read, crop, and convert image to grayscale
def format_image(image_path, img):
    if img.split('.')[1] == 'png': # Avoid formatting files that are not images
        input_img = cv2.imread((image_path + img), cv2.IMREAD_COLOR) # Convert to RGB to detect faces via MTCNN    
        input_img = crop_face(input_img)    
        if input_img != []: # If face was detected
            input_img = cv2.resize(input_img, (112, 150), cv2.INTER_LINEAR)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        return input_img
    return []

data_labels = []
data_images = []

# Loops will run through the labels folder in order to use only sessions with an associated label
# Images with associated empty label folder will be ignored
for subject in sorted(os.listdir(CK_EXTENDED_LABELS_PATH)):
    for session in sorted(os.listdir(CK_EXTENDED_LABELS_PATH + str(subject))):
        image_path = CK_EXTENDED_DATASET_PATH + str(subject) + '/' + str(session) + '/'
        label_path = CK_EXTENDED_LABELS_PATH + str(subject) + '/' + str(session) + '/'
        label = get_label(label_path)
        if label != -1: # If there is a label
            images_count = len(os.listdir(image_path))
            range_count = int(images_count / 2 - 2)
            for i, img in zip(range(1), sorted(os.listdir(image_path))): # Neutral batch
                print ('[+] Current file: ', img)
                input_img = format_image(image_path, img)
                if input_img != []:
                    data_labels.append(emotion_to_vec(0))
                    data_images.append(input_img)
            for i, img in zip(range(range_count), reversed(sorted(os.listdir(image_path)))): # Emotion batch            
                print ('[+] Current file: ', img)
                input_img = format_image(image_path, img)
                if input_img != []:
                    data_labels.append(emotion_to_vec(label))
                    data_images.append(input_img)

x_train, x_test, y_train, y_test = train_test_split(data_images, data_labels, test_size=0.15, random_state=2, shuffle=True)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

np.save(SAVE_DATASET_CK_EXTENDED_IMAGES_FILENAME, x_train)
np.save(SAVE_DATASET_CK_EXTENDED_LABELS_FILENAME, y_train)
np.save(SAVE_DATASET_CK_EXTENDED_IMAGES_TEST_FILENAME, x_test)
np.save(SAVE_DATASET_CK_EXTENDED_LABELS_TEST_FILENAME, y_test)

print ('[+] CK+ dataset and labels saved!')