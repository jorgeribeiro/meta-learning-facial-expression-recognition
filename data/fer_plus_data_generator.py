from constants import *

from sklearn.model_selection import train_test_split

from mtcnn.mtcnn import MTCNN

import numpy as np
import os, csv, cv2

# Instantiate face detector
detector = MTCNN()

# Prepare label
def emotion_to_vec(x):
    d = np.zeros(len(FER_2013_EMOTIONS))
    d[x] = 1.0
    return d

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
            input_img = cv2.resize(input_img, (34, 44), cv2.INTER_LINEAR)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        return input_img
    return []

data_labels = []
data_images = []
test_labels = []
test_images = []

# Open csv train and and test files containing image filename and labels
# 1. Test set
print ('[+] Starting test set')
csv_file_path = FER_2013_DATASET_PATH + FER_2013_TEST_FOLDER + 'label.csv'
with open(csv_file_path) as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        emotions = list(map(float, row[2:len(row)]))
        emotion = np.argwhere(emotions == np.amax(emotions)).flatten().tolist()
        if len(emotion) == 1 and emotion[0] < len(FER_2013_EMOTIONS): # If there is a highest emotion and not-unknown or not-a-face          
            image_path = FER_2013_DATASET_PATH + FER_2013_TEST_FOLDER
            img = row[0]
            print ('[+] Current file: ', img)
            input_img = format_image(image_path, img)
            if input_img != []:                
                test_labels.append(emotion_to_vec(emotion[0]))
                test_images.append(input_img)
print ('[+] Test set completed')

# 2. Train set
print ('[+] Starting train set')
csv_file_path = FER_2013_DATASET_PATH + FER_2013_TRAIN_FOLDER + 'label.csv'
with open(csv_file_path) as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        emotions = list(map(float, row[2:len(row)]))
        emotion = np.argwhere(emotions == np.amax(emotions)).flatten().tolist()
        if len(emotion) == 1 and emotion[0] < len(FER_2013_EMOTIONS): # If there is a highest emotion and not-unknown or not-a-face          
            image_path = FER_2013_DATASET_PATH + FER_2013_TRAIN_FOLDER
            img = row[0]
            print ('[+] Current file: ', img)
            input_img = format_image(image_path, img)
            if input_img != []:
                data_labels.append(emotion_to_vec(emotion[0]))
                data_images.append(input_img)
print ('[+] Train set completed')

x_train = np.asarray(data_images)
y_train = np.asarray(data_labels)
x_test = np.asarray(test_images)
y_test = np.asarray(test_labels)

np.save(SAVE_DATASET_FER_2013_IMAGES_FILENAME, x_train)
np.save(SAVE_DATASET_FER_2013_LABELS_FILENAME, y_train)
np.save(SAVE_DATASET_FER_2013_IMAGES_TEST_FILENAME, x_test)
np.save(SAVE_DATASET_FER_2013_LABELS_TEST_FILENAME, y_test)     

print ('[+] FERPlus dataset and labels saved!')