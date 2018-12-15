import numpy as np
import cv2

from dataset_loader import DatasetLoader
from network_builder import NetworkBuilder

from constants import *

# Load dataset
# Comment/uncomment to select dataset to use
dataset = DatasetLoader()

# JAFFE
dataset.jaffe_load_from_save()
num_classes = len(JAFFE_EMOTIONS)

# CK Extended
# dataset.ck_extended_load_from_save()
# num_classes = len(CK_EXTENDED_EMOTIONS)

# FERPlus
# dataset.fer_2013_load_from_save()
# num_classes = len(FER_2013_EMOTIONS)

x_train, x_test = dataset.images, dataset.images_test
y_train, y_test = dataset.labels, dataset.labels_test

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

exit()

# Build model
network_builder = NetworkBuilder()
model = network_builder.build_simplenet(dataset.images.shape[1:], num_classes)

# Train
print ('[+] Training network')
with tf.device('/cpu:0'):
	model.compile(loss='categorical_crossentropy', 
				  optimizer='adadelta', 
				  metrics=['accuracy'])

model = multi_gpu_model(model, gpus=2)
model.fit(
	x_train, y_train, 
	validation_data=(x_test, y_test),
	epochs=100, batch_size=50, verbose=1, shuffle=True)

# Save model
print ('[+] Training complete')
model.save(MODELS_PATH + 'model_test_jaffe_simplenet_3.h5')
print ('[+] Model saved')