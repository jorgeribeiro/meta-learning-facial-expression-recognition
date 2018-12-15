import plot_model

from keras.preprocessing.image import ImageDataGenerator

from dataset_loader import DatasetLoader
from network_builder import NetworkBuilder

from utils import load_json_result, print_json
from constants import *

# This lines should be added on Windows (fuck Windows)
# No need to remove on Linux, works anyway
# The path to add is the graphviz's path on the machine
import os
os.environ['PATH'] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

# Load dataset
# Comment/uncomment to select dataset to use
dataset = DatasetLoader()

# JAFFE
# dataset.jaffe_load_from_save()
# num_classes = len(JAFFE_EMOTIONS)

# CK Extended
# dataset.ck_extended_load_from_save()
# num_classes = len(CK_EXTENDED_EMOTIONS)

# FERPlus (select one)
# dataset.fer_2013_small_load_from_save() # SMALL
# dataset.fer_2013_medium_load_from_save() # MEDIUM
dataset.fer_2013_load_from_save() # FULL
num_classes = len(FER_2013_EMOTIONS)

x_train, x_test = dataset.images, dataset.images_test
y_train, y_test = dataset.labels, dataset.labels_test

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# Load hyperparameters
result_name = 'best_space_000005.json'
space = load_json_result(result_name)
print ('[+] Hyperparameters:')
print_json(space)

# Build model
# Comment/uncomment to select model to use
network_builder = NetworkBuilder()
# model = network_builder.build_simplenet_opt(dataset.images.shape[1:], num_classes, space)
model = network_builder.build_vgg_16_opt(dataset.images.shape[1:], num_classes, space)

# Compile
# model.compile(loss='categorical_crossentropy', optimizer=space['optimizer'], metrics=['accuracy']) # Compile for SimpleNet
model.compile(loss='categorical_crossentropy', optimizer=space['optimizer'], metrics=['accuracy']) # Compile for VGG-16  

# Plot model for visualization
plot_model.plot(model, filename='result_vgg_16_4')

# Train
DATA_AUG = space['data_aug']
epochs=200
batch_size=int(space['batch_size'])
if not DATA_AUG:
	print ('[+] Training network without data augmentation')
	history = model.fit(
		x_train, y_train, 
		validation_data=(x_test, y_test),
		epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True).history
else:
	print ('[+] Training network with data augmentation')
	history = model.fit_generator(
		datagen.flow(x_train, y_train, batch_size=batch_size), 
		validation_data=(x_test, y_test),
		steps_per_epoch=len(x_train) // batch_size, epochs=epochs,
		verbose=1, shuffle=True).history
print ('[+] Training complete')

# Save model
if DATA_AUG:
	model.save(MODELS_PATH + 'model_fer_2013_full_vgg_16_data_aug_opt.h5')
else:
	model.save(MODELS_PATH + 'model_fer_2013_full_vgg_16_opt.h5')
print ('[+] Model saved')