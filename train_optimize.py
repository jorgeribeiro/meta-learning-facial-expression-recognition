import search_spaces
import plot_model

from hyperopt import tpe, fmin, Trials, STATUS_OK
from keras.preprocessing.image import ImageDataGenerator

from dataset_loader import DatasetLoader
from network_builder import NetworkBuilder

from utils import save_json_result, print_json
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

# Function to minimize
# In this case, the network's hyperparameters
def optimize_network(space):
    print ('[+] Search space:')
    print_json(space)
    
    # Build model
    # Comment/uncomment to select model to use
    network_builder = NetworkBuilder()
    # model = network_builder.build_simplenet_opt(dataset.images.shape[1:], num_classes, space)
    model = network_builder.build_vgg_16_opt(dataset.images.shape[1:], num_classes, space)

    # Compile
    # Comment/uncomment to select appropriate compile for model
    # model.compile(loss='categorical_crossentropy', optimizer=space['optimizer'], metrics=['accuracy']) # Compile for SimpleNet
    model.compile(loss='categorical_crossentropy', optimizer=space['optimizer'], metrics=['accuracy']) # Compile for VGG-16  

    # Plot model for visualization
    plot_model.plot(model, filename='optimized_vgg_16')

    # Train and optimize
    epochs=100
    if not space['data_aug']:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs, batch_size=int(space['batch_size']),
            verbose=1, shuffle=True).history
    else:
        history = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=int(space['batch_size'])), 
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // int(space['batch_size']), epochs=epochs,
            verbose=1, shuffle=True).history

    result = {
        'loss': max(history['loss']),
        'status': STATUS_OK,
        'space': space
    }
    return result

print ('[+] Training network and optimizing hyperparameters')
trials = Trials()
best = fmin(
    optimize_network,
    search_spaces.space_vgg_16,
    algo=tpe.suggest,
    trials=trials,
    max_evals=10)

print ('[+] Training and optimization completed')
print ('[+] Best space:')
print_json(best)
save_json_result(best, 'best_space_000005.json')