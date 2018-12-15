import numpy as np
import time
from keras.preprocessing.image import save_img
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras import backend as K

from dataset_loader import DatasetLoader
from constants import *

# Dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32

# Load dataset
# Comment/uncomment to select dataset to use
dataset = DatasetLoader()

# CK Extended no resize
dataset.ck_extended_load_from_save()
classes = CK_EXTENDED_EMOTIONS

# Load model
model = load_model(MODELS_PATH + 'model_ck_extended_inception_v3_1.h5')
print ('[+] Model loaded')
print (model.summary())

from quiver_engine.server import launch
launch(model, classes=classes, input_folder='./imgs')