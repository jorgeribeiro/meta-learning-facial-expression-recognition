import tensorflow as tf

import keras
import keras.backend as K

from keras.utils import np_utils, multi_gpu_model
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from resnet import ResnetBuilder

class NetworkBuilder:
    def __init__(self):
        pass

    def build_simplenet(self, shape, num_classes):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), input_shape=shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 2
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 3
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 4
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # First Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(Dropout(0.2))
        
        
        # Block 5
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 6
        self.model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 7
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        # Second Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        
        # Block 8
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 9
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        # Third Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        
        
        # Block 10
        self.model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Block 11  
        self.model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Block 12  
        self.model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        # Fourth Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.model.add(Dropout(0.2))

        # Block 13
        self.model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
        self.model.add(Activation('relu'))
        # Fifth Maxpooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=2))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] SimpleNet model built')
        return self.model

    def build_simplenet_opt(self, shape, num_classes, space):
        self.model = Sequential()

        # Block 1
        n_filters = int(64 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer'], input_shape=shape))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 2
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 3
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 4
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        # First Maxpooling
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))
        self.model.add(Dropout(space['dropout']))
        
        
        # Block 5
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 6
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 7
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        # Second Maxpooling
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        
        # Block 8
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 9
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        # Third Maxpooling
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))
        
        
        # Block 10
        n_filters = int(512 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        if space['use_BN']:
        	self.model.add(BatchNormalization())
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))

        # Block 11  
        n_filters = int(2048 * space['conv_filter_size_mult'])
        k = 1
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        self.model.add(Activation(space['activation']))
        self.model.add(Dropout(space['dropout']))
        
        # Block 12  
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = 1
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        self.model.add(Activation(space['activation']))
        # Fourth Maxpooling
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))
        self.model.add(Dropout(space['dropout']))

        # Block 13
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
        	kernel_size=(k,k), padding=space['conv_padding'], 
        	kernel_initializer=space['conv_kernel_initializer']))
        self.model.add(Activation(space['activation']))
        # Fifth Maxpooling
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] SimpleNet model to optimize built')
        return self.model

    def build_vgg_16(self, shape, num_classes):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=shape))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 2
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 3
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 4
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 5
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] VGG-16 model built')
        return self.model

    def build_vgg_16_opt(self, shape, num_classes, space):
        self.model = Sequential()

        # Block 1
        n_filters = int(64 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation'], input_shape=shape))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Block 2
        n_filters = int(128 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Block 3
        n_filters = int(256 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Block 4
        n_filters = int(512 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Block 5
        n_filters = int(512 * space['conv_filter_size_mult'])
        k = int(space['conv_kernel_size'])
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        self.model.add(Conv2D(filters=n_filters, 
            kernel_size=(k,k), padding=space['conv_padding'], 
            kernel_initializer=space['conv_kernel_initializer'], 
            activation=space['activation']))
        p = space['pool_size']
        s = int(space['pool_strides'])
        self.model.add(MaxPooling2D(pool_size=(p,p), strides=s, padding=space['pool_padding']))

        # Final Classifier
        self.model.add(Flatten())
        n_filters = int(4096 * space['conv_filter_size_mult'])
        self.model.add(Dense(n_filters, activation=space['activation']))
        self.model.add(Dropout(space['dropout']))
        self.model.add(Dense(n_filters, activation=space['activation']))
        self.model.add(Dropout(space['dropout']))
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] VGG-16 model to optimize built')
        return self.model

    def build_vgg_19(self, shape, num_classes):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=shape))
        self.model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 2
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 3
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 4
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 5
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Final Classifier
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))        
        
        print ('[+] VGG-19 model built')
        return self.model

    def build_resnet_18(self, shape, num_classes):
        self.model = ResnetBuilder.build_resnet_18(shape, num_classes)

        print ('[+] ResNet-18 model built')
        return self.model

    def build_resnet_34(self, shape, num_classes):
        self.model = ResnetBuilder.build_resnet_34(shape, num_classes)

        print ('[+] ResNet-34 model built')
        return self.model

    def build_resnet_50(self, shape, num_classes):
        self.model = ResnetBuilder.build_resnet_50(shape, num_classes)

        print ('[+] ResNet-50 model built')
        return self.model

    def build_resnet_101(self, shape, num_classes):
        self.model = ResnetBuilder.build_resnet_101(shape, num_classes)

        print ('[+] ResNet-101 model built')
        return self.model

    def build_resnet_152(self, shape, num_classes):
        self.model = ResnetBuilder.build_resnet_152(shape, num_classes)

        print ('[+] ResNet-152 model built')
        return self.model