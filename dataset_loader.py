from os.path import join
import numpy as np
from constants import *

class DatasetLoader(object):

	def __init__(self):
		pass

	def jaffe_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_JAFFE_IMAGES_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_JAFFE_LABELS_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_JAFFE_IMAGES_TEST_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_JAFFE_LABELS_TEST_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] JAFFE dataset loaded')

	def ck_extended_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_IMAGES_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_LABELS_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_IMAGES_TEST_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_LABELS_TEST_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] CK+ dataset loaded')

	def ck_extended_no_resize_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_NO_RESIZE_IMAGES_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_NO_RESIZE_LABELS_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_NO_RESIZE_IMAGES_TEST_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_CK_EXTENDED_NO_RESIZE_LABELS_TEST_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] CK+ dataset no resize loaded')

	def fer_2013_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_TEST_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_TEST_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] FERPlus [FULL] dataset loaded')

	def fer_2013_medium_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_MEDIUM_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_MEDIUM_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_TEST_MEDIUM_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_TEST_MEDIUM_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] FERPlus [MEDIUM] dataset loaded')

	def fer_2013_small_load_from_save(self):
		self._images = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_SMALL_FILENAME))
		self._labels = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_SMALL_FILENAME))
		self._images_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_IMAGES_TEST_SMALL_FILENAME))
		self._labels_test = np.load(join(DATA_PATH, SAVE_DATASET_FER_2013_LABELS_TEST_SMALL_FILENAME))
		self._images = self._images.reshape([-1, self._images.shape[1], self._images.shape[2], 1])
		self._images_test = self._images_test.reshape([-1, self._images_test.shape[1], self._images_test.shape[2], 1])
		print ('[+] FERPlus [SMALL] dataset loaded')

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def images_test(self):
		return self._images_test

	@property
	def labels_test(self):
		return self._labels_test

	@property
	def num_examples(self):
		return self._num_examples