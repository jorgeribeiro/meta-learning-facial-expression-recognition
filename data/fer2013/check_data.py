import numpy as np
import os, cv2

imgs = np.load('data_set_fer_2013.npy')
lbls = np.load('data_labels_fer_2013.npy')

for i in range(imgs.shape[0]):
	print (lbls[i])
	cv2.imshow('img', imgs[i])
	cv2.waitKey(0)