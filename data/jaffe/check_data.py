import numpy as np
import os, cv2

imgs = np.load('test_set_jaffe.npy')
lbls = np.load('test_labels_jaffe.npy')

for i in range(imgs.shape[0]):
	print (lbls[i])
	cv2.imshow('Img', imgs[i])
	cv2.waitKey(0)