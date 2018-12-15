import numpy as np
import os, cv2

imgs = np.load('test_set_ck_extended_no_resize.npy')
lbls = np.load('test_labels_ck_extended_no_resize.npy')

for i in range(imgs.shape[0]):
	print (lbls[i])
	cv2.imshow('img', imgs[i])
	cv2.waitKey(0)
