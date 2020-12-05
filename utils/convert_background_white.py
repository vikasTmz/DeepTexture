import cv2
import numpy as np

for i in range(0,5):
	name = '00' + str(i) + '.png'
	image = cv2.imread(name)
	for background in [[190,190,190]]:
		image[np.where((image>background).all(axis=2))] = [255,255,255]
		image[np.where((image>background).all(axis=1))] = [255,255,255]
	cv2.imwrite(name, image)
