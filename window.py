import cv2
import numpy as np
import time

def gabor(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def build_filter(fsize = 11, orientation = 8, scale = 4):
    filters = []
    lambd = 1
    gamma = 0.25
    sigma = np.sqrt(3)
    for theta in np.arange(0, np.pi, np.pi/(orientation*scale)):
        kern = cv2.getGaborKernel((fsize, fsize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def detect_window(img, filters):
    cv2.imshow('img', gabor(img, filters))
    x, y = img.shape
    print x, y
    for i in range(int(x/2)):
        for j in range(int(y/2)):
            tmp = img[i:int(x/2)+i, j:int(y/2)+j]
            cv2.imshow('tmp', gabor(tmp, filters))
            cv2.waitKey(0)
            cv2.destroyWindow('tmp')
    cv2.destroyAllWindows()

import make_train_set as mts
import sys
cls = sys.argv[1] # 'horse'

filters = build_filter()
print "loading example data..."
trainDataPos, trainDataNeg = mts.make_train_set(cls)

for data in trainDataPos[:10]:
    img = data[0]
    img = cv2.imread(img, 0)
    detect_window(img, filters)