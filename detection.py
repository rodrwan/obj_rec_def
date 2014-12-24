import cv2
import numpy as np
import make_train_set as mts
import make_test_set as mtes
from numpy.linalg import norm

bin_n = 8 # number of bins

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 0.001, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist

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

def gabor(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def down_sample(gimg, bins):
    x = gimg.shape[0]/bins
    y = gimg.shape[1]/bins
    nimg = np.zeros(64)
    j = 0
    for i in np.arange(0, gimg.shape[0], bins):
        cell_tmp = gimg[i:i+bins, i:i+bins]
        if len(cell_tmp[0]) > 0:
            _max = np.amax(cell_tmp)
            nimg[j] = np.fabs(_max)
            j += 1
    return nimg[:64]

def scale_linear_bycolumn(rawpoints):
    high = 1.0
    low = 0.0
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

#scale_linear_bycolumn
def extract_features(alldata, filters, _type, nmax = 500):
    gabordata = []
    i = 1
    maxexample = len(alldata[:nmax])
    classes = [] #np.zeros(maxexample, np.int32)
    for data in alldata[:nmax]:
        # print ">>> example " + str(i) + "/" + str(maxexample)
        img = data[0]
        bndbox = data[1]
        cls = data[2]

        img = cv2.imread(img, 0)
        img = img[bndbox[3]:bndbox[2], bndbox[1]:bndbox[0]]
        # cv2.imshow('asf', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if _type == 'gabor':
            general_data = np.array(scale_linear_bycolumn(down_sample(gabor(img, filters), 8)))
        else:
            general_data = hog(img)
        classes.append(cls)
        gabordata.append(general_data)
        # duplicate example by flipping the input image
        if cls == 1:
            img = cv2.flip(img, 1)
            if _type == 'gabor':
                general_data = np.array(scale_linear_bycolumn(down_sample(gabor(img, filters), 8)))
            else:
                general_data = hog(img)
            classes.append(cls)
            gabordata.append(general_data)

        i += 1

    if _type == 'gabor':
        return np.float32(gabordata).reshape(-1, 64)/255, np.array(classes)
    else:
        return np.float32(gabordata), np.array(classes)

def evaluate_model(resp, labels):
    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print


print "Loading dataset...\n"
filters = build_filter()
import sys
cls = sys.argv[1] # 'horse'
algo = sys.argv[2] # 'hog'
trainDataPos, trainDataNeg = mts.make_train_set(cls)

print "Training: %s, using %s" % (cls, algo)
################### training process ###################
print "Process positive examples"
trainDataPos, respPos = extract_features(trainDataPos, filters, algo, nmax=2000)
print "Process negative examples"
trainDataNeg, respNeg = extract_features(trainDataNeg, filters, algo, nmax=2000)

trainData = np.vstack((trainDataPos, trainDataNeg))
respPos = np.vstack(respPos)
respNeg = np.vstack(respNeg)
responses = np.vstack((respPos, respNeg))

print "\nTraining..."
model = SVM()
model.train(trainData, responses)
model.save('svm_data.dat')

print "Testing...\n"
################### testing process ###################
testData = mtes.make_test_set(cls)
testData, responses = extract_features(testData, filters, algo)
result = model.predict(testData)

################### Accuracy ###################
mask = result==responses
correct = np.count_nonzero(mask)
print "Accuracy: %.2f%%" % (correct*100.0/result.size)
err = (result != responses).mean()
print 'error: %.2f%%' % (err*100)

evaluate_model(result, responses)