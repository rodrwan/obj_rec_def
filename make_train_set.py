import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
try:
  from BeautifulSoup import BeautifulSoup
except:
  from bs4 import BeautifulSoup


class MakeTrainSet(object):
    def __init__(self, cls):
        self.cls = cls
        self.main_path = './VOC2011/VOCdevkit/VOC2011/'
        self.annotation_path = self.main_path + 'Annotations/'
        self.image_path = self.main_path + 'JPEGImages/'
        self.training_path = self.main_path + 'ImageSets/Main/trainval.txt'
        self.train_set_image_pos = []
        self.train_set_image_neg = []
        self.annotation_set = []

    def read_train_set(self):
        f = open(self.training_path, 'r')
        content = f.read()
        lines = content.split('\n')
        for line in lines:
            a_file = line + '.xml'
            self.annotation_set.append(a_file)

    def read_annotation(self):
        files = [f for f in self.annotation_set if isfile(join(self.annotation_path, f))]
        cfile = 1
        nfile = len(files)
        for fi in files:
            #print "Reading trainval "+str(cfile)+"/"+str(nfile)
            cfile+=1
            file_name = self.annotation_path + fi
            f = open(file_name, 'r')
            content = f.read()
            y = BeautifulSoup(str(content))
            img = self.image_path + y.filename.string

            for obj in y.findAll('object'):
                bs_obj = BeautifulSoup(str(obj))
                name = bs_obj.findAll('name')
                if name[0].string == self.cls:
                    cls = 1
                else:
                    cls = 0

                bndbox = bs_obj.findAll('bndbox')
                (x,y,w,h) = (
                    int(bndbox[0].xmax.string),
                    int(bndbox[0].xmin.string),
                    int(bndbox[0].ymax.string),
                    int(bndbox[0].ymin.string)
                )
                data = img, (x,y,w,h), cls
                if cls == 1:
                    self.train_set_image_pos.append(data)
                else:
                    self.train_set_image_neg.append(data)

def make_train_set(cls):
    mts = MakeTrainSet(cls)
    mts.read_train_set()
    mts.read_annotation()
    return mts.train_set_image_pos, mts.train_set_image_neg
