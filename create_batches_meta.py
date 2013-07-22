import sys
import os
import Image
import numpy as np
import cPickle
import random
import csv
from math import ceil
from time import time

SIZE_BATCH = 1024
SIZE_IMG = 48

class CreateBatchesMeta:
     def __init__(self, fcsv, dir_batch):
        self.data = self.load_csv(fcsv)
        self.dir_batch = dir_batch

     def load_csv(self, fcsv):
        data = []
        label = []
        reader = csv.reader(file(fcsv, 'r')) 
        for line in reader:
           if reader.line_num == 1:
              continue
           tmp1 = line[1].split()
           tmp2 = [int(pixel) for pixel in tmp1]
           data.append(tmp2)
        return data

     def build_batch_meta(self):
          imgs = np.array(self.data)
          sum = imgs.sum(axis=0) 
          avg = sum/len(imgs)
          path_batch = self.dir_batch + '/batches.meta'
          f = open(path_batch, 'wb')
          data = {}
          data['label_names'] = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
          data['num_vis'] = SIZE_IMG * SIZE_IMG
          data['num_cases_per_batch'] = SIZE_BATCH
          data['data_mean'] = np.reshape(avg, (data['num_vis'],1))
          cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
          f.close()
          print path_batch
          
          

if __name__ == "__main__":
    fcsv = sys.argv[1]
    dir_batch = sys.argv[2]
    model = CreateBatchesMeta(fcsv, dir_batch)
    model.build_batch_meta()
             
