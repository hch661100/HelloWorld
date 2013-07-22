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

class CreateDataBatch:
     def __init__(self, fcsv, dir_batch):
        self.data, self.label = self.load_csv(fcsv)
        self.dir_batch = dir_batch

     def load_csv(self, fcsv):
        data = []
        label = []
        reader = csv.reader(file(fcsv, 'r')) 
        for line in reader:
           if reader.line_num == 1:
              continue
           label.append(int(line[0]))
           tmp1 = line[1].split()
           tmp2 = [int(pixel) for pixel in tmp1]
           data.append(tmp2)
        print len(data)
        return data, label

     def build_batch(self):
        num_batch = len(self.data)/SIZE_BATCH
        for i in range(num_batch):
          batch_data = self.data[i*SIZE_BATCH: (i+1)*SIZE_BATCH]
          batch_label = self.label[i*SIZE_BATCH: (i+1)*SIZE_BATCH]
          path_batch = self.dir_batch + '/data_batch_' + str(i+1)
          f = open(path_batch, 'wb')
          imgs = np.array(batch_data)
          data = {}
          data['batch_label'] = 'training batch ' + str(i+1) + ' of ' + str(num_batch)
          data['labels'] = batch_label
          data['data'] = imgs.T
          data['filenames'] = [i for i in range(SIZE_BATCH)]
          cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
          f.close()
          print path_batch
          
          

if __name__ == "__main__":
    fcsv = sys.argv[1]
    dir_batch = sys.argv[2]
    model = CreateDataBatch(fcsv, dir_batch)
    model.build_batch()
             
