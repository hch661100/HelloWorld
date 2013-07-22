import sys
import os
import Image
import numpy as np
import cPickle
import random
import csv
from math import ceil
from time import time

SIZE_IMG = 48

class CreateTestBatch:
     def __init__(self, fcsv, dir_batch):
        self.data, self.label = self.load_csv(fcsv)
        self.dir_batch = dir_batch

     def load_csv(self, fcsv):
        data = []
        label = []
        reader = csv.reader(file(fcsv, 'r')) 
        for line in reader:
           if reader.line_num <= 28710:
              continue
           label.append(int(line[0]))
           tmp1 = line[1].split()
           tmp2 = [int(pixel) for pixel in tmp1]
           data.append(tmp2)
        print len(data)
        return data, label

     def build_batch(self):
          path_batch = self.dir_batch + '/data_batch_' + str(1000)
          f = open(path_batch, 'wb')
          imgs = np.array(self.data)
          data = {}
          data['batch_label'] = 'test_label'
          data['labels'] = self.label
          data['data'] = imgs.T
          data['filenames'] = [i for i in range(len(self.label))]
          cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
          f.close()
          print path_batch
          
          

if __name__ == "__main__":
    fcsv = sys.argv[1]
    dir_batch = sys.argv[2]
    model = CreateTestBatch(fcsv, dir_batch)
    model.build_batch()
             
