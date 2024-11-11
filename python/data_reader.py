from __future__ import print_function
import cPickle as pickle
import config
import random

class Data_Reader:
    def __init__(self, cur_train_index=0, load_list=False):
        self.training_data = pickle.load(open(config.training_data_path, 'rb'))
        self.data_size = len(self.training_data)
        if load_list:
            self.shuffle_list = pickle.load(open(config.index_list_file, 'rb'))
        else:    
            self.shuffle_list = self.shuffle_index()
        self.train_index = cur_train_index
    def get_batch_num(self, batch_size):
        return self.data_size // batch_size