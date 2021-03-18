'''
    Python Standard Modules
'''

import json

from collections import OrderedDict, namedtuple

###################################################################################################################################################

'''
    Data Analysis Modules
'''

import numpy as np 
import pandas as pd

###################################################################################################################################################

'''
    Data Selection Modules
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

from imblearn.under_sampling import RandomUnderSampler

###################################################################################################################################################

class Splitter:
    
    def __init__(self, filename, proportions, splitter, parametres):
        
        self.indexes = None
        self.dataframe = pd.read_csv(filename)
        self.parametres = parametres
        self.proportions = proportions
        self.splitter = splitter 
        
    def filter_df(self, filter_query):
        self.dataframe = self.dataframe[filter_query].reset_index()
        
    def get_balanced(self):
        random_sampler = RandomUnderSampler(random_state = self.parametres['random_seed'],
                                            ratio = self.proportions, 
                                            return_indices = True)
        
        text_under,data_status_under,indexes = random_sampler.fit_sample(self.dataframe.index.reshape(-1, 1), 
                                                                         self.dataframe.data_status.reshape(-1, 1))
        
        self.dataframe = self.dataframe.iloc[indexes, :].reset_index().drop('index', axis = 1)
        
    def get_indexes(self):
        
        self.indexes = OrderedDict(self.splitter(self, **self.parametres))
            

def train_test_split_simple(splitter, **parametres):
    
    _, _,train_index, test_index, = train_test_split(splitter.dataframe.text, 
                                                     splitter.dataframe.index,
                                                     random_state = parametres['random_seed'], 
                                                     test_size = parametres['test_size'], 
                                                     shuffle = parametres['shuffle'], 
                                                     stratify = parametres['stratify']) 

    return {'train' : train_index, 'test' : test_index}


def train_test_split_KFOLD(splitter, **parametres):
    split_part = 0
    train_test_indexes = {}
    
    k_fold = KFold(n_splits = parametres['num_of_split'], 
                   shuffle = parametres['shuffle'], 
                   random_state = parametres['random_seed'])
    
    for train_index, test_index in k_fold.split(splitter.dataframe.index):
        split_part += 1
        train_test_indexes[split_part] = {
            'train' : train_index,
            'test' : test_index
        }
        
    return train_test_indexes

def train_test_split_StratifiedKFOLD(splitter, **parametres):
    split_part = 0
    train_test_indexes = {}
    stratified_k_fold = StratifiedKFold(n_splits = parametres['num_of_splits'], 
                                        shuffle = parametres['shuffle'], 
                                        random_state = parametres['random_seed'])
    
    for train_index, test_index in stratified_k_fold.split(splitter.dataframe.index, splitter.dataframe.post_label):
        split_part += 1
        train_test_indexes[split_part] = {
            'train' : train_index,
            'test' : test_index
        }
        
    return train_test_indexes