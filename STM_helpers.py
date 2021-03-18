import pandas as pd

from functools import reduce

from constants import *
from data_selection import *
from pytorch_data_processing import *


""" 
    Pytorch Modules
"""

import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as Functional


from torch.autograd import Variable
import torch.utils.data as data_utils


SHUFFLE = True 
BATCH_SIZE = 128

train_test_splitter = Splitter(filename = 'additional_data/hermes_all_data.csv', 
                               proportions = balanced_data, 
                               splitter = train_test_split_simple,
                               parametres = {
                                   'random_seed' : 42, 
                                   'shuffle' : True, 
                                   'test_size' : 0.1,
                                   'stratify' : None,
                                   'num_of_split' : None
                               })

# For balance dataset
train_test_splitter.get_balanced()

# Indexes for train_test
train_test_splitter.get_indexes()


if 'KFOLD' in train_test_splitter.splitter.__name__:
    train_indexes = reduce(lambda x,y : x.extend(y), sequence = [item['train'] for item in list(train_test_splitter.indexes.values())])
    test_indexes = reduce(lambda x,y : x.extend(y), sequence = [item['test'] for item in list(train_test_splitter.indexes.values())])

else:
    train_indexes = train_test_splitter.indexes['train']
    test_indexes = train_test_splitter.indexes['test']

train_dataframe = train_test_splitter.dataframe.iloc[train_indexes, :]
test_dataframe = train_test_splitter.dataframe.iloc[test_indexes, :]
val_dataframe = pd.read_csv(val_data_path)



transform_text = transforms.Compose(             
        [
            Deeppavlov_Spelling_Corrector(),
            # Yandex_Lemmatizer(),
            BertPreprocessor(),
            ToTensor()
        ]
    )
 
train_dataset = Complaints_Dataset(dataframe = train_dataframe,  transform = transform_text)
test_dataset = Complaints_Dataset(dataframe = test_dataframe, transform = transform_text)
val_dataset = Complaints_Dataset(dataframe = val_dataframe, ransform = transform_text)


train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
validation_loader = data_utils.DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)



for batch_index, batch in enumerate(train_loader):
    texts, post_labels, input_ids, input_masks = batch['text'], batch['post_label'], \
                                                 batch['bert_input_ids'], batch['bert_input_masks']

    print('batch_index : {}'.format(batch_index))
    print()
    print('text.shape : {}'.format(batch_index))
    print('post_labels.shape : {}'.format(batch_index))
    print('input_ids.shape : {}'.format(batch_index))
    print('input_masks.shape : {}'.format(batch_index))
    print()

    if(batch_index == 10):
        break 