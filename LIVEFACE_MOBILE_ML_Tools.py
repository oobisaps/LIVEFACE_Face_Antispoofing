'''
    Standard Modules
'''

import os
import re
import sys
import time
import copy
import itertools
import linecache
import collections
import pickle as pkl

from itertools import cycle
from collections import Counter

from types import MappingProxyType

'''
    External Modules
'''

# import plotly
import numpy as np
import pandas as pd

# import seaborn as sns
import matplotlib.pyplot as plt

# from tqdm import tqdm
# from tqdm import tqdm_notebook
from matplotlib.pyplot import figure, show
# from imblearn.under_sampling import RandomUnderSampler,NearMiss

'''
    model selection modules
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

'''
    metrics modules
'''

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from imblearn.metrics import classification_report_imbalanced

'''
    models modules
'''

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

'''
    Pytorch Modules
'''

import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image

'''
    Internal Modules
'''

from NN_summary import summary
from Loss_Functions import *
from Images_DataLoader import *




def plot_confusion_matrix(y_true,y_pred,labels, path):
    
    def plot_confusion_matrix_demo(cm, classes,
                                   normalize = False,
                                   title = 'Confusion matrix',
                                   cmap = plt.get_cmap('Blues')):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        
        plt.xticks(tick_marks, classes) #, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment = "center", 
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.grid('off')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        title = 'Confusion_Maxtrix.png' if not normalize else 'Normalized_Confusion_Maxtrix.png' 

        plt.savefig(os.path.join(path,title))
    
    
    
    cnf_matrix = confusion_matrix(y_true,y_pred)
    np.set_printoptions(precision = 2)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix_demo(cnf_matrix, classes = labels,
                          title = 'Confusion matrix, without normalization')

    

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix_demo(cnf_matrix, classes = labels, normalize=True,
                          title = 'Normalized confusion matrix')

    plt.show()


def result_wrapper(acc,prec,rec,f1,roc_auc,title,parametrs):
    
    results = {
        'title' : title,
        'accuracy' : acc,
        'precision' : prec,
        'recall':rec,
        'f1_score' : f1,
        'roc_auc' : roc_auc,
        # 'time' : time,
        # 'vector_size' : vector_size,
        'parametrs' : parametrs,
    }
    
    return results


def pickle_load(path):
    pkl_file = open(path, 'rb')
    data = pkl.load(pkl_file)
    pkl_file.close()
    
    return data

def pickle_dump(path,data):
    pickle_out = open(path,"wb")
    pkl.dump(data, pickle_out)
    pickle_out.close()


def pandas_report(frame,data):
    data = pd.DataFrame(data)
    
    return frame.append(data)

def get_results(y_true,y_pred, time = 0):
    
    result_med = collections.namedtuple(
                                    'result_med',
                                        [
                                            'precision',
                                            'recall',
                                            'f1_score',
                                            'roc_auc',
                                            'accuracy',
                                            # 'time'
                                    ]
                            )
    
    result_med.accuracy = accuracy_score(y_true,y_pred)
    result_med.precision = precision_score(y_true,y_pred,average = None)
    result_med.recall = recall_score(y_true,y_pred,average = None)
    result_med.f1_score = f1_score(y_true,y_pred,average = None)
    result_med.roc_auc = roc_auc_score(y_true,y_pred)
    # result_med.time = time

    return result_med


def get_exception(filename = sys.stdout):

    filename = open(filename, 'a+') if filename != sys.stdout else filename

    _, exc_obj, traceback = sys.exc_info()
    frame = traceback.tb_frame
    line_number = traceback.tb_lineno
    filename = frame.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, line_number, frame.f_globals)
    output = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, line_number, line.strip(), exc_obj)
    
    print(output)


def pandas_status_indexing(data_statistics,indexes,new_key):
        '''
            set status text will be used in train set or test set.
            
            Keyword arguments : 
            
            data_statistics -- pandas dataframe with all data
            indexes -- indexes of data split(train indexes)
            new_key -- new column with assigned values
            
            return data_statistics
        '''
        
        all_indexes = list(data_statistics.index)
        values = ['train' if index in indexes else 'test' for index in all_indexes]
        
        data_statistics[new_key] = values
        
        return data_statistics

def print_results(res_dict, filename = sys.stdout):

        print(' ', file = filename)
        print('TRAIN SIZE Features : ',res_dict['X_train'].shape[0], file = filename)
        print('TEST SIZE Features : ',res_dict['X_test'].shape[0], file = filename)
        print('TRAIN indexes : ',res_dict['train_indexes'][:10], file = filename)
        print('TEST indexes : ',res_dict['test_indexes'][:10], file = filename)
        print('TRAIN and TEST indexes intersection : ',list(set(res_dict['train_indexes']).intersection(set(res_dict['test_indexes']))), file = filename)
        print(' ', file = filename)
        print('CLass Distribution : ' , file = filename)
        print('TRAIN :', file = filename)
        
        unique_tr,counts_tr = np.unique(res_dict['Y_train'], return_counts = True)
        counter_tr = dict(zip(unique_tr,counts_tr))
        
        print('  LIVE ',counter_tr['LIVE'], file = filename)
        print('  FAKE ',counter_tr['FAKE'], file = filename)
        print ('TEST :', file = filename)
        
        unique_ts,counts_ts = np.unique(res_dict['Y_test'], return_counts = True)
        counter_ts = dict(zip(unique_ts,counts_ts))
        
        print('  LIVE ',counter_ts['LIVE'], file = filename)
        print('  FAKE ',counter_ts['FAKE'], file = filename)



def get_train_test_split(image_path, labels, split_case, seed, test_size = None, n_splits = 'warn', shuffle = False, images_df = None, filename = sys.stdout):

    labels = np.array(labels)
    image_path = np.array(image_path)

    keys = ('X_train', 'Y_train', 'train_indexes', 'X_test', 'Y_test', 'test_indexes')

    filename = open(filename, 'a+') if filename != sys.stdout else filename

    
    if split_case == 'train_test_split':

        X_train,Y_train,X_test,Y_test = train_test_split(image_path,
                                                         labels,
                                                         test_size = test_size,
                                                         random_state = seed)
        
        values = (X_train, Y_train, None, X_test, Y_test, None)
        result_dict = dict(list(zip(keys, values)))

        images_df[split_case] = 'index_unreachable'

    elif split_case == 'KFold':
        
        kF = KFold(n_splits, shuffle, random_state = seed)

        for train_indexes, test_indexes in kF.split(image_path):
            X_train, Y_train = image_path[train_indexes], labels[train_indexes]
            X_test, Y_test = image_path[test_indexes], labels[test_indexes]

        values = (X_train, Y_train, train_indexes, X_test, Y_test, test_indexes)
        result_dict = dict(list(zip(keys, values)))

        print_results(res_dict = result_dict, filename = filename)

        images_df = pandas_status_indexing(data_statistics = images_df,
                                           indexes = train_indexes,
                                           new_key = split_case)

        # print('{} : {}'.format(len(X_train), len(train_indexes)))
    
    else:
        
        skF = StratifiedKFold(n_splits, shuffle, random_state = seed)

        for train_indexes, test_indexes in skF.split(image_path,labels):
            X_train, Y_train = image_path[train_indexes], labels[train_indexes]
            X_test, Y_test = image_path[test_indexes], labels[test_indexes]

        values = (X_train, Y_train, train_indexes, X_test, Y_test, test_indexes)
        result_dict = dict(list(zip(keys, values)))

        print_results(res_dict = result_dict, filename = filename)

        images_df = pandas_status_indexing(data_statistics = images_df,
                                           indexes = train_indexes,
                                           new_key = split_case)

        # print('{} : {}'.format(len(X_train), len(train_indexes)))

    return (result_dict, images_df)



def test_model(loader,model_net,abs_path,title,type_of,filename, device):

    statistics = pd.DataFrame()

    y_true, y_pred, probas, path_all = [], [], [], []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 0):
            images, labels, path = batch['image'], batch['label'], batch['path']  
            images, labels = images.to(device), labels.to(device)

            outputs = model_net(images.float())

            proba, y_pred_batch = torch.max(outputs,1)
            proba = list(map(lambda x : float(x), proba))
            y_pred_batch = list(map(lambda x : int(x), y_pred_batch))
            y_true_batch = list(map(lambda x : int(x), labels))

            probas.extend(proba)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
            path_all.extend(path)

            # break


    statistics['PATH'] = path_all
    statistics['LABEL'] = y_true
    statistics['PREDICTED'] = y_pred
    statistics['PROBA'] = probas

    statistics.to_csv(os.path.join(abs_path,title + '_' + type_of + '_statistics.csv'), index = False)

    
    
    print(type_of, file = filename)    
    print(100 * '.', file = filename)
    print('', file = filename)
    print('TOTAL_RESULTS : ', file = filename)
    print('', file = filename)
    print(classification_report(y_true, y_pred, labels = [0,1]), file = filename)

    


    return None


def train_model(epochs, learning_rate, model_net, batch_size, num_patches, loader, test_loader, val_loader, input_size, color_space, title, abs_path, version, filename, std_out = sys.stdout, optimization_type = None):

    statistics = pd.DataFrame()
    
    PATH_SAVE = os.path.join(abs_path,title, title + '_' + version)

    if os.path.exists(PATH_SAVE): pass    
    else: os.makedirs(name = PATH_SAVE, exist_ok = True)

    current_time = time.ctime()
    current_time = current_time.replace(' ','_')

    filename = open(os.path.join(PATH_SAVE,title + '_model_logs_' + current_time + '.txt'), 'w+') if filename != sys.stdout else filename

    model_logs = dict(list(zip(range(epochs), [None] * epochs)))    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_net.parameters(), lr = learning_rate, momentum = 0.9)

    # model_summary = summary(model = model_net, input_size = input_size, filename = os.path.join(PATH_SAVE,title + '.txt'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_net.to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!", file = filename)
        
        model_net = nn.DataParallel(model_net)
    

    model_net = model_net.float()

    models_state_info = {
        'model_state_dict' : None,
        'optimizer_state_dict' : None,
        # 'summary' : model_summary,
        'color_space' : color_space,
        'epochs' : epochs,
        'learnin_rate' : learning_rate,
        'total_loss_train' : None,
        'model_logs' : None
    }

    total_loss = 0.0

    parametrs_string = 'learning_rate : {}, momentum : {}, epochs : {}, input_shape : {}, color_space : {}'.format(learning_rate,0.9,epochs,input_size,color_space)

    print('HYPERPARAMETRS : ', file = filename)
    print(parametrs_string, file = filename)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        y_true, y_pred, probas,path_all = [], [], [], []
        total_loss = 0.0
        
        print('Epoch_Num : {}'.format(epoch), file = filename)
        print('Epoch_Num : {}'.format(epoch),)

        for batch_index, batch in enumerate(loader, 0):
                
            images, labels, path = batch['image'], batch['label'], batch['path']  
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = model_net(images.float())

            proba, y_pred_batch = torch.max(outputs,1)
            proba = list(map(lambda x : float(x), proba))


            y_pred_batch = list(map(lambda x : int(x), y_pred_batch))
            y_true_batch = list(map(lambda x : int(x), labels))

            probas.extend(proba)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
            path_all.extend(path)

            # logs_string_format = 'EPOCH {} -> batch {} -> loss_batch -> {}'.format(epoch, batch_index)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print('{} -> Epoch {} -> Batch_index {} -> {} -> {}'.format(time.ctime() ,epoch, batch_index,'loss', loss.item()), file = filename)
            running_loss += loss.item()
            total_loss += loss.item()

            if batch_index % 2000 == 1999:    # print every 2000 mini-batches
                print(' ',file = filename)
                print(100 * '*', file = filename)
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, batch_index + 1, running_loss / 2000), file = filename)
                running_loss = 0.0
                print(100 * '*', file = filename)
                print(' ',file = filename)
            
            # break


        norm_coeff = total_loss if batch_index == 0 else total_loss / batch_index

        epoch_logs_format = '{} -> Epoch_loss : {}'.format(time.ctime(), total_loss / norm_coeff)

        print(' ', file = filename) 
        print(100 * '-', file = filename)
        print(epoch_logs_format, file = filename)
        print('Metrics :',file = filename)
        print(classification_report(y_true, y_pred, labels = [0,1]), file = filename)
        print(' ', file = filename)
        print(100 * '-',file = filename)
        print('', file = filename)

        results = get_results(y_true, y_pred)

        results = result_wrapper(title = title + '_' + version,
                             acc = results.accuracy,
                             prec = results.precision,
                             rec = results.recall,
                             f1 = results.f1_score,
                             roc_auc = results.roc_auc,
                             parametrs = parametrs_string)
        
        # print(results)

        model_logs[epoch] = results

        # break

    

    statistics['PATH'] = path_all
    statistics['LABEL'] = y_true
    statistics['PREDICTED'] = y_pred
    statistics['PROBA'] = probas

    statistics.to_csv(os.path.join(PATH_SAVE,title + '_statistics.csv'), index = False)

    # final_results = model_logs[epochs - 1]
    final_results = model_logs[0]
    
    final_results = pd.DataFrame(final_results)
    final_results.groupby(['title'], as_index = False).mean().to_csv(os.path.join(PATH_SAVE, title + '_model_results.csv'), index = False)

    model_state_info = {
        'model_state_dict' : None,
        'optimizer_state_dict' : None,
        # 'summary' : model_summary,
        'color_space' : color_space,
        'epochs' : epochs,
        'learnin_rate' : learning_rate,
        'total_loss_train' : None,
        'model_logs' : None
    }

    model_state_info['model_state_dict'] = model_net.state_dict()
    model_state_info['optimizer_state_dict'] = optimizer.state_dict()
    model_state_info['total_loss_train'] = total_loss / norm_coeff
    model_state_info['model_logs'] = model_logs

    torch.save(model_state_info, os.path.join(PATH_SAVE,title + '_model_state_info.pth'))
    pickle_dump(path = os.path.join(PATH_SAVE,title + '_model_logs.pkl'), data = model_logs)

    plot_confusion_matrix(y_true = y_true, y_pred = y_pred, labels = ['FAKE','LIVE'], path = os.path.join(PATH_SAVE))
    
    print(100 * '.', file = filename)
    print('', file = filename)
    print('Finished Training', file = filename)
    print('', file = filename)
    print('TOTAL_RESULTS : ', file = filename)
    print('', file = filename)
    print('LOSS : ', total_loss / norm_coeff, file = filename)
    print('', file = filename)
    print(classification_report(y_true, y_pred, labels = [0,1]), file = filename)
    print('', file = filename)

    test_model(loader = test_loader, model_net = model_net, abs_path = PATH_SAVE, title = title, type_of = 'TEST', filename = filename, device = device)
    test_model(loader = val_loader, model_net = model_net, abs_path = PATH_SAVE, title = title, type_of = 'VAL', filename = filename, device = device)




def get_style_model_and_losses(cnn, 
                               normalization_mean, 
                               normalization_std,
                               style_image, 
                               content_image,
                               content_layers, 
                               style_layers):
    
    cnn = copy.deepcopy(cnn)

    normalization = Normalizer(normalization_mean, normalization_std)

    contents_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
        
            i += 1
            name = 'conv_{}'.format(i)
        
        elif isinstance(layer, nn.ReLU):
            name = 'reli_{}'.format(i)
            layer = nn.ReLU(inplace = False)
        
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        
        else:
            raise RuntimeError('Unrecognized layer : {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        if name in content_layers:

            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            contents_losses.append(content_loss)
        
        if name in style_losses:

            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)
        
        
        for i in range(len(model) - 1, - 1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i+1)]

    return model, style_losses, contents_losses


def get_input_optimizer(imput_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=1000,
                        style_weight=100000, content_weight=1):

        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                #взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img
    




def make_style_transfer(loader, abs_path, style_image, iter_num):

    content_layers_default = ['conv_5']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vgg_19 = models.vgg19(pretrained = True).features.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!", file = filename)
        
        vgg_19 = nn.DataParallel(vgg_19).eval()
    

    for batch_index, batch in enumerate(loader, 0):

        image, label, _ = batch['image'], batch['label']

        break
    
    return None


     