'''
    Standard Modules
'''

import os
import sys


'''
    Internal_Modules
'''

from tqdm import tqdm
from Images_DataLoader import *
import LIVEFACE_MOBILE_ML_Tools as ml_tools



all_set = pd.read_csv('Data_Locations/FINAL_ALL_LIVEFACE_DATASET_V4.csv')
train_set = all_set[all_set['set_status'] == 'Train'].reset_index().drop('index', axis = 1)
validation_set = all_set[all_set['set_status'] == 'Validation'].reset_index().drop('index', axis = 1)

labels = train_set['label']
image_path = train_set['image_path_cropped']

dict_container, train_set = ml_tools.get_train_test_split(image_path = image_path, 
                                                          labels = labels, 
                                                          split_case = 'SKFold', 
                                                          seed = 17, 
                                                          images_df = train_set,
                                                          n_splits = 15,
                                                          filename = sys.stdout)

train_set.to_csv('Data_Locations/TRAIN_DATASET.csv', index = False)
validation_set.to_csv('Data_Locations/VALIDATION_DATASET.csv', index = False)


# train_set = pd.read_csv('Data_Locations/TRAIN_DATASET.csv')
# validation_set = pd.read_csv('Data_Locations/VALIDATION_DATASET.csv')


train_dataset = train_set[train_set['SKFold'] == 'train'].reset_index().drop('index', axis = 1)
test_dataset = train_set[train_set['SKFold'] == 'test'].reset_index().drop('index', axis = 1)



def extraction(dataset, dataset_type, output_size, patch_num):
    
    def extract(sample, dir_path, filename):
        
        path, labels = [], []

        image_path, label = sample['image_path'], sample['label']

        image = cv2.imread(image_path)

        h, w = image.shape[:2]

        new_h, new_w = output_size

        tops = np.random.randint(low = 0, high = h - new_h, size = patch_num)
        lefts = np.random.randint(low = 0, high = w - new_w, size = patch_num)

        for i in range(len(tops)):
            
            patch_filename = 'patch_{}_{}'.format(i,filename)
            full_path = os.path.join(dir_path,patch_filename)
            cv2.imwrite(filename = full_path, img = image[tops[i] : tops[i] + new_h, lefts[i] : lefts[i] + new_w])

            path.append(full_path)

        labels.extend([label] * patch_num)    
    
        return path, labels

    path, labels, metadata = [], [], []
    new_dataset = pd.DataFrame()

    for i in tqdm(range(len(dataset['image_path_cropped']))):
        
        sample = {
            'image_path' : dataset['image_path_cropped'][i],
            'label' : dataset['label'][i]
        }
        
        path_save = os.path.join('LIVEFACE_DATA_CROPPED_PATCHES', dataset_type, sample['label']) 
        
        if not os.path.exists(path_save) : os.makedirs(path_save, exist_ok = True)

        filename = dataset['image_path_cropped'][i].split('/')[-1]

        patch_path, patch_labels = extract(sample = sample, dir_path = path_save, filename = filename)

        path.extend(patch_path)
        labels.extend(patch_labels)
        metadata.extend(['_'.join(dataset['image_metadata'][i].split('/')[3:])] * patch_num)

    new_dataset['PATH'] = path
    new_dataset['METADATA'] = metadata
    new_dataset['LABEL'] = labels

    return new_dataset

train_dataset_patches = extraction(dataset = train_dataset, dataset_type = 'TRAIN',output_size = (96,96), patch_num = 8)
test_dataset_patches = extraction(dataset = test_dataset, dataset_type = 'TEST',output_size = (96,96), patch_num = 8)
validation_dataset_patches = extraction(dataset = validation_set, dataset_type = 'validation',output_size = (96,96), patch_num = 8)


train_dataset_patches.to_csv('Data_Locations/TRAIN_DATASET_PATCHES.csv', index = False)
test_dataset_patches.to_csv('Data_Locations/TEST_DATASET_PATCHES.csv', index = False)
validation_dataset_patches.to_csv('Data_Locations/VALIDATION_DATASET_PATCHES.csv', index = False)

print('train_datset_length {} : train_datset_patches_length {} : expected value {}'.format(len(train_dataset), len(train_dataset_patches), len(train_dataset) * 8))
print('test_datset_length {} : test_datset_patches_length {} : expected value {}'.format(len(test_dataset), len(test_dataset_patches), len(test_dataset) * 8))
print('validation_datset_length {} : validation_datset_patches_length {} : expected value {}'.format(len(validation_set), len(validation_set), len(validation_set) * 8))
