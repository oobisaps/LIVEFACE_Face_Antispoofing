from torch.nn import *

import torch.optim as optim
import torch.nn.functional as Functional

from NN_summary import summary

class Patch_Based_CNN(Module):

    def __init__(self, in_channels, input_shape, num_classes):
        super(Patch_Based_CNN, self).__init__()

        self.in_channels = in_channels
        self.input_shape = input_shape
        self.num_classes = num_classes

        # layer 1
        self.conv_1 = Conv2d(in_channels = self.in_channels, out_channels = 50, kernel_size = (5,5), stride = 1, padding = 2)
        # self.batch_norm_1 = BatchNorm2d(num_features = 50 * self.input_shape[0] * self.input_shape[1])
        self.batch_norm_1 = BatchNorm2d(num_features = 50)
        self.max_pooling_1 = MaxPool2d(kernel_size = (2,2), stride = 2)

        # layer 2
        self.conv_2 = Conv2d(in_channels = 50, out_channels = 100, kernel_size = (3,3), stride = 1, padding = 1)
        # self.batch_norm_2 = BatchNorm2d(num_features = int(100 * (self.input_shape[0] / 2) * (self.input_shape[1] / 2)))
        self.batch_norm_2 = BatchNorm2d(num_features = 100)
        self.max_pooling_2 = MaxPool2d(kernel_size = (2,2), stride = 2)

        # layer 3
        self.conv_3 = Conv2d(in_channels = 100, out_channels = 150, kernel_size = (3,3), stride = 1, padding = 1)
        # self.batch_norm_3 = BatchNorm2d(num_features = int(150 * (self.input_shape[0] / 4) * (self.input_shape[1] / 4)))
        self.batch_norm_3 = BatchNorm2d(num_features = 150)
        self.max_pooling_3 = MaxPool2d(kernel_size = (2,2), stride = 2)

        # layer 4
        self.conv_4 = Conv2d(in_channels = 150, out_channels = 200, kernel_size = (3,3), stride = 1, padding = 1)
        # self.batch_norm_4 = BatchNorm2d(num_features = int(200 * (self.input_shape[0] / 8) * (self.input_shape[1] / 8)))
        self.batch_norm_4 = BatchNorm2d(num_features = 200)
        self.max_pooling_4 = MaxPool2d(kernel_size = (2,2), stride = 2)

        # layer 5
        self.conv_5 = Conv2d(in_channels = 200, out_channels = 250, kernel_size = (3,3), stride = 1, padding = 1)
        # self.batch_norm_5 = BatchNorm2d(num_features = int(250 * (self.input_shape[0] / 16) * (self.input_shape[1] / 16)))
        self.batch_norm_5 = BatchNorm2d(num_features = 250)
        self.max_pooling_5 = MaxPool2d(kernel_size = (2,2), stride = 2)

        # layer 6
        self.fully_connected_1 = Linear(in_features = int(250 * (self.input_shape[0] / 32) * (self.input_shape[1] / 32)), out_features = 1000)
        self.batch_norm_6 = BatchNorm1d(num_features = 1000, )

        # layer 7
        self.dropout = Dropout2d(p = 0.5)
        self.fully_connected_2 = Linear(in_features = 1000, out_features = 400)
        self.batch_norm_7 = BatchNorm1d(num_features = 400)
        
        # output layer with probas
        self.fully_connected_3 = Linear(in_features = 400, out_features = self.num_classes)



    def forward(self, x):
        # layer 1
        # print('LAYER_1')
        conv_1_output = self.conv_1(x)                                                 # Size changes from (3, 96, 96) to (50, 96, 96)
        # print('{} : {}'.format('conv_1',conv_1_output.shape))
        
        batch_norm_1_output = self.batch_norm_1(conv_1_output)                         # Computes the batch_normalization of the first convolution
        # print('{} : {}'.format('batch_norm_1',batch_norm_1_output.shape))
        
        activation_layer_1 = Functional.relu(batch_norm_1_output)                      # Computes the activation of the first convolution
        # print('{} : {}'.format('activation_layer_1',activation_layer_1.shape))
        
        max_pool_1_output = self.max_pooling_1(activation_layer_1)                     # Size changes from (50, 96, 96) to (50, 48, 48)
        # print('{} : {}'.format('max_pool_1',max_pool_1_output.shape))

        # layer 2
        # print('LAYER_2')
        conv_2_output = self.conv_2(max_pool_1_output)                                 # Size changes from (50, 48, 48) to (100, 48, 48)
        # print('{} : {}'.format('conv_2',conv_2_output.shape))

        batch_norm_2_output = self.batch_norm_2(conv_2_output)                         # Computes the batch_normalization of the second convolution
        # print('{} : {}'.format('batch_norm_2',batch_norm_2_output.shape))
        
        activation_layer_2 = Functional.relu(batch_norm_2_output)                      # Computes the activation of the second convolution
        # print('{} : {}'.format('activation_layer_2',activation_layer_2.shape))
        
        max_pool_2_output = self.max_pooling_2(activation_layer_2)                     # Size changes from (100, 48, 48) to ((100, 24, 24)) 
        # print('{} : {}'.format('max_pool_2',max_pool_2_output.shape))

        # layer 3
        # print('LAYER_3')
        conv_3_output = self.conv_3(max_pool_2_output)                                 # Size changes from (100, 24, 24) to ((150, 24, 24))
        # print('{} : {}'.format('conv_3',conv_3_output.shape))

        batch_norm_3_output = self.batch_norm_3(conv_3_output)                         # Computes the batch_normalization of the third convolution
        # print('{} : {}'.format('batch_norm_3',batch_norm_3_output.shape))
        
        activation_layer_3 = Functional.relu(batch_norm_3_output)                      # Computes the activation of the third convolution
        # print('{} : {}'.format('activation_layer_3',activation_layer_3.shape))
        
        max_pool_3_output = self.max_pooling_3(activation_layer_3)                     # Size changes from (150, 24, 24) to ((150, 12, 12))
        # print('{} : {}'.format('max_pool_3',max_pool_3_output.shape))


        # layer 4
        # print('LAYER_4')
        conv_4_output = self.conv_4(max_pool_3_output)                                 # Size changes from (150, 12, 12) to ((200, 12, 12))
        # print('{} : {}'.format('conv_4',conv_4_output.shape))
        
        batch_norm_4_output = self.batch_norm_4(conv_4_output)                         # Computes the batch_normalization of the fourth convolution
        # print('{} : {}'.format('batch_norm_4',batch_norm_4_output.shape))
        
        activation_layer_4 = Functional.relu(batch_norm_4_output)                      # Computes the activation of the fourth convolution
        # print('{} : {}'.format('activation_layer_4',activation_layer_4.shape))
        
        max_pool_4_output = self.max_pooling_4(activation_layer_4)                     # Size changes from (200, 12, 12) to ((200, 6, 6))
        # print('{} : {}'.format('max_pool_4',max_pool_4_output.shape))


        # layer 5
        # print('LAYER_5')
        conv_5_output = self.conv_5(max_pool_4_output)                                 # Size changes from (200, 6, 6) to ((250, 6, 6))
        # print('{} : {}'.format('conv_5',conv_5_output.shape))

        batch_norm_5_output = self.batch_norm_5(conv_5_output)                         # Computes the batch_normalization of the fifth convolution
        # print('{} : {}'.format('batch_norm_5',batch_norm_5_output.shape))
        
        
        activation_layer_5 = Functional.relu(batch_norm_5_output)                      # Computes the activation of the fifth convolution
        # print('{} : {}'.format('activation_layer_5',activation_layer_5.shape))
        
        max_pool_5_output = self.max_pooling_1(activation_layer_5)                     # Size changes from (250, 6, 6) to ((250, 3, 3))
        # print('{} : {}'.format('max_pool_5',max_pool_5_output.shape))

                                                                                       # Reshape data to input to the input layer of the neural net
                                                                                       # Size changes from (250, 1, 1) to (1, 2250)
                                                                                       # Recall that the -1 infers this dimension from the other given dimension

        fully_connected_1_input = max_pool_5_output.view(-1, int(250 * (self.input_shape[0] / 32) * (self.input_shape[1] / 32)))
        
        # layer 6
        # print('LAYER_6')
        fully_connected_1_output = self.fully_connected_1(fully_connected_1_input)     # Size changes from (1, 250) to (1, 1000) 
        # print('{} : {}'.format('fully_connected_1',fully_connected_1_output.shape))

        batch_norm_6_output = self.batch_norm_6(fully_connected_1_output)              # Computes the batch_normalization of the first_fullyconected
        # print('{} : {}'.format('batch_norm_6',batch_norm_6_output.shape))
        
        activation_layer_6 = Functional.relu(batch_norm_6_output)                      # Computes the activation of the first_fullyconnected
        # print('{} : {}'.format('activation_layer_5',activation_layer_5.shape))

        # layer 7
        dropout_output = self.dropout(activation_layer_6)                              # Computes the droput of the first_fullyconnected with p = 0.5
        # print('{} : {}'.format('dropout',dropout_output.shape))
        
        fully_connected_2_output = self.fully_connected_2(dropout_output)              # Size changes from (1, 1000) to (1, 400)
        # print('{} : {}'.format('fully_connected_2',fully_connected_2_output.shape))
        
        batch_norm_7_output = self.batch_norm_7(fully_connected_2_output)              # Computes the batch_normalization of the second_fullyconected
        # print('{} : {}'.format('batch_norm_7',batch_norm_7_output.shape))
        
        activation_layer_7 = Functional.relu(batch_norm_7_output)                      # Computes the activation of the second_fullyconnected
        # print('{} : {}'.format('activation_layer_7',activation_layer_7.shape))

        # output layer with probas
        # print('OUTPUT_LAYER')
        fully_connected_3_output = self.fully_connected_3(activation_layer_7)          # Size changes from (1, 400) to (1, 2)
        # print('{} : {}'.format('fully_connected_3',fully_connected_3_output.shape))
        
        proba_response = Functional.softmax(fully_connected_3_output, dim = 1)         # Computes the final activation of the third_fullyconnected with SOFTMAX                   
        # print('{} : {}'.format('PROBA_RESPONSE',proba_response.shape))

        return proba_response


# Initialize model
model = Patch_Based_CNN(in_channels = 6, input_shape = (96, 96), num_classes = 2)

a = summary(model, input_size = (6, 96, 96))





class CFG_CNN(Module):

    def __init__(self, input_size):

        self.input_size = input_size
    


