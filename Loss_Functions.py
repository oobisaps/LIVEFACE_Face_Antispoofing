import torch 
from torch.nn import *
import torch.nn.functional as Functional




def GramMatrix(input):

    batch_size, h, w, f_map_num = input.size()

    features = input.view(batch_size * h, w * f_map_num) # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())

    return G.div(batch_size * h * w * f_map_num)


class ContentLoss(Module):

    def __init__(self, target):
        
        super(ContentLoss, self).__init__()

        self.target = target.detach()
        self.loss = Functional.mse_loss(self.target, self.target)
    
    def forward(self, input):

        self.loss = Functional.mse_loss(input, self.target)
        return input


class StyleLoss(Module):

    def __init__(self,target_feature):
        super(StyleLoss, self).__init__()
        self.target = GramMatrix(input = target_feature).detach()
        self.loss = Functional.mse_loss(self.target, self.target)

    def forward(self, input):
        G = GramMatrix(input)
        self.loss = Functional.mse_loss(G, self.target)

        return input