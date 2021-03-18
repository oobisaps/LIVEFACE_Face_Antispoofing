class Model(object):

    def __init__(self, data_flow, model,parametrs = None,loader = None):
        
        self.model = model
        self.loader = loader
        self.data_flow = data_flow
        self.parametrs = parametrs

    
    # def train(self):
