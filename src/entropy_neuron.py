import numpy as np

class EntropyNeuron:
    def __init__(self, 
                 learning_rate=0.001, 
                 conscience_learning_rate=0.001, 
                 conscience_factor=0.5,
                 resolution = 2.0,
                 ):
        self.resolution = resolution
        self.equiprobability = 1.0 / resolution
        self.lr = learning_rate
        self.conscience_lr = conscience_learning_rate
        self.conscience_factor = conscience_factor
        self.w = [0.0, 1.0]                             # Weights
        self.b = np.zeros(resolution)                   # Conscience bias
        self.p = np.full(resolution, 1.0/resolution)    # Win percentage
        self.y = np.zeros(resolution)                   # Win state 

    def step(self, observation):
        # neuron at index 0 is for observation 0, neuron at index 1 is for observation 1
        bmu = self.find_cbmu(observation)
        # Update conscience
        self.update_conscience()
        return bmu

    def find_cbmu(self, observation):
        d = np.abs(self.w - observation) - self.b  # Conscience-biased distance
        cbmu_index = np.argmin(d)
        # Update win states for marginal neurons
        self.y = np.zeros(self.resolution)
        self.y[cbmu_index] = 1
        return cbmu_index
    
    def update_conscience(self):
        self.p = self.p + self.conscience_lr* (self.y - self.p)
        self.b = self.conscience_factor * (self.equiprobability - self.p)
        return 




        

    