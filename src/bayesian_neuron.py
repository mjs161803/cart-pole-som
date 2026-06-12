import numpy as np

class BayesianNeuron:
    def __init__(
        self,
        resolution=10,
        lr = 0.001,
        conscience_lr = 0.001,
        conscience_factor = 0.5,
        prior_ema_alpha = 0.001,
        obs_min=None,
        obs_max=None,
        visualize=False,
        viz_update_interval=100,
    ):
        self.resolution = resolution
        self.equiprobability = 1.0 / resolution
        self.lr = lr
        self.conscience_lr = conscience_lr
        self.conscience_factor = conscience_factor
        self.prior_ema_alpha = prior_ema_alpha
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.visualize = visualize
        self.viz_update_interval = viz_update_interval
        rng = np.random.default_rng()
        self.w_m = rng.random(resolution)   # Weights for marginal neuron array
        self.w_m.sort()  
        self.w_c = rng.random(resolution)   # Weights for conditional neuron array
        self.w_c.sort()  
        self.v_m = np.zeros(resolution)     # Voronoi regions for marginal neuron array
        self.v_c = np.zeros(resolution)     # Voronoi regions for conditional neuron array
        self.b_m = np.zeros(resolution)     # Conscience bias for marginal neuron array
        self.b_c = np.zeros(resolution)     # Conscience bias for conditional neuron array
        self.p_m = np.full(resolution, 1.0/resolution)  # Win percentage for marginal neuron array
        self.p_c = np.full(resolution, 1.0/resolution)  # Win percentage for conditional neuron array
        self.y_m = np.zeros(resolution)     # Win state of marginal neuron array (1 if won, 0 otherwise)
        self.y_c = np.zeros(resolution)     # Win state of conditional neuron array (1 if won, 0 otherwise)
        self.prior = 0.0                    # Prior probability P(instruction = 1)
        self.posterior = 0.0                # Posterior probability P(instruction = 1 | observation)
        self.prediction = 0.0               # Final prediction after thresholding posterior

    def step(self, observation, instruction):
        #1) Update self.obs_min and self.obs_max based on the new observation
        self.update_minmax(observation)

        #2) Update self.prior using an EMA of the instruction
        self.update_prior(instruction)

        #3) Find BMU index of marginal neuron array using conscience-biased distance
        bmu_cb_m = self.find_cbmu_marginal(observation)

        #4) Update conscience bias of marginal neurons
        self.update_conscience_marginal()

        #5) Update weight of winning marginal neuron and resort (w_m, b_m, p_m) arrays
        self.update_weight_marginal(bmu_cb_m, observation)

        #6) Update Voronoi region of winning marginal neuron and nearest neighbors
        self.update_voronoi_marginal()

        #7) If instruction == 1, Update conditional neuron array using conscience-biased distance, weight update, and Voronoi update:
        if instruction == 1:
            #7.1) Find BMU index of conditional neuron array using conscience-biased distance
            bmu_cb_c = self.find_cbmu_conditional(observation)
            #7.2) Update conscience bias of conditional neurons
            self.update_conscience_conditional()
            #7.3) Update weight of winning conditional neuron and resort (w_c, b_c, p_c) arrays
            self.update_weight_conditional(bmu_cb_c, observation)
            #7.4) Update Voronoi region of winning conditional neuron and nearest neighbors
            self.update_voronoi_conditional()


        #8) Find BMU index of marginal neuron array using standard distance
        bmu_m = self.find_bmu_marginal(observation)

        #9) Find BMU index of conditional neuron array using standard distance
        bmu_c = self.find_bmu_conditional(observation)

        #10) Calculate posterior probability P(instruction = 1 | observation) using Bayes' theorem
        self.calc_posterior(bmu_m, bmu_c)

        #11) Calculate final prediction by thresholding posterior at 0.5
        self.calc_prediction()

        #12) Calculate pointwise mutual information between observation and instruction using marginal and conditional BMUs
        pmi = self.calc_pmi(bmu_m, bmu_c)
    
        return self.prediction, self.posterior, pmi
    
    def update_minmax(self, observation):
        self.obs_min = min(self.obs_min, observation) if self.obs_min is not None else observation
        self.obs_max = max(self.obs_max, observation) if self.obs_max is not None else observation
        return
    
    def update_prior(self, instruction):
        self.prior = self.prior_ema_alpha * instruction + (1 - self.prior_ema_alpha) * self.prior
        return
    
    def find_cbmu_marginal(self, observation):
        d = np.abs(self.w_m - observation) - self.b_m  # Conscience-biased distance
        cbmu_index = np.argmin(d)
        # Update win states for marginal neurons
        self.y_m = np.zeros(self.resolution)
        self.y_m[cbmu_index] = 1
        return cbmu_index
    
    def update_conscience_marginal(self):
        #1) Update marginal neurons' win percentages using current win states
        self.p_m = self.p_m + self.conscience_lr* (self.y_m - self.p_m)
        self.b_m = self.conscience_factor * (self.equiprobability - self.p_m)
        return    
    
    def update_weight_marginal(self, bmu_index, observation):
        #1) Update weight of winning marginal neuron using learning rate
        self.w_m[bmu_index] += self.lr * (observation - self.w_m[bmu_index])
        #2) Resort (w_m, b_m, p_m) arrays based on updated weights while keeping them aligned
        sorted_indices = np.argsort(self.w_m)
        self.w_m = self.w_m[sorted_indices]
        self.b_m = self.b_m[sorted_indices]
        self.p_m = self.p_m[sorted_indices]        
        return
    
    def update_voronoi_marginal(self):
        # For each marginal neuron, calculate its Voronoi region as the midpoint between its weight and the weights of its neighbors
        #   If it's the first neuron, its Voronoi region extends from self.min to the midpoint with the next neuron
        #   If it's the last neuron, its Voronoi region extends from the midpoint with the previous neuron to self.max
        #   Otherwise, its Voronoi region extends from the midpoint with the previous neuron to the midpoint with the next neuron
        for i in range(self.resolution):
            if i == 0:
                self.v_m[i] = (self.w_m[0] + self.w_m[1]) / 2.0 - self.obs_min
            elif i == self.resolution - 1:
                self.v_m[i] = self.obs_max - (self.w_m[-2] + self.w_m[-1]) / 2.0
            else:
                self.v_m[i] = (self.w_m[i+1] - self.w_m[i-1]) / 2.0
        return

    
    def find_cbmu_conditional(self, observation):
        d = np.abs(self.w_c - observation) - self.b_c  # Conscience-biased distance
        cbmu_index = np.argmin(d)
        # Update win states for conditional neurons
        self.y_c = np.zeros(self.resolution)
        self.y_c[cbmu_index] = 1
        return cbmu_index
    
    def update_conscience_conditional(self):
        #1) Update conditional neurons' win percentages using current win states
        self.p_c = self.p_c + self.conscience_lr* (self.y_c - self.p_c)
        self.b_c = self.conscience_factor * (self.equiprobability - self.p_c)
        return    
    
    def update_weight_conditional(self, bmu_index, observation):
        #1) Update weight of winning conditional neuron using learning rate
        self.w_c[bmu_index] += self.lr * (observation - self.w_c[bmu_index])
        #2) Resort (w_c, b_c, p_c) arrays based on updated weights while keeping them aligned
        sorted_indices = np.argsort(self.w_c)
        self.w_c = self.w_c[sorted_indices]
        self.b_c = self.b_c[sorted_indices]
        self.p_c = self.p_c[sorted_indices]        
        return
    
    def update_voronoi_conditional(self):
        # For each conditional neuron, calculate its Voronoi region as the midpoint between its weight and the weights of its neighbors
        #   If it's the first neuron, its Voronoi region extends from self.min to the midpoint with the next neuron
        #   If it's the last neuron, its Voronoi region extends from the midpoint with the previous neuron to self.max
        #   Otherwise, its Voronoi region extends from the midpoint with the previous neuron to the midpoint with the next neuron
        for i in range(self.resolution):
            if i == 0:
                self.v_c[i] = (self.w_c[0] + self.w_c[1]) / 2.0 - self.obs_min
            elif i == self.resolution - 1:
                self.v_c[i] = self.obs_max - (self.w_c[-2] + self.w_c[-1]) / 2.0
            else:
                self.v_c[i] = (self.w_c[i+1] - self.w_c[i-1]) / 2.0
        return
    
    def find_bmu_marginal(self, observation):
        idx = np.searchsorted(self.w_m, observation)
        if idx == 0:
            bmu_index = 0
        elif idx == self.resolution:
            bmu_index = self.resolution - 1
        else:
            if abs(observation - self.w_m[idx-1]) <= abs(observation - self.w_m[idx]):
                bmu_index = idx - 1
            else:
                bmu_index = idx
        return bmu_index
    
    def find_bmu_conditional(self, observation):
        idx = np.searchsorted(self.w_c, observation)
        if idx == 0:
            bmu_index = 0
        elif idx == self.resolution:
            bmu_index = self.resolution - 1
        else:            
            if abs(observation - self.w_c[idx-1]) <= abs(observation - self.w_c[idx]):
                bmu_index = idx - 1
            else:
                bmu_index = idx
        return bmu_index
    
    def calc_posterior(self, bmu_m, bmu_c):
        if self.v_c[bmu_c] == 0.0:
            self.posterior = 0.0
        else:
            self.posterior = self.prior * (self.v_m[bmu_m] / self.v_c[bmu_c])  # P(instruction=1|observation) = P(instruction=1) * P(observation|instruction=1) / P(observation)
        return
    
    def calc_prediction(self):
        self.prediction = 1 if self.posterior >= 0.5 else 0
        return
    
    def calc_pmi(self, bmu_m, bmu_c):
        if self.v_c[bmu_c] == 0.0 or self.v_m[bmu_m] == 0.0:
            return 0.0
        pmi = np.log2(self.v_m[bmu_m]/self.v_c[bmu_c])  # PMI(observation; instruction) = log2(P(observation|instruction=1) / P(observation))
        return pmi
    


                