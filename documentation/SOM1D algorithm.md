# SOM1D Classification Algorithm

## Each step of the Envrironment
1) Update min and max sensation
2) Update prior P(instruction=1)
3) Find conscience-biased marginal BMU
4) Update marginal weight of marginal BMU
5) Update marginal Voronoi regions
6) Update marginal conscience biases
7) Find conscience-biased conditional BMU
8) If instruction = 1:
    8.1) Update conditional weight of conditional BMU
    8.2) Update conditional Voronoi regions
    8.3) Update conditional consciences
9) Find unbiased marginal BMU
10) Find unbiased conditional BMU
11) Calculate score = the posterior P(instruction=1)
12) Calculate prediction = 1 if score > 0.5, = 0 otherwise
13) Calculate pointwise mutual information I(X=x; instruction=1)
14) return score, prediction, pmi 

## Neuronal-Perspective
For each step from the environment, each neuron follows the same algorithm:
1) Process sensor input (x) & instruction input (instr)
    1.1) Calculate its own conscience-biased Euclidean distance between x and marginal weight (E_cb)
    1.2) If E_cb is the lowest (i.e. - E_cb < All E'_cb of neighbors):
        1.2.1) Set activation_state = True
        1.2.2) Update marginal weight
        1.2.3) Find closest_above neighbor weight
        1.2.4) Find closest_below neighbor weight
        1.2.5) Update Voronoi region
    1.3) Update conscience based on activation_state
    1.4) If instr = True: Update conditional weight
    1.5) Update prior for P(instr=1)
    1.6) If activation_state = True: Transmit post-synaptic information i
2) Process neighbor input (i.e. - post-synaptic transmission from BMU)
    2.1) Update neighbor info:
            2.1.1) marginal weight
            2.1.2) conditional weight
            2.1.3) marginal conscience bias
            2.1.4) conditional conscience bias
            2.1.4) Voronoi region
    2.2) Check against marginal&conditional cloesest_above and below, and update if necessary


