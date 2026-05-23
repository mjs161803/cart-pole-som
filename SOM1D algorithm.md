# SOM1D Classification Algorithm

## 1. Extract Codebooks and Volumes

As online data is presented to the 1D-SOM in the form of (observation, instruction), all three trained Conscience Self-Organizing Maps (CSOMs) will train their weights such that the winning neuron transmits an estimate of their Voronoi region.  These are used to create 'Codebooks'. You will have:

- **Map $X$ (Marginal):** Codebook $C_X$, Volumes $V_X$
- **Map 1 (Conditional $\text{instruction}=1$):** Codebook $C_1$, Volumes $V_1$
- **Map 0 (Conditional $\text{instruction}=0$):** Codebook $C_0$, Volumes $V_0$

## 2. Estimate the Prior via Least Squares

The prior $\pi = P(\text{instruction}=1)$ is found using $P(x) = \pi P(x|\text{instruction}=1) + (1-\pi) P(x|\text{instruction}=0)$. 

We evaluate the equation over the $N$ prototype vectors of your marginal map ($C_X$) and solve using Ordinary Least Squares (OLS) to get a robust global estimate.

For each prototype vector $c_j \in C_X$:

- The inverse volume in the marginal map: $d_j = \frac{1}{V_{X, j}}$
- Find the BMU for $c_j$ in Map 1 and get its inverse volume: $d_{1,j} = \frac{1}{V_{1, w_1(c_j)}}$
- Find the BMU for $c_j$ in Map 0 and get its inverse volume: $d_{0,j} = \frac{1}{V_{0, w_0(c_j)}}$

Now, solve for the estimated prior $\hat{\pi}$ using the OLS closed-form solution:

$$\hat{\pi} = \frac{\sum_{j=1}^N (d_j - d_{0,j})(d_{1,j} - d_{0,j})}{\sum_{j=1}^N (d_{1,j} - d_{0,j})^2}$$

> **Constraint:** Clip $\hat{\pi}$ to the interval $[0, 1]$ in case of extreme quantization artifacts.

## 3. Evaluate the New Observation

For a new, unclassified input vector $x_{\text{new}}$:

1. Pass $x_{\text{new}}$ into Map 1 to find its BMU. Retrieve that neuron's volume, $V_{1,\text{new}}$.
2. Pass $x_{\text{new}}$ into Map 0 to find its BMU. Retrieve that neuron's volume, $V_{0,\text{new}}$.

> **Note:** You do not need to pass $x_{\text{new}}$ into the marginal Map $X$; it was only needed in Step 2 to establish the prior.

## 4. Apply the Decision Rule

Using Bayes' classifier, predict the class that yields the highest posterior probability. Because the denominator $P(x)$ is the same for both classes, you only need to compare the scaled likelihoods.

Calculate the classification scores:

- **Score for $\text{instruction}=1$:** $S_1 = \dfrac{\hat{\pi}}{V_{1,\text{new}}}$
- **Score for $\text{instruction}=0$:** $S_0 = \dfrac{1 - \hat{\pi}}{V_{0,\text{new}}}$

**Prediction:**

- If $S_1 > S_0$, predict $\text{instruction} = 1$.
- Otherwise, predict $\text{instruction} = 0$.

