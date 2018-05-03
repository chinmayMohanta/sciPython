
# PROBABILITY AND DISTRIBUTION

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

# Probability of model

# SIMULATING THE RESULT OF A MODEL

# Simulating tossing of a coin
def throw_a_coin(N):
    
    # the function np.random.choice, which will with equal probability for all items pick an item from a list
    return np.random.choice(['H','T'], size=N)
    
def testFrequentisProbabilityOfHead():
    
    trials=[10, 20, 50, 70, 100, 200, 500, 800, 1000, 2000, 5000, 7000, 10000]
    fig1, axarr = plt.subplots(1)
    plt.plot(trials, [np.sum(throw_a_coin(j)=='H')/np.float(j) for j in trials], 'o-', alpha=0.6);
    # OBSERVATION
    #Thus, the true odds fluctuate about their long-run value of 0.5, 
    #in accordance with the model of a fair coin 
    #(which we encoded in our simulation by having np.random.choice choose 
    #between two possibilities with equal probability), with the fluctuations 
    #becoming much smaller as the number of trials increases. These fluctations 
    #are what give rise to probability distributions.
    
    # Each finite length run is called a SAMPLE, which has been obtained from 
    #the generative model of our fair coin. Its called GENERATIVE as we can 
    #use the model to generate, using simulation, a set of samples we can play 
    #with to understand a model.
    
# RANDOM VARIABLE
def testBernoulli():
    from scipy.stats import bernoulli
    #bernoulli random variable
    brv=bernoulli(p=0.3)
    sample = brv.rvs(size=20)
    print(sample)
    
def testUniform():
    sample = np.random.uniform(0,1)
    print(sample)
    
def testBinomial():
    from scipy.stats import binom
    fig1, axarr = plt.subplots(1)
    k = np.arange(0, 200)
    p = 0.1
    rv = binom(200, p)
    
def testRandomNumber():
    #The various ways to get random numbers
    # np.random.choice chooses items randomly from an array, with or without replacement
    # np.random.random gives us uniform randoms on [0.0,1.0)
    # np.random.randint gives us random integers in some range
    # np.random.randn gives us random samples from a Normal distribution, which we talk about later.
    # scipy.stats.distrib gives us stuff from a distribution. Here distrib 
    # could be binom for example, as above. distrib.pdf or distrib.pmf give us the 
    #density or mass function, while cdf gives us the cumulaive distribution function. 
    #Just using distrib as a function with its params creates a random 
    #variable generating object, from which random variables can be generated 
    #in the form distrib(params).rvs(size).

    

    



if __name__=='__main__':
    #testFrequentisProbabilityOfHead()
    #testBernoulli()
    testUniform()
    