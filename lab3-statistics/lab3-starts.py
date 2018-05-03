

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


# SAMPLING AND DISTRIBUTION

# Law of large number

def testLawOfLargeNumber():
    from scipy.stats.distributions import bernoulli
    def throw_a_coin(n):
        brv = bernoulli(0.5)
        return brv.rvs(size=n)
        
    random_flips = throw_a_coin(10000)
    running_means = np.zeros(10000)
    # Step wise increase the length of the sequence
    sequence_lengths = np.arange(1,10001,1)
    for i in sequence_lengths:
        running_means[i-1] = np.mean(random_flips[:i])
        
    fig1, axarr = plt.subplots(1)
    plt.plot(sequence_lengths, running_means);
    plt.xscale('log')
    
def testAverageOfSampleMeans():
    def throw_a_coin(n):
        from scipy.stats.distributions import bernoulli
        brv = bernoulli(0.5)
        return brv.rvs(size=n)
        
    def make_throws(number_of_samples, sample_size):
        start=np.zeros((number_of_samples, sample_size), dtype=int)
        for i in range(number_of_samples):
            start[i,:]=throw_a_coin(sample_size)
        return np.mean(start, axis=1)
        
    throw_results = make_throws(number_of_samples=20, sample_size=10)
    #print(throw_results)
    
    # Let us now do 200 replications, each of which has a sample size of 1000 flips,
    #and store the 200 means for each sample zise from 1 to 1000 in sample_means.
    
    sample_sizes=np.arange(1,1001,1)
    sample_means = [make_throws(number_of_samples=200, sample_size=i) for i in sample_sizes]
    # print(sample_means) // array of array of sample means
    mean_of_sample_means = [np.mean(means) for means in sample_means]
    fig1, axarr = plt.subplots(1)
    axarr.plot(sample_sizes, mean_of_sample_means);


def testSamplingDistribution():
#In data science, we are always interested in understanding the world from incomplete data,
#in other words from a sample or a few samples of a population at large. 
#Our experience with the world tells us that even if we are able to repeat an 
#experiment or process, we will get more or less different answers the next time. 
#If all of the answers were very different each time, we would never be able to make any predictions.
#
#But some kind of ANSWERS DIFFERS ONLY A LITTLE, especially as we get to LARGE sample sizes.
# So the important question then becomes one of the distribution of these quantities from sample
# to sample, also known as a sampling distribution.
#
#Since, in the real world, we see only one sample, this distribution helps us do inference,
# or figure the uncertainty of the estimates of quantities we are interested in.
# If we can somehow cook up samples just somewhat different from the one we were given,
# we can calculate quantities of interest, such as the mean on each one of these samples.
# By seeing how these means vary from one sample to the other, we can say how typical the 
# mean in the sample we were given is, and whats the uncertainty range of this quantity.
# This is why the mean of the sample means is an interesting quantity; it characterizes
# the sampling distribution of the mean, or the distribution of sample means.



    

                            
                    
        
def test():
    val = np.zeros((20, 10), dtype=int)
    print(val)
    
        
        
    
    
if __name__ == '__main__':
    #testLawOfLargeNumber()
    #test()
    testAverageOfSampleMeans()
