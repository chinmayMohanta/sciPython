# -*- coding: utf-8 -*-

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
#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("poster")

# DATA AND MODEL
# Why we find model from data - understanding date using a distribution

def testUnderstandingData():
    df = pd.read_table("babyboom.dat.txt", header=None, sep='\s+', 
                   names=['24hrtime','sex','weight','minutes'])
    print(df.dtypes)
    print(df.describe()) # Describe the data, basic statistics
    print(df.head())
    
    print(df.minutes.mean())
    
    # Finding CORRELATION
    print(df.corr())
    
    boy = df[df.sex == 1]
    #print(boy)
    
    girl = df[df.sex == 2]
    #print(boy)
    #boy.weight.hist()
    #girl.weight.hist()
    # Side by side bar chat
    #plt.hist(boy.weight)
    #plt.hist(girl.weight)
    
    fig1, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title('boy-weight')
    axarr[0].hist(boy.weight)
    axarr[1].set_title('girl-weight')
    axarr[1].hist(girl.weight)
    
    #####################################################################################
    #######  CHARECTERIZE sample data using a well known probability distribution ####
    #####################################################################################
   
    
    f = lambda x, l: l*np.exp(-l*x)*(x>0) # exponential probability density function
    xpts=np.arange(-2,3,0.1)
    fig2, axar = plt.subplots(4, sharex=True)
    axar[0].plot(xpts,f(xpts, 0.5),'o');  # lamda =0.5
    axar[1].plot(xpts,f(xpts, 1),'o');    # lambda=1
    axar[2].plot(xpts,f(xpts, 2),'o');
    axar[3].plot(xpts,f(xpts, 4),'o');

    # Drawing data (taking sample) from exponential distubution
    from scipy.stats import expon
    fig3, axar = plt.subplots(2)
    xpts=np.arange(-2,3,0.1)
    axar[0].plot(xpts,expon.pdf(xpts, scale=1./2.),'o')
    axar[0].hist(expon.rvs(size=1000, scale=1./2.), normed=True, alpha=0.5, bins=30); # take 1000 samples
    axar[0].set_title("exponential pdf and samples(normalized)")
    
    # Alternatively it can be created as follow
    rv = expon(scale=0.5)
    axar[1].plot(xpts,rv.pdf(xpts),'o')
    axar[1].hist(rv.rvs(size=1000), normed=True, alpha=0.5, bins=30); # take 1000 samples
    axar[1].plot(xpts, rv.cdf(xpts));
    axar[1].set_title("exponential pdf, cdf and samples(normalized)")
    
    
    # understanding data using a well known distribution
    # What does this mean ?? [ POINT ESTIMATION of the scale or rate parameter lambda]
    
    # Considering EXPONENTIAL DISTRIBUTION - Interested to understand distribution of inter-birth time    
# Lets play with our data a bit to understand it:
#The first birth occurred at 0005, and the last birth in the 24-hour period at 2355.
# Thus the 43 inter-birth times happened over a 1430-minute period, 
# giving a theoretical mean of 1430/43 = 33.26 minutes between births.
    

    #Lets plot a histogram of the inter-birth times
    timediffs = df.minutes.diff()[1:]
    fig4, axar = plt.subplots(1)
    axar.hist(timediffs,bins=20)
    
     # The mean of data exponentially distributed with rate parameter lambda is 1/lambda
    # Thus calculating rate parameter LAMBDA from estimated MEAN
    lambda_from_mean = 1./timediffs.mean()
    print(lambda_from_mean, 1./lambda_from_mean)
    
    fig5, axar = plt.subplots(1)
    minutes=np.arange(0, 160, 5)
    rv = expon(scale=1./lambda_from_mean)
    axar.plot(minutes,rv.pdf(minutes),'o')
    axar.hist(timediffs,normed=True,alpha=0.50)
    #axar.hist(normed=True, alpha=0.5);
    #axar.xlabel("minutes");
    axar.set_title("Normalized data and model for estimated $\hat{\lambda}$");
    # What did we just do? We made a 'point estimate' of the scale or rate parameter 
    # as a compression of our data. 
    

    # CONSIDERING Poisson distribution - Interested to understand distribution of number of births per hour
    from scipy.stats import poisson
    k = np.arange(15)
    fig6, axar = plt.subplots(1)
    for i, lambda_ in enumerate([1, 2, 4, 6]):
        axar.plot(k, poisson.pmf(k, lambda_), '-o', label=lambda_)
        
    per_hour = df.minutes // 60
    num_births_per_hour=df.groupby(per_hour).minutes.count()
    print(num_births_per_hour)
    
    
    k = np.arange(5)
    fig7, axar = plt.subplots(1)
    tcount=num_births_per_hour.sum()
    axar.hist(num_births_per_hour, alpha=0.4,  lw=3, normed=True, label="normed hist")
    axar.plot(k, poisson.pmf(k, num_births_per_hour.mean()), '-o',label="poisson")
    axar.set_xlabel("rate")
    axar.set_ylabel("births per hour")
    axar.set_title("Baby births")
    
#MAXIMUM LIKELIHOOD ESTIMATION
# ===============================
#how did we know that the sample mean was a good thing to use?
# One of the techniques used to estimate such parameters in frequentist 
#statistics is maximum likelihood estimation.  
# Key idea : How likely are the observations if the model is true?

# A crucial property is that, for many commonly occurring situations, 
# maximum likelihood parameter estimators have an approximate normal distribution when n is large.
# =================================

# FREQUENTIST STATISTICS
# =============================
#In frequentist statistics, the data we have in hand, is viewed as a sample 
#from a population. So if we want to estimate some parameter of the population,
# like say the mean, we estimate it on the sample.

#This is because we've been given only one sample. 
#Ideally we'd want to see the population, but we have no such luck.
# The parameter estimate is computed by applying 
#an estimator  FF  to some data  DD , so  λ̂ =F(D)λ^=F(D) .

# In FREQUENT STATISTICS The parameter is viewed as fixed and the data as random, 
# which is the exact OPPOSIT of the BAYESIAN
# ==============================


def testSampleDistribution():
    from scipy.stats.distributions import bernoulli
    
    # Simpulating a single binomial experiment => A sequence of bernullian experiments
    # Throwing a coin n times
    def throw_a_coin(n):
        brv = bernoulli(0.5)
        return brv.rvs(size=n)
    
    # Simulating a sequence of binomial experiments of different sizes
    def make_throws(number_of_samples, sample_size):
        start=np.zeros((number_of_samples, sample_size), dtype=int)
        for i in range(number_of_samples):
            start[i,:]=throw_a_coin(sample_size)
        return np.mean(start, axis=1)
        








    
def testFrequentistStatistics():
    pass

#Bootstrap tries to approximate our sampling distribution. 
#If we knew the true parameters of the population, we could generate M fake datasets. 
#Then we could compute the parameter (or another estimator) on each one of these, 
#to get a empirical sampling distribution of the parameter or estimator, 
#and which will give us an idea of how typical our sample is, and thus, 
#how good our parameter estimations from our sample are. (again from murphy)
#
#But we dont have the true parameter. So we generate these samples, using the 
#parameter we calculated. Or, alteratively, we sample with replacement the X from 
#our original sample D, generating many fake datasets, and then compute the distribution 
#on the parameters as before.
#
#We do it here for the mean of the time differences. We could also do it for its inverse,  λ .

def testBootStrap():
    
    # NON-PARAMETRIC boot strap
    # Idea : Resample the data
    
    df = pd.read_table("babyboom.dat.txt", header=None, sep='\s+', 
                   names=['24hrtime','sex','weight','minutes'])
    
    timediffs = df.minutes.diff()[1:]  # This is tricky
    
    
    M_samples=10000
    N_points = timediffs.shape[0]
    print(N_points)
    bs_np = np.random.choice(timediffs, size=(M_samples, N_points))
    print(type(bs_np))
    sd_mean=np.mean(bs_np, axis=1)
    print(sd_mean)
    sd_std=np.std(bs_np, axis=1)
    print(sd_std)
    
    fig, axar = plt.subplots(2)
    axar[0].hist(sd_mean, bins=30, normed=True, alpha=0.5,label="samples");
    axar[0].set_title("Bootstrap Nonr-parametric")

   # PARAMETRIC boot strap
   
   #We get an "estimate" of the parameter from our sample, 
   # and them use the exponential distribution to generate many datasets, 
   # and then fir the parameter on each one of those datasets. 
   # We can then plot the distribution of the mean time-difference.
   
    from scipy.stats import expon
    lambda_from_mean = 1./timediffs.mean()  # Estrimate the parameter (in this case lambda) from the sample
    M_samples=10000
    N_points = timediffs.shape[0]
    rv = expon(scale=1./lambda_from_mean)
    bs_p = rv.rvs(size=(M_samples, N_points))
    sd_mean_p=np.mean(bs_p, axis=1)
    sd_std_p=np.std(bs_p, axis=1)
    axar[1].hist(sd_mean_p, bins=30, normed=True, alpha=0.5,label="samples");
    axar[1].set_title("Bootstrap parametric")
   


    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
def test():
    a1 = np.array([1,2,3])
    b1 = np.array([3,4,5])
    
    print(a1*b1)
    
    df = pd.read_table("babyboom.dat.txt", header=None, sep='\s+', 
                   names=['24hrtime','sex','weight','minutes'])
    per_hour = df.minutes // 60
    print(per_hour)
    print(type(per_hour))
    









if __name__ == '__main__':
    #test()
    #testUnderstandingData()
    testBootStrap()
    

