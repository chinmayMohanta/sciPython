# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
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
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

from pandas import DataFrame





#The various ways to get random numbers
#1.np.random.choice chooses items randomly from an array, with or without replacement
#2.np.random.random gives us uniform randoms on [0.0,1.0)
#3.np.random.randint gives us random integers in some range
#4.np.random.randn gives us random samples from a Normal distribution, which we talk about later.
#5.scipy.stats.distrib gives us stuff from a distribution. Here distrib could be binom for example, 
#as above. distrib.pdf or distrib.pmf give us the density or mass function, 
#while cdf gives us the cumulaive distribution function. Just using distrib as a 
#
#function with its params creates a random variable generating object, 
#from which random variables can be generated in the form distrib(params).rvs(size).

def probablityAndDistributionTest():
    
    # Simulating the reslut of a model
    # 1. Model : Tossing a coin
    def modelTossingAcoing():
        # Simulating throwing a coin
        def throw_a_coin(N):
            return np.random.choice(['H','T'], size=N)
        # Simulating results :: Small number of trials
        
        throws=throw_a_coin(40)
        print("Throws:"," ".join(throws))
        print("Number of Heads:", np.sum(throws=='H'))
        print("p1 = Number of Heads/Total Throws:", np.sum(throws=='H')/40.)
    
        # Simulating results :: Large number of tials
        throws=throw_a_coin(10000)
        print("Throws:"," ".join(throws))
        print("Number of Heads:", np.sum(throws=='H'))
        print("p1 = Number of Heads/Total Throws:", np.sum(throws=='H')/10000.)
    
        # The larger number of trials we do, the closer we seem to get to half 
        #the tosses showing up heads. Lets see this more systematically
    
        trials=[10, 20, 50, 70, 100, 200, 500, 800, 1000, 2000, 5000, 7000, 10000]
        plt.plot(trials, [np.sum(throw_a_coin(j)=='H')/np.float(j) for j in trials], 'o-', alpha=0.6);
        plt.xscale("log")
        plt.axhline(0.5, 0, 1, color='r');
        plt.xlabel('number of trials');
        plt.ylabel('probability of heads from simulation');
        plt.title('frequentist probability of heads');

    #modelTossingAcoing()
    
    # CONLUSION : Each finite length run is called a SAMPLE, 
    #which has been obtained from the GENERATIVE model of our fair coin. 
    #Its called generative as we can use the model to generate, using 
    #simulation, a set of samples we can play with to understand a model.
    
    def modelElection():
        # Let us load the election data into PredictWise
        # PredictWise aggregated polling data and, for each state, estimated 
        #the probability that the Obama or Romney would win. Here are those estimated probabilities
        
        predictwise = pd.read_csv('predictwise.csv').set_index('States')
        #print(predictwise.head())
        
        #  we will assume that the outcome in each state is the result of an 
        #independent coin flip whose probability of coming up Obama is given 
        #by the Predictwise state-wise win probabilities. Lets write a function 
        #simulate_election that uses this predictive model to simulate the 
        #outcome of the election given a table of probabilities.
        
        # The model created by combining the probabilities we obtained from 
        #Predictwise with the simulation of a biased coin flip corresponding 
        #to the win probability in each states leads us to obtain a histogram 
        #of election outcomes. We are plotting the probabilities of a prediction, 
        #so we call this distribution over outcomes the "PREDICTIVE PROBABILITY DISTRIBUTION". 
        #Simulating from our model and plotting a histogram allows us to visualize 
        #this predictive distribution. In general, such a set of probabilities is 
        #called a probability distribution or probability mass function. 
        
        def simulate_election(model, n_sim):
            simulations = np.random.uniform(size=(51, n_sim))
            print(len(simulations))
            print(len(simulations[0]))
           
            print(simulations)
            # We run 10,000 simulations. In each one of these simulations, 
            #we toss 51 biased coins, and assign the vote to obama if the 
            #output of np.random.uniform is less than the probablity of an obama win.
            
            obama_votes = (simulations < model.Obama.values.reshape(-1, 1)) * model.Votes.values.reshape(-1, 1)
            #summing over rows gives the total electoral votes for each simulation
            return obama_votes.sum(axis=0)
        result = simulate_election(predictwise, 10000)
        print((result >= 269).sum())
        print(result)
        
        
        # Displaying the prediction
        def plot_simulation(simulation):
            plt.hist(simulation, bins=np.arange(200, 538, 1),label='simulations', align='left', normed=True)
            plt.axvline(332, 0, .5, color='r', label='Actual Outcome')
            plt.axvline(269, 0, .5, color='k', label='Victory Threshold')
            p05 = np.percentile(simulation, 5.)
            p95 = np.percentile(simulation, 95.)
            iq = int(p95 - p05)
            pwin = ((simulation >= 269).mean() * 100)
            plt.title("Chance of Obama Victory: %0.2f%%, Spread: %d votes" % (pwin, iq))
            plt.legend(frameon=False, loc='upper left')
            plt.xlabel("Obama Electoral College Votes")
            plt.ylabel("Probability")
            sns.despine()
        plot_simulation(result)
        
    #modelElection();
    
    # Random variables provide the link from events and sample spaces to data, 
    # and it is their PROBABILITY DISTRIBUTION that we are interested in.
    # REf :https://gist.github.com/mattions/6113437/
    def randomVariableTest():
        
        # Bernoulli random varabl
        # X∼Bernoulli(p)
        def bernoulliRandomVariableTest():
            from scipy.stats import bernoulli
            #bernoulli random variable
            
            brv=bernoulli(p=0.2)
            arr=brv.rvs(size=20)
            print(arr)
            
        #BernoulliRandomVariableTest()    
          
        # Uniform random variable
        # It gives you a random number between 0 and 1, uniformly. 
        #In other words, the number is equally likely to be between 
        #0 and 0.1, 0.1 and 0.2, and so on. This is a very intuitive idea, 
        #but it is formalized by the notion of the Uniform Distribution.
        
        # X∼Uniform([0,1)
        def bernoulliRandomVariableTest():
            simulations = np.random.uniform(size=(51, 10))
            print(simulations)
            
        bernoulliRandomVariableTest()
        
        
        # This is an empirical Probability Mass Function or 
        #Probability Density Function. The word density is strictly used when 
        #the random variable X takes on continuous values, as in the 
        #uniform distribution, rather than discrete values such as here, 
        #but we'll abuse the language and use the word probability distribution in both cases.
            
       #def cdfTest():
           
        #   CDF = lambda x: np.float(np.sum(result < x))/result.shape[0] for votes in [200, 300, 320, 340, 360, 400, 500]:
         #      print "Obama Win CDF at votes=", votes, " is ", CDF(votes)
           
        
         # Binomial random variable
         #Let us consider a population of coinflips, n of them to be precise, x 1 ,x 2 ,...,x n x1,x2,...,xn.
         # The distribution of coin flips is the binomial distribution. By this we mean that each coin flip 
         #represents a bernoulli random variable (or comes from a bernoulli distribution) with mean p=0.5
         # p=0.5.
         # At this point, you might want to ask the question, what is the probability of obtaining k 
         # K  heads in n n  flips of the coin. We have seen this before, when we flipped 2 coins. 
         #What happens when when we flip 3?

        def binomialRandomVariableTest():
            
            colors = mpl.rcParams['axes.color_cycle']
            from scipy.stats import binom
            plt.figure(figsize=(12,6))
            k = np.arange(0, 200)
            for p, color in zip([0.1, 0.3, 0.7, 0.7, 0.9], colors):
                rv = binom(200, p)
                plt.plot(k, rv.pmf(k), '.', lw=2, color=color, label=p)
                plt.fill_between(k, rv.pmf(k), color=color, alpha=0.5)
            q=plt.legend()
            plt.title("Binomial distribution")
            plt.tight_layout()
            q=plt.ylabel("PDF at $k$")
            q=plt.xlabel("$k$")
            
       #binomialRandomVariableTest()
        
        def exponentialDistributionTest():
            
            f = lambda x, l: l*np.exp(-l*x)*(x>0)
            xpts=np.arange(-2,3,0.1)
            plt.plot(xpts,f(xpts, 2),'o');
        plt.xlabel("x")
        plt.ylabel("exponential pdf")
       
        #exponentialDistributionTest()

            
    #randomVariableTest()
    
    def statisticsTest():
    
        #Let x1 ,x2 ,...,xn be a sequence of independent, identically-distributed (IID) 
        #random variables. Suppose that X  has the finite mean μ
        # then the average of the first n of them:
        def lawOfLargeNumberTest():
            
            def throw_a_coin(n):                
                from scipy.stats.distributions import bernoulli
                brv = bernoulli(0.5)
                return brv.rvs(size=n)
            random_flips = throw_a_coin(10000)
            running_means = np.zeros(10000)
            sequence_lengths = np.arange(1,10001,1)
            for i in sequence_lengths:
                running_means[i-1] = np.mean(random_flips[:i])
            plt.plot(sequence_lengths, running_means)
            plt.xscale('log')
        
        #lawOfLargeNumberTest()
        
        # What is a Sample ??
        #Let usestablish some terminology at first. What we did there was 
        #to do a large set of replications M, in each of which we did many 
        #coin flips N. We'll call the result of each coin flip an OBSERVATION, 
        #and a single replication a SAMPLE of observations. Thus the number of 
        #samples is M, and the sample size is N. These samples have been chosen 
        #from a population of size n >> N.
        
        def sampleTest():
            
            def throw_a_coin(n):                
                from scipy.stats.distributions import bernoulli
                brv = bernoulli(0.5)
                return brv.rvs(size=n)
            def make_throws(number_of_samples, sample_size):
                start=np.zeros((number_of_samples, sample_size), dtype=int)
                for i in range(number_of_samples):
                    start[i,:]=throw_a_coin(sample_size)
                return np.mean(start, axis=1)
            arr = make_throws(number_of_samples=20, sample_size=10)
            #print(arr)
            sample_sizes=np.arange(1,101,1)
            sample_means = [make_throws(number_of_samples=200, sample_size=i) for i in sample_sizes]
                            
            print(sample_means)
            mean_of_sample_means = [np.mean(means) for means in sample_means]
            plt.plot(sample_sizes, mean_of_sample_means)
            plt.ylim([0.480,0.520]);

        #sampleTest()
    
        def sampleDistributionTest():
            pass
        
        #The sampling distribution of the mean itself has a mean μ and variance σ2N.
        # This distribution is called the Gaussian or Normal Distribution, 
        #and is probably the most important distribution in all of statistics.
        def GaussianDistributionTest():
            norm =  sp.stats.norm
            x = np.linspace(-5,5, num=200)
            fig = plt.figure(figsize=(12,6))
            colors = mpl.rcParams['axes.color_cycle']
            for mu, sigma, c in zip([0.5]*3, [0.2, 0.5, 0.8], colors):
                plt.plot(x,norm.pdf(x, mu, sigma), lw=2,c=c, label = r"$\mu = {0:.1f}, \sigma={1:.1f}$".format(mu, sigma))
                plt.fill_between(x, norm.pdf(x, mu, sigma), color=c, alpha = .4)
    
    
            plt.xlim([-5,5])
            plt.legend(loc=0)
            plt.ylabel("PDF at $x$")
            plt.xlabel("$x$") 
        GaussianDistributionTest()
            
    
        
    statisticsTest()
        

# Let us understand data
def dataAndModelTest():
    # A Simple Dataset for Demonstrating Common Distributions
    #Forty-four babies -- a new record -- were born in one 24-hour period at 
    #the Mater Mothers' Hospital in Brisbane, Queensland, Australia, on December 
    #18, 1997. For each of the 44 babies, The Sunday Mail recorded the time of 
    #birth, the sex of the child, and the birth weight in grams. Also included is 
    #the number of minutes since midnight for each birth.
    
    # Read the data from file and load it into the data frame
    df = DataFrame()
    # Load a DataFrame from a CSV file
    df = pd.read_table("babyboom.dat.txt", header=None, sep='\s+', 
                   names=['24hrtime','sex','weight','minutes'])
    print(df.head())
    
    # Find the average mean time of birth from mid-night
    print(np.mean(df.minutes))
    
    # Finding the correlation
    print(df.corr())
    #g = sns.FacetGrid(col="sex", data=df, size=8)
    #g.map(plt.hist, "weight")
    
    # Choosing a model
    # The arrival of babies can be modelled using a posson process with 
    # inter-arrival time as exponential distribution
    
    # Reminder :: The exponential distribution occurs naturally when 
    #describing the lengths of the inter-arrival times in a homogeneous Poisson process.
    
    # In our example above, we have the arrival times of the babies. 
    #There is no reason to expect any specific clustering in time, 
    #so one could think of modelling the arrival of the babies via a poisson process.
    #Furthermore, the Poisson distribution can be used to model the number of 
    #births each hour over the 24-hour period.
    
    #The first birth occurred at 0005, and the last birth in the 24-hour period at 2355. 
    #Thus the 43 inter-birth times happened over a 1430-minute period, giving a 
    #theoretical mean of 1430/43 = 33.26 minutes between births.
    #Lets plot a histogram of the inter-birth times
    
    timediffs = df.minutes.diff()[1:]
    #print(timediffs)
    timediffs.hist(bins=20);


    
    




if __name__ == '__main__':
    #probablityAndDistributionTest()
    dataAndModelTest()
    