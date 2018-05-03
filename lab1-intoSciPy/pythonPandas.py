

#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
#import seaborn as sns #sets up styles and gives us more plotting options

def testUnderstandingData():
    #df = DataFrame()
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)
    print(df.head(1))
    
    df.info()  # Find information about columns and their data types
    print(df.shape)
    
    # REMINDER :: Internally data frame stores each column as a series object
    print(type(df.rating)) # Find the type of a column; it is alwyas of type SERIESE
    
def testQuery():
    #df = DataFrame()
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)
    
    # Find the movies with rating less than 3
    print(type(df.rating < 3)) # type is of type SERIESE; such a series is known as MASK
    
    print(df[df.rating < 3])
    print(df[df.rating < 3].filter(items=['name','rating'],axis=1))
    
    # Filtering usig query
    print(df.query('rating > 4.5'))
    
def testDataCleaning():
    
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)

    # Clean the data
    # Remove the incomplete data (Data with missing  values)
    #  - Remove the ratings with year missing
        
    df = df[df.year.notnull()]  # using MASKING
    print(df.shape)
    
    print(df.dtypes)
    # Modify the data type of review_count,rating_count from object(string) to int and year from float to int
    df.rating_count =df.rating_count.astype(int)
    df.review_count =df.review_count.astype(int)
    df.year =df.year.astype(int) 
    print(df.dtypes)
    
    
    
def testVisualizationHisto():
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)

    # Clean the data
    # Remove the incomplete data (Data with missing  values)
    #  - Remove the ratings with year missing
        
    df = df[df.year.notnull()]  # using MASKING
    print(df.shape)
    
    print(df.dtypes)
    # Modify the data type of review_count,rating_count from object(string) to int and year from float to int
    df.rating_count =df.rating_count.astype(int)
    df.review_count =df.review_count.astype(int)
    df.year =df.year.astype(int) 
    print(df.dtypes)
    
    # Histogram
    #df.rating.hist()
    df.rating.hist(bins=30, alpha=0.4);
    # Very important - REGULARIZATION of data
    #One can see the SPARSENESS of review counts.
    #This will be important when we learn about recommendations: 
    #we'll have to regularize our models to deal with it.
    

def testVisualizationRescaling():
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)

    # Clean the data
    # Remove the incomplete data (Data with missing  values)
    #  - Remove the ratings with year missing
        
    df = df[df.year.notnull()]  # using MASKING
    print(df.shape)
    
    print(df.dtypes)
    # Modify the data type of review_count,rating_count from object(string) to int and year from float to int
    df.rating_count =df.rating_count.astype(int)
    df.review_count =df.review_count.astype(int)
    df.year =df.year.astype(int) 
    print(df.dtypes)
    df.review_count.hist(bins=np.arange(0, 40000, 400))
    # rescaling : The above plot can be rescale to logarithemic for better visualization
    df.review_count.hist(bins=100)
    plt.xscale("log");
    

    

def testVisualizationScatter():
    df=pd.read_csv("all.csv", header=None,names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],)

    # Clean the data
    # Remove the incomplete data (Data with missing  values)
    #  - Remove the ratings with year missing
        
    df = df[df.year.notnull()]  # using MASKING
    print(df.shape)
    
    print(df.dtypes)
    # Modify the data type of review_count,rating_count from object(string) to int and year from float to int
    df.rating_count =df.rating_count.astype(int)
    df.review_count =df.review_count.astype(int)
    df.year =df.year.astype(int) 
    print(df.dtypes)
    # Scatter plot
    # Rating vs year
    # By setting the alpha transparency low we can show how the density
    # of highly rated books on goodreads has changed.
    plt.scatter(df.year, df.rating, lw=0, alpha=.08) # To understand the parameter alpha = http://matthiaseisen.com/pp/patterns/p0174/
    plt.xlim([1900,2010])
    plt.xlabel("Year")
    plt.ylabel("Rating")
    
def testVectorization():
    # Numpy array and panda Series (based on Numpy array) support vector operations.
    # In other words; we cand add to Numpy array straight away
    vec1 = np.array([4,5,6])
    print(type(vec1))
    vec2 = np.array([-1,4,3])
    res = vec1 + vec2
    print(res)
    
    
if __name__ == '__main__':
    #testUnderstandingData()
    #testQuery()
    #testDataCleaning()
    #testVisualizationHisto()
    #testVisualizationRescaling()
    #testVisualizationScatter()
    testVectorization()
    
    
    