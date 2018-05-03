

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


 # Data Engineering, the process of gathering and preparing data for analysis,
    # is a very big part of Data Science.
    # Datasets might not be formatted in the way you need 
    #(e.g. you have categorical features but your algorithm requires numerical features); 
    #or you might need to cross-reference some dataset to another that has a different format; 
    # or you might be dealing with a dataset that contains missing or invalid data.
    # These are just a few examples of why data retrieval and cleaning are so important.

def testDataEngineering():
   pass

    # SPLIT - APPLY - COMBINE design pattern
def testSplitApplycombineDesignPattern():
    #  Split-apply-combine  - Finding the OLAP structure in BI terms
    # splitting the data into groups based on some criteria
    # applying a function to each group independently
    # combining the results into a data structure
    
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
    
    # Q1. Find average rating per book
    # SPLIT - Using a set of DIMENSIONS and a MEASURE
    df_group_by_name = df.rating.groupby(df.name)
    print(df_group_by_name)
    
    # APPLY and COMBINE
    rating_count_per_book = df_group_by_name.count()
    print(rating_count_per_book)
    

######### Web scraping ##################
def testRetrievingInfoFromWeb():
    import requests
    req = requests.get("https://en.wikipedia.org/wiki/Harvard_University")
    print(req)
    print(type(req))
    print(dir(req))
    page = req.text
    print(page)
    
    # Parsing HTML data using BeautifulSoup
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page, 'html.parser')
    print(soup)
    print(soup.title)
    
    
    
    
    


if __name__=='__main__':
    #testSplitApplycombineDesignPattern()
    testRetrievingInfoFromWeb()
    


