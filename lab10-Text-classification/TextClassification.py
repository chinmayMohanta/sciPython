

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")


# Understanding Bag Of Word (BoW) model
def bowTest():
    # Using the Rotten Tomotto data set
    critics = pd.read_csv('./critics.csv')
    
    # DATA CLEANING
    #let's drop rows with missing quotes
    critics = critics[~critics.quote.isnull()]
    #print(critics.head())
    
    # Explore
    
    n_reviews = len(critics)
    n_movies = critics.rtid.unique().size
    n_critics = critics.critic.unique().size

    print("Number of reviews: %i",n_reviews)
    print("Number of critics: %i",n_critics)
    print("Number of movies:  %i",n_movies)
    
    df = critics.copy()
    df['fresh'] = df.fresh == 'fresh'
    #print(df.head())
    grp = df.groupby('critic')
    
    counts = grp.critic.count()  # number of reviews by each critic
    #print(grp.groups)
    means = grp.fresh.mean()     # average freshness for each critic
    
    #means[counts > 100].hist(bins=10, edgecolor='w', lw=1)
    #plt.xlabel("Average rating per critic")
    #plt.ylabel("N")
    #plt.yticks([0, 2, 4, 6, 8, 10]);
    
    # A matrix (or transpose of it) representing column as the documents and 
    # rows as the term (feature) frequencies
    def termDocumentMatrix():
        from sklearn.feature_extraction.text import CountVectorizer
        text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']
        print("Original text is\n", '\n'.join(text))
        
        vectorizer = CountVectorizer(min_df=0)
        # call `fit` to build the vocabulary
        vectorizer.fit(text)
        # call `transform` to convert text to a bag of words
        x = vectorizer.transform(text)
        # CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
        # convert back to a "normal" numpy array
        x = x.toarray()
        print("Transformed text vector is \n", x)
        # `get_feature_names` tracks which word is associated with each column of the transformed x

        print("Words for each feature:")
        print(vectorizer.get_feature_names())
        
    termDocumentMatrix()
    
        
    
                
               
               
               
               
    




if __name__ == '__main__':
    bowTest()

