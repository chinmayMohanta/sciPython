# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:53:58 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

# Topics covered:
    
#Regression Models
#Linear, Logistic
#Prediction using linear regression
#Some re-sampling methods 
#Train-Test splits
#Cross Validation


#Regression is a method to model the relationship between a set of 
#independent variables X  (also knowns as explanatory variables, features, 
#predictors) and a dependent variable Y. This method assumes the relationship 
#between each predictor X is linearly related to the dependent variable Y. 
# How do you estimate the coefficients? 
# The method called least squares is one of the most common methods
def linearRegressionTest():
    # Let us consider the Boston housing pricing data
    # This data is already available in sklearn
    
    from sklearn.datasets import load_boston
    boston = load_boston()
    print(boston.keys())
    print(boston.data.shape)
    # Print column names
    print(boston.feature_names)
    # Print description of Boston housing data set
    print(boston.DESCR)
    
    # Let us explore the dataset
    bos = pd.DataFrame(boston.data)
    #print(bos.head())
    # There are no column names in the DataFrame. Let's add those. 
    bos.columns = boston.feature_names
    print(bos.head())
    








if __name__ == '__main__':
    linearRegressionTest()
    




