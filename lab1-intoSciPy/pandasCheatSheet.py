# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:52:59 2016

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series



# Series object
s1 = Series(range(0,4)) # -> 0, 1, 2, 3
s2 = Series(range(1,5)) # -> 1, 2, 3, 4
s3 = s1 + s2 # -> 1, 3, 5, 7

# Getting data into a data frame
def createDataFrameTest():
    df = DataFrame()
    # Load a DataFrame from a CSV file
    df = pd.read_csv('mlg.csv')
    print(df)
    # Get data from inline CSV text to a DataFrame
    #Load DataFrames from a Microsoft Excel file
    #Load a DataFrame from a MySQL database

    # DataFrame by concatenating time series
    s1 = Series(range(6))
    s2 = s1 * s1
    s2.index = s2.index + 2# misalign indexes
    df = pd.concat([s1, s2], axis=1)
    print(df)

# Inspecting the dataframe
def inspectDataFrameTest():
    
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    df = pd.read_csv('mlg.csv')
    dfInfo = df.info() # index & data types
    print(dfInfo)
    dfh = df.head(5) # get first 5 rows
    print(dfh)
    dft = df.tail(5) # get last i rows
    print(dft)
    dfs = df.describe() # summary stats cols
    print(dfs)
    top_left_corner_df = df.iloc[:4, :4]
    print(top_left_corner_df)

    df = DataFrame()
    # Load a DataFrame from a CSV file 
    df = pd.read_csv('mlg.csv')
    dfT = df.T # transpose rows and cols
    print(dfT)    
    l = df.axes # list row and col indexes
    print(l)
    (r, c) = df.axes # from above
    print((r,c))
    s = df.dtypes # Series column data types
    print(s)
    b = df.empty # True for empty DataFrame
    print(b)
    i = df.ndim # number of axes (it is 2)
    print(i)
    t = df.shape # (row-count, column-count)
    print(t)
    i = df.size # row-count * coldef dataframeNonIndexingAttributeTest():umn-count
    print(i)
    a = df.values # get a numpy array for df (Converting a data frame to 2-dimensional numpy array)
    print(a)
    
def dataFrameUtilityMethodTest():
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    df = pd.read_csv('mlg.csv')
    df = df.copy() # copy a DataFrame
    df = df.rank() # rank each col (default)
    df = df.sort_values(by='maker')
    print(df)
    df = df.sort_values(by=['maker', 'modelyear'])
    print(df)
    df = df.sort_index()
    df = df.astype(int) # type conversion
    print(df)
    
def dataFrameIterationTest():
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    df = pd.read_csv('mlg.csv')
    
    #df.iteritems()# (col-index, Series) pairs
    #df.iterrows() # (row-index, Series) pairs
    
    # example ... iterating over columns
    for (name, series) in df.iteritems():
        print('Col name: ' + str(name))
        print('First value: ' +
              str(series.iat[0]) + '\n')
        
def saveDataFrameTest():
    import os
    print(os.curdir)
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    df = pd.read_csv('mlg.csv')
    
    # Save data frame in CSV format
    #df.to_csv('test_export.csv', encoding='utf-8')
    
    # Saving DataFrames to an Excel Workbook
    # Saving a DataFrame to MySQL
    
    # Saving to Python objects
    d = df.to_dict() # to dictionary
    #print(d)
    str = df.to_string() # to string
    #print(str)
    m = df.as_matrix() # to numpy matrix
    #print(m)
    
def dataFrameMathTest():
    #Note : The methods that return a series default to working on columns.
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    org_df = pd.read_csv('mlg.csv')
    df = org_df.iloc[:,1:7]
    
    resAbs = df.abs() # absolute values
    print(resAbs)
    #resAdd = df.add(o) # add df, Series or value
    #print(resAdd)
    resCount = df.count() # non NA/null values
    print(resCount)
    resCumMax = df.cummax() # (cols default axis)
    print(resCumMax)
    resCumMin = df.cummin() # (cols default axis)
    print(resCumMin)
    resCumSum = df.cumsum() # (cols default axis)
    print(resCumSum)
    resDiff = df.diff() # 1st diff (col def axis)
    print(resDiff)
    resDiv = df.div(12) # div by df, Series, value
    print(resDiv)
    #resDot = df.dot(13) # matrix dot product
    #print(resDot)
    resMax = df.max() # max of axis (col def)
    print(resMax)
    resMean = df.mean() # mean (col default axis)
    print(resMean)
    resMedian = df.median()# median (col default)
    print(resMedian)
    resMin = df.min() # min of axis (col def)
    print(resMin)
    resMul = df.mul(2) # mul by df Series val
    print(resMul)
    resSum = df.sum() # sum axis (cols default)
    print(resSum)
    resWhere = df.where(df > 0.5, other=np.nan)
    print(resWhere)
    
def dataFrameSelectOnColumnTest():
    df = DataFrame()
    # Load a DataFrame from a CSV file    
    df = pd.read_csv('mlg.csv')
    # Note: select takes a Boolean function, for cols: axis=1
    # Note: filter defaults to cols; select defaults to rows
    df_filter_row = df.filter(items=['maker', 'horsepower']) # by col
    print(df_filter_row)
    df_filter_item = df.filter(items=[3,5,8], axis=0) #by row
    print(df_filter_item)
    df_filter_like = df.filter(like='6') # keep x in col
    print(df_filter_like)
    df_filter_regx = df.filter(regex='x') # regex in col
    print(df_filter_regx)
    df_select_lambda = df.select(lambda x: not x%5)# Every 5th rows
    print(df_select_lambda)
    
def columSelectionTest():
    raw_data = {'first_name': ['Jason', 'Molly', np.nan, np.nan, np.nan],
        'nationality': ['USA', 'USA', 'France', 'UK', 'UK'],
        'age': [42, 52, 36, 24, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'nationality', 'age'])
    flter = df['first_name'].notnull() & (df['nationality'] == "USA")
    print(df[flter])
    
    # Select using MASK    
    american = df['nationality'] == "USA"
    
    elderly = df['age'] > 50
    print(elderly)

    print(df[american & elderly])
    print(df[american & elderly].info())

    

def rangeTest():
    arr = np.arange(5)
    print(arr)
    
    
    



if __name__=='__main__':
    #createDataFrameTest()
    #inspectDataFrameTest()
    #dataframeNonIndexingAttributeTest()
    #dataFrameUtilityMethodTest()
    #dataFrameIterationTest()
    #saveDataFrameTest()
    
    #dataFrameMathTest()
    #dataFrameSelectOnColumnTest()
    
    #columSelectionTest()
    
    
    