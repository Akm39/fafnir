import numpy as np
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
import pickle as pkl

def check(data,labels):
    """
    Feed data in as a numpy array of data points & list of labels (make sure at least 1 param is a constant)
    Returns KPSS Stat of optimal grouping of assets (float), p-value (float) & optimal weights (np.array) stored as a tuple
    """
    X=data[:, :-1]
    y=data[:, -1:]
    coefs=np.linalg.lstsq(X,y,rcond=None)[0]
    resids=(X@coefs)-y
    stat,pval,_,_=kpss(resids,regression='c')
    return stat,pval,np.concatenate([coefs, [[-1]]]),labels

def shift(arr):
    """
    Rolls array over by 1 (make sure to roll over labels as well)
    """
    return np.roll(arr, shift=1, axis=-1)

def list_shift(lst):
    """
    Shifts list over 1 value
    """
    return [lst[-1]] + lst[:-1]

def check_all(data,label):
    """
    Iterates over all permutations
    """
    stats=[]
    pvals=[]
    weights=[]
    labels=[]
    for i in range(data.shape[1]):
        if i==0:
            #no shift
            pass
        else:
            data=shift(data)
            label=list_shift(label)
        try:
            a,b,c,d=check(data,label)
        except:
            a,b,c,d=(10000,-10000,'ERROR',label)
        stats.append(a)
        pvals.append(b)
        weights.append(c)
        labels.append(d)
    return stats,pvals,weights,labels

def stock_list(sector):
    stocks = list(pd.read_excel('Sectors.xlsx')[sector])
    return [x for x in stocks if x==x]

def data_pool(tickers,start='2021-01-01',end='2024-01-01'):
    data_pool = []
    for ticker in tickers:
        data_pool.append(yf.download(ticker,start=start,end=end)['Adj Close'])
    data=pd.concat(data_pool,axis=1)
    data.columns=tickers
    return data

def read_pkl(filepath):
    with open(filepath, 'rb') as file:
        return pkl.load(file)