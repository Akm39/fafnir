import pandas as pd
import numpy as np

def price_hist_to_dict(data):
    # Convert the price history to a dictionary
    return dict(data.iterrows())

def weight_matrix(prices,coef_matrix,margin=0.2):
    # Calculate the weight matrix based on coef matrix & current prices (default is 20% margin)
    prices['CONST']=0
    outs=coef_matrix*prices
    return outs.div(outs.abs().sum(axis=1)*margin, axis=0).drop(columns='CONST')

def all_weight_matrix(data,coef_matrix,margin=0.2):
    # Calculate the weight matrix for all dates
    price_dict=price_hist_to_dict(data)
    weight_matrixes={}
    for date in price_dict.keys():
        weight_matrixes[date]=weight_matrix(price_dict[date],coef_matrix,margin=margin)
    return weight_matrixes