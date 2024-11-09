from itertools import combinations
from utils import check_all
import pickle as pkl

def data_fixer(data):
    """Adds a constant to the data"""
    data['CONST']=1
    return data

def combo_maker(stocks,size):
    """
    Makes list of lists of all combinations in list given size of combination (nCr)
    """
    combinations_list = list(combinations(stocks, size))
    return [list(x) for x in combinations_list]

def combo_prepper(combo,data):
    """
    Prepares a single combination for check_all
    """
    combo=combo+['CONST']
    return data[combo].values, combo

def iter_all(data,size,override=False):
    """
    Iterates over all combinations in dataset of 
    """
    stocks=list(data.columns)
    stocks=[x for x in stocks if x!='CONST']
    data_t=data_fixer(data)
    combos=combo_maker(stocks,size)
    if ((len(combos)*len(combos[0]))>3e7) & (not override):
        return "MORE THAN 30 MILLION COMBINATIONS, ENTER override = True AS A PARAMETER TO RUN"
    stats=[]
    pvals=[]
    weights=[]
    order=[]
    for combo in combos:
        subset,labels=combo_prepper(combo,data_t)
        stat,pval,weight,label=check_all(subset,labels)
        stats+=stat
        pvals+=pval
        weights+=weight
        order+=label
    return stats,pvals,weights,order

def iter_dump(data,size,title,override=False):
    """
    Wrapper for iter_all that dumps a dictionary with stats, pvals, weights, labels
    """
    stats,pvals,weights,labels=iter_all(data,size,override)
    wrapper={}
    wrapper['stats']=stats
    wrapper['pvals']=pvals
    wrapper['weights']=weights
    wrapper['labels']=labels
    with open(title+'.pkl', 'wb') as file:
        pkl.dump(wrapper, file)

def iter_load(title):
    """
    Loads the iter_dump
    """
    with open(title+'.pkl', 'rb') as file:
        wrapper = pkl.load(file)
    return wrapper['stats'],wrapper['pvals'],wrapper['weights'],wrapper['labels']

def iter_loop(data,size_range,override=False):
    """
    Loops over list of ints of size range
    """
    stats=[]
    pvals=[]
    weights=[]
    labels=[]
    for size in size_range:
        a,b,c,d=iter_all(data,size,override)
        stats+=a
        pvals+=b
        weights+=c
        labels+=d
    return stats,pvals,weights,labels

def iter_loop_dump(data,size_range,title,override=False):
    """
    Wrapper for iter_all that dumps a dictionary with stats, pvals, weights, labels
    """
    stats,pvals,weights,labels=iter_loop(data,size_range,override)
    wrapper={}
    wrapper['stats']=stats
    wrapper['pvals']=pvals
    wrapper['weights']=weights
    wrapper['labels']=labels
    with open(title+'.pkl', 'wb') as file:
        pkl.dump(wrapper, file)
