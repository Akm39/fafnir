import pandas as pd
import numpy as np
from iter import iter_all
from valid import apply_tstat_mask,apply_weight_mask,apply_similarity_mask

def group_to_coef(train_data,group_name,max_loop=5,t_stat=0.3,weight_std=1,corr_thresh=0.9,weight_type='std'):
    """
    Wrapper for iter.py & valid.py
    """
    it_upto=(min(train_data.shape[1],max_loop))
    its=list(range(2,it_upto+1))
    coef_mats=[]
    tstats=[]
    for it in its:
        stats,_,weights,order=iter_all(train_data,it)
        stats_m1,coefs_m1,order_m1=apply_tstat_mask(stats,weights,order,max_stat=t_stat)
        if len(list(stats_m1))<1:
            continue
        coef_matrix,stats_m2=apply_weight_mask(train_data,stats_m1,coefs_m1,order_m1,weight_mask=weight_std,type=weight_type)
        if len(list(stats_m2))<1:
            continue
        coef_matrix_2,stats_m3=apply_similarity_mask(train_data,stats_m2,coef_matrix,corr_threshold=corr_thresh)
        if coef_matrix_2.shape[0]>1000:
            coef_matrix_2,stats_m3=apply_similarity_mask(train_data,stats_m3,coef_matrix_2,corr_threshold=corr_thresh)
        coef_mats.append(coef_matrix_2)
        tstats.append(stats_m3)
    agg_coef=pd.concat(coef_mats).fillna(0)
    agg_stats=np.array([item for sublist in tstats for item in sublist])
    coef_matrix_final,stats_final=apply_similarity_mask(train_data,agg_stats,agg_coef,corr_threshold=corr_thresh)
    coef_matrix_final,stats_final=apply_similarity_mask(train_data,stats_final,coef_matrix_final,corr_threshold=corr_thresh)
    coef_matrix_final.index=[group_name+f" {i}" for i in range(1, coef_matrix_final.shape[0]+1)]
    return coef_matrix_final

def agg_sectors(sector_coef_matrixes):
    """
    Feed sector_coef_matrixes in as list of dataframes
    """
    return pd.concat(sector_coef_matrixes).fillna(0)