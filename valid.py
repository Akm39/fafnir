import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

class spread_tester:
    def __init__(self,data):
        # if used for testing only need industry data
        self.data = data
    def make_ts(self,coef,order):
        # make time series from coefficients and order
        # coef: list of coefficients
        # order: list of orders
        return self.data[order].values@coef
    def loop_make_ts(self,coefs,orders):
        #loop above function
        return np.array([self.make_ts(coef,order) for coef,order in zip(coefs,orders)]).squeeze()
    def coef_matrix(self,coefs,orders):
        frames=[]
        for a,b in (zip(orders,coefs)):
            frames.append(pd.DataFrame(dict(zip(a,b))))
        coef_matrix=pd.concat(frames).fillna(0).reset_index(drop=True) #[self.data.columns] Excluding this so trades with no values are not included
        return coef_matrix
    def weight_matrix(self,coef_matrix,margin):
        prices=self.data.mean(axis=0) #sub out with live price in final implementation
        prices['CONST']=0
        outs=coef_matrix*prices
        return outs.div(outs.abs().sum(axis=1)*margin, axis=0).drop(columns='CONST')
    def cluster_matrix(self,matrix):
        """
        Need to determine uniqueness by pre-clustering
        """
        cluster=MiniBatchKMeans(n_clusters=int(matrix.shape[0]/100),random_state=0)
        matrix['Cluster'] = cluster.fit_predict(matrix)
        return matrix
    
def apply_tstat_mask(stats,coefs,order,max_stat=0.347):
    """
    Applies Filter for MAX Test Stat
    """
    stats=np.array(stats)
    coefs=np.array(coefs)
    order=np.array(order)
    mask_stat=stats<=max_stat #HYPERPARAMETER 1
    stats_m1=stats[mask_stat]
    coefs_m1=coefs[mask_stat]
    order_m1=order[mask_stat]
    return stats_m1,coefs_m1,order_m1

def apply_weight_mask(data,stats,coefs,order,weight_mask=1,type='std',margin=0.2):
    """
    Applied filter for maximum imbalance in weights
    Based on either abs = actual imbalance
    or std = standard deviation of weights
    Uses 20% margin as default
    """
    tester=spread_tester(data)
    coef_matrix=tester.coef_matrix(coefs,order)
    weights=tester.weight_matrix(coef_matrix,margin)
    net_weights=weights.sum(axis=1)
    if type=='std':
        weight_avg=net_weights.mean()
        weight_std=net_weights.std()
        mask_weights=(((net_weights-weight_avg)>-(weight_mask*weight_std))&((net_weights-weight_avg)<(weight_mask*weight_std)))
    elif type=='abs':
        mask_weights=np.abs(net_weights)<=weight_mask
    return coef_matrix[mask_weights],stats[mask_weights]

def apply_similarity_mask(data,stats,coef_matrix,corr_threshold=0.95):
    """
    Applies filter for similarity between coefficients
    """
    train_ts=(coef_matrix.values@data[coef_matrix.columns].values.T)
    cluster=MiniBatchKMeans(n_clusters=max(1,int(train_ts.shape[0]/100)),random_state=0)
    preds=cluster.fit_predict(train_ts)
    filtered_coef_matrix=coef_matrix.copy()
    filtered_coef_matrix['Cluster']=preds
    filtered_coef_matrix['TStat']=stats
    filtered_coef_matrix.sort_values('Cluster',inplace=True)
    filtered_coef_matrix=filtered_coef_matrix.reset_index(drop=True)
    out_list=[]
    for group_num in range(max(1,int(train_ts.shape[0]/100))):
        t_stats=list(filtered_coef_matrix[filtered_coef_matrix['Cluster']==group_num]['TStat'])
        grouped_ts=(filtered_coef_matrix[filtered_coef_matrix['Cluster']==group_num].drop(columns=['Cluster','TStat']).values@data[coef_matrix.columns].values.T)
        corr_mat=np.corrcoef(grouped_ts)
        indices_to_drop = set()
        if grouped_ts.shape[0]<2:
            out_list.append(filtered_coef_matrix[filtered_coef_matrix['Cluster']==group_num])
            continue
        for i in range(corr_mat.shape[0]):
            for j in range(i + 1, corr_mat.shape[1]):
                if abs(corr_mat[i, j]) > corr_threshold:
                    if t_stats[i] > t_stats[j]:
                        indices_to_drop.add(i)
                    else:
                        indices_to_drop.add(j)
        to_kill=list(indices_to_drop)
        to_keep = list([item for item in list(range(corr_mat.shape[0])) if item not in to_kill])
        output=filtered_coef_matrix[filtered_coef_matrix['Cluster']==group_num].iloc[to_keep]
        out_list.append(output)
    return pd.concat(out_list).drop(columns=['Cluster','TStat']),np.array(pd.concat(out_list)['TStat'])
