from utils import data_pool
import pandas as pd
import numpy as np

class cov_B_maker:
    def __init__(self,data):
        self.data=data
        self.returns=self.data.pct_change()
    def calc_covs(self,lag=250):
        self.dates=list(self.returns.index)
        self.covs={}
        for i in range(len(self.dates)):
            self.covs[self.dates[i]]=((self.returns.iloc[max(i-lag+1,0):1+i].cov()*252)) #*252 to annualize
        return self.covs
    def loadspy(self,start,end):
        self.spy=data_pool(['SPY'],start=start,end=end)
        self.spy_rets=self.spy.pct_change()
    def beta(self,asset_rets,mkt_rets):
        return np.cov(asset_rets,mkt_rets)[0,1]/np.var(mkt_rets)
    def rolling_beta(self,asset_rets,mkt_rets,lag=250):
        agg=pd.concat([asset_rets,mkt_rets],axis=1)
        betas=[]
        for i in range(agg.shape[0]):
            temp_cov=(agg.iloc[max(i-lag+1,0):i+1].cov())
            betas.append(temp_cov.iloc[0,1]/temp_cov.iloc[1,1])
        return betas
    def calc_betas(self,lag=250):
        self.assets=list(self.returns.columns)
        self.beta_df=self.returns.copy()
        for asset in self.assets:
            self.beta_df[asset]=self.rolling_beta(self.beta_df[asset],self.spy_rets['SPY'],lag=lag)
        return dict(self.beta_df.iterrows())
    def get_rets(self):
        return dict(self.returns.iterrows())