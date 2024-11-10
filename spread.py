import pandas as pd
import datetime as dt

def make_spread(coef_matrix,price_df):
    asset_order=list(coef_matrix.columns)
    return ((coef_matrix@price_df[asset_order].T)).T

class spread_manager:
    def __init__(self, data,coef_matrix):
        """
        ENSURE ASSET DATA & COEF_MATRIX ARE IN THE SAME ORDER BEFORE IMPORTING
        """
        self.data = data
        self.coef_matrix = coef_matrix
        self.asset_order=list(coef_matrix.columns)
        self.spreads=((self.coef_matrix@self.data[self.asset_order].T)).T
        self.spread_list=list(self.spreads.columns)
    def calc_costs(self,margin=0.2):
        self.abs_coefs=abs(self.coef_matrix.copy())
        self.abs_coefs['CONST']=0
        self.costs=margin*((self.abs_coefs@self.data[self.asset_order].T)).T
    def raw_signal(self):
        self.signal=-self.spreads/self.costs
        return self.signal
    def z_score(self,lag=60):
        stds=[]
        for i in range(self.signal.shape[0]):
            stds.append(self.signal.iloc[max(i-lag+1,0):i+1].std())
        stdevs=pd.concat(stds,axis=1).T
        stdevs.index=self.signal.index
        self.signal_z=self.signal/stdevs
        return self.signal_z
    def new_z_score(self,cutoff=dt.date(2023,12,31)):
        signals_std=self.signal[self.signal.index<=pd.Timestamp(cutoff)].std()
        self.signal_z=self.signal/signals_std
        return self.signal_z
    def signal_z_score_filter(self,threshold=3):
        self.filterer=(self.signal_z<threshold)&(self.signal_z>-threshold)
        return self.filterer
    def signal_z_score(self):
        return self.signal*self.filterer