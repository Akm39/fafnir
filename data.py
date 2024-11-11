import glob
import yfinance as yf
import pickle as pkl
from covs import cov_B_maker
import datetime as dt
import pandas as pd
import numpy as np

class data_saver:
    def __init__(self,dataset='2024-01-01'):
        self.dataset=dataset
        self.spx=pd.read_excel(glob.glob('data/'+self.dataset+'/SPX as of *.xlsx')[0])
        self.spx.rename(columns={'Beta:Y-1\n':'Beta:Y-1','P/E\n':'P/E','Market Cap\n':'Market Cap','GICS Ind Grp Name\n':'GICS Ind Grp Name','GICS SubInd Name\n':'GICS SubInd Name','GICS Ind Name\n':'GICS Ind Name','GICS Sector\n':'GICS Sector'},inplace=True)
        self.raw_tickers=self.spx['Ticker'].apply(lambda x:x.split()[0])
        self.spx['Raw Tickers']=self.raw_tickers
    def data_pool(self,tickers,start='2021-01-01',end='2024-11-02'):
        data_pool = []
        for ticker in tickers:
            data_pool.append(yf.download(ticker,start=start,end=end)['Adj Close'])
        data=pd.concat(data_pool,axis=1)
        data.columns=tickers
        return data
    def fetch_data(self,start='2021-01-01',end='2024-11-02'):
        self.start=start
        self.end=end
        self.all_data=self.data_pool(self.raw_tickers,start,end)
    def missing_data(self):
        missing_counts=self.all_data.isnull().sum()
        return missing_counts[missing_counts>0]
    def drop_missings(self):
        self.final_data=self.all_data.dropna(axis=1)
    def dump_data(self,grouping='GICS Ind Grp Name'):
        self.drop_missings()
        groups=self.dict_labels(grouping)
        cov_maker=cov_B_maker(self.final_data)
        covs=cov_maker.calc_covs()
        cov_maker.loadspy(self.start,self.end)
        betas=cov_maker.calc_betas()
        rets=cov_maker.get_rets()
        out_data={'Price Data':self.final_data,
                  'Group Data':groups,
                  'Covariances':covs,
                  'Betas':betas,
                  'Returns':rets}
        filepath='data/'+self.dataset+'/all_data.pkl'
        with open(filepath, 'wb') as file:
            pkl.dump(out_data, file)
    def dict_labels(self,grouping='GICS Ind Grp Name'):
        missing_tickers=list(self.missing_data().index)
        filtered_spx=self.spx[~self.spx['Raw Tickers'].isin(missing_tickers)]
        return filtered_spx.groupby(grouping)['Raw Tickers'].apply(list).to_dict()
    
class data_loader:
    def __init__(self,dataset='2024-01-01'):
        self.dataset=dataset
        with open('data/'+self.dataset+'/all_data.pkl', 'rb') as file:
            self.data = pkl.load(file)
        self.set_cutoff() #Just in case i forget to manually assign it
    def set_cutoff(self,cutoff=dt.datetime(2023,12,31)):
        """
        Defines cutoff between training & testing
        """
        self.cutoff=cutoff
    def sector_list(self):
        return list(self.data['Group Data'].keys())
    def load_price(self,sector='all',tnt='all'):
        """
        Loads price data for a given sector
        """
        price_data=self.data['Price Data'].copy()
        if sector=='all':
            tickers=list(price_data.columns)
        else:
            tickers=self.data['Group Data'][sector]
        price_data=price_data[tickers]

        if tnt=='all':
            return price_data
        elif tnt=='train':
            return price_data[price_data.index<=self.cutoff]
        elif tnt=='test':
            return price_data[price_data.index>self.cutoff]
        
    def load_data(self,date,dtype='cov'):
        """
        Load specific type of data (other than prices) on specific datetime
        """
        timestamp=pd.Timestamp(date)
        if dtype=='cov':
            return self.data['Covariances'][timestamp]
        elif dtype=='beta':
            return self.data['Betas'][timestamp]
        elif dtype=='returns':
            return self.data['Returns'][timestamp]
        
    def all_dates(self):
        """
        Returns list of all dates
        """
        return np.array(list(self.data['Returns'].keys()))