import numpy as np

def half_life(ts):
    """
    Calculate half life of mean reversion of a time series
    """
    diff = np.diff(ts)
    lagged=ts[:-1]
    beta=np.linalg.lstsq(np.vstack((lagged,np.ones(lagged.shape))).T,diff.T,rcond=-1)[0][0]
    return (-np.log(2) / np.log(1 + beta))

def rolling_hl(ts,lag=30):
    """
    Rolling half life mean reversion speed calculation
    """
    roll_mr=[np.nan]*lag
    for i in range(lag,len(ts)):
        roll_mr.append(half_life(ts[i-lag:i]))
    return np.array(roll_mr)

def df_to_rev(data,lag=60):
    """
    Applies rolling hl to time series stored in pd.DataFrame as seperate columns
    """
    dat_rev=data.copy()
    cols=list(dat_rev.columns)
    for col in cols:
        dat_rev[col]=rolling_hl(dat_rev[col],lag=60)
    return dat_rev

def df_to_dict(df):
    """
    MAKE SURE CORRECT INDEX IS PROVIDED (USED AS INDEX IN DICT)
    """
    return dict(df.iterrows())

def df_to_inv_mom(data,lag=30):
    """
    Use Z Scores not actual signals
    """
    return -(data - data.shift(lag))