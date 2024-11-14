
import numpy as np
import pandas as pd

# Define functions for calculating various performance metrics

def calculate_std_of_returns(equity):
    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity)
    returns = equity.pct_change()
    return returns.std()

def drawdown(equity):
    equity_series = pd.Series(equity)
    max_value = equity_series.expanding(min_periods=1).max()
    return equity_series - max_value

def max_draw_down(equity):
    dd = drawdown(equity)
    return dd.min()

def num_operations(operations):
    return operations.count()

def average_return(operations):
    return round(operations.mean(), 0)

def percent_profitable_trade(operations):
    return round((operations[operations > 0].count() / operations.count() * 100), 2)

def max_draw_down_perc(equity):
    equity_series = pd.Series(equity)
    max_value = equity_series.expanding(min_periods=1).max()
    drawdown = equity_series - max_value
    drawdown_pct = (drawdown / max_value) * 100
    return drawdown_pct.min()

def avg_dd_nozero(equity):
    dd = drawdown(equity)
    return round(dd[dd < 0].mean(), 0)

def average_loss(operations):
    return round(operations[operations < 0].mean(), 0)

def max_loss(operations):
    return round(operations.min(), 0)

def max_loss_date(operations):
    return operations.idxmin()

def average_gain(operations):
    return round(operations[operations > 0].mean(), 0)

def max_gain(operations):
    return round(operations.max(), 0)

def max_gain_date(operations):
    return operations.idxmax()

def gross_profit(operations):
    return round(operations[operations > 0].sum(), 0)

def gross_loss(operations):
    return round(operations[operations <= 0].sum(), 0)

def profit_factor(operations):
    a = gross_profit(operations)
    b = gross_loss(operations)
    return round(abs(a / b), 2) if b != 0 else np.inf

def annualize_returns(equity, periods_per_year=252):
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    compounded_growth = (1 + returns).prod()
    n_periods = len(returns)
    annualized_returns = (compounded_growth ** (periods_per_year / n_periods)) - 1
    return annualized_returns

def annualize_volatility(equity, periods_per_year=252):
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    volatility = returns.std()
    return volatility * np.sqrt(periods_per_year)

def sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=252):
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    excess_returns = returns - risk_free_rate / periods_per_year
    annualized_return = excess_returns.mean() * periods_per_year
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    return annualized_return / annualized_volatility if annualized_volatility != 0 else np.inf

# Additional functions for CVaR, Sortino Ratio, etc., can be added similarly

def calculate_performance_metrics(trades, equity):
    metrics = {}
    operations = trades['Profit/Loss']

    metrics['Number of Operations'] = num_operations(operations)
    metrics['Average Return'] = average_return(operations)
    metrics['Maximum Drawdown'] = max_draw_down(equity)
    metrics['Maximum Drawdown Percentage'] = max_draw_down_perc(equity)
    metrics['Average Loss'] = average_loss(operations)
    metrics['Maximum Loss'] = max_loss(operations)
    metrics['Date of Maximum Loss'] = max_loss_date(operations)
    metrics['Average Gain'] = average_gain(operations)
    metrics['Maximum Gain'] = max_gain(operations)
    metrics['Date of Maximum Gain'] = max_gain_date(operations)
    metrics['Gross Profit'] = gross_profit(operations)
    metrics['Gross Loss'] = gross_loss(operations)
    metrics['Profit Factor'] = profit_factor(operations)
    metrics['% Profitable Trade'] = percent_profitable_trade(operations)
    metrics['Annualized Returns'] = annualize_returns(equity)
    metrics['Annualized Volatility'] = annualize_volatility(equity)
    metrics['Sharpe Ratio'] = sharpe_ratio(equity)

    return metrics
