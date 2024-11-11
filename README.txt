STRUCTURE:

FINISHED/POSSIBLE BUGS:
+data.py - used to save & load data, precompute cov matrixes, betas, and returns (dumped in the data folder)
covs.py - calcs cov matrixes, betas, returns (wrapped inside data.py)

utils.py - runs a singular regression & contains global utils
iter.py - wrapper of utils (iterates regression over all possible combos of stocks) (wrapped inside grouptester.py)
valid.py - applies stability, cost, and similarity filters to regression coefficients (wrapped inside grouptester.py)
grouptester.py - applies iter.py & valid.py over all stocks in a sector - outputs coefficient matrix
+all_coefs.py - wrapper for grouptester.py, iterates over all industry groups applying iter.py & valid.py, & compiles coefficient matricies

+coefs.py - given coefficient matrix & price vector calculates weight matrix (probably shouldn't precompute price matrix due to speed)

+spread.py - takes coefficient matrix & calcs spreads, signals, z scores
+revert.py - takes spreads & calculates the inverse-momentum (based on a lag)

+pnl.py - calculates pnl based on old weights -> new weights (transaction costs) & new weights -> next period weights (price change)

TO DO:
1. optimization.py - takes all precomputed data (coef matrix, signals, reversions, cov matrix, betas, returns) and does a day by day optimization of the portfolio w/weight matrixes computed by coefs.py and pnl computed by pnl.py