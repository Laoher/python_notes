import numpy as np  # vector / matrix algebra
import matplotlib.pyplot as plt  # plotting
import pandas as pd  # indexed data
import statsmodels.api as sm  # econometric models
import numpy.linalg as la  # solving matrix equations
import seaborn as sns  # pretty plotting
from scipy.stats import norm

# number of simulations
num_sims = 2 ** 17

# model parameters
# expected market return
mu = 0.08

# initial yield curve with a constant interest rate
r = 0.04
# initial survival curve with a constant hazard rate
q = 0.02
# volatility parameters for the Black-Scholes Model
# dS(t) = S(t)[r dt + sigma dW(t)]
sigma = 0.16

# fixing and payment times
num_years = 5

fixed_rate = 0.04

###############################################################################
# Create the time grid
###############################################################################

# T_0, ..., T_N
times = np.arange(0.0, num_years + 0.1)
# T_0, ..., T_{N-1}
fix_times = times[:-1]
# T_1, ..., T_{N}
pay_times = times[1:]

###############################################################################
# Compute the discount factors and surivival probabilities on the relevant
# time points
###############################################################################

# initial yield curve
# Z(0,T_0),..., Z(0,T_N)
Z_0T = np.exp(-r * times)

# credit / survival curve
# Q(0,T_0),..., Q(0,T_N)
Q_0T = np.exp(-q * times)

###############################################################################
# For these computations we assume a discrete time grid.
###############################################################################

# fixed leg
K = fixed_rate

# equity return
# log_return = (mu - 0.5 * sigma ** 2) * 1 + sigma * 1
log_return = (mu - 0.5 * sigma ** 2) * 1
# compute the value of the swap
# We compute the value at times T_0,...,T_{N-1} all is the same
##X_K = (log_return - K) * norm.cdf((mu - K) / sigma) + sigma * norm.pdf((mu - K) / sigma)
X_K = (log_return - K)
# DF = Z_0T * Q_0T
DF = Z_0T
# value_swap = X_K * (sum(DF[1:]))
value = X_K * DF[1:]

value_swap = [0, 1, 2, 3, 4]


for i in range(0, 5):
    value_swap[i] = sum(value[i:]) / DF[i + 1]

print(value_swap)

# Compute the Expected Positive Exposure
EPE = [0, 1, 2, 3, 4]
for i in range(0, 5):
    EPE[i] = (mu - K) * norm.cdf((mu - K) / sigma) + sigma * norm.cdf((mu - K) / sigma) * DF[i + 1]

print(EPE)
# Compute the Expected Negative Exposure
ENE = [0, 1, 2, 3, 4]
for i in range(0, 5):
    ENE[i] = (K - mu) * norm.cdf((K - mu) / sigma) + sigma * norm.cdf((mu - K) / sigma) * DF[i + 1]

print(ENE)
EE=[0,1,2,3,4]
for i in range(0,5):
    EE[i]=EPE[i]+ENE[i]
print(EE)
# Compute the Expected Exposure
# EE = EPE + ENE
# Plot Exposure profiles
plt.plot(pay_times, EE, label='EE')
plt.plot(pay_times, EPE, label='EPE')
plt.plot(pay_times, ENE, label='ENE')
plt.grid()
plt.legend()
plt.title('Expected Exposures')
plt.show()
# 回头乘个本金
# Compute the Potential Future Exposure with 95%
# PFE =

# cva dva
CVA = 0
for i in range(0, 5):
    CVA = CVA + (Q_0T[i] - Q_0T[i + 1]) * EPE[i] * DF[i]
print(CVA)
DVA = 0
for i in range(0, 5):
    DVA = DVA + (Q_0T[i] - Q_0T[i + 1]) * ENE[i] * DF[i]
print(DVA)
PFE=sigma*1.65
print(PFE)
###############################################################################
# Monte-Carlo valuation
###############################################################################
# N
no_times = len(pay_times)

# integrated variance
gt = sigma ** 2 * times
dgt = gt[1:] - gt[:-1]

# generate normal random variables
eps = np.random.normal(0.0, 1.0, (num_sims, no_times))

# generate  equity returns
returns = (mu - 0.5 * gt[1:]) * pay_times + dgt * eps
sim_log_return = np.mean(returns - K, 0)

# compute the value of the swap
# sim_value_swap = sum(sim_log_return * DF[1:])
for i in range(0,5):
    value[i] = sim_log_return[i] * DF[i+1]

for i in range(0, 5):
    value_swap[i] = sum(value[i:]) / DF[i + 1]

print(value_swap)

# Compute the Expected Exposuredg

# Compute the Expected Positive Exposure

# Compute the Expected Negative Exposure


# Compute the Potential Future Exposure with 95%

# PFE = norm.ppf(0.95)

#monthly calculation from here
num_months = 60

# T_0, ..., T_N
times = np.arange(0.0, num_months + 0.1)
# T_1, ..., T_{N}
pay_times = times[1:]


###############################################################################
# Compute the discount factors and surivival probabilities on the relevant
# time points
###############################################################################

# initial yield curve
# Z(0,T_0),..., Z(0,T_N)
Z_0T = np.exp(-r * times)

# credit / survival curve
# Q(0,T_0),..., Q(0,T_N)
Q_0T = np.exp(-q * times)

###############################################################################
# For these computations we assume a discrete time grid.
###############################################################################

# fixed leg
K = fixed_rate

# equity return
###############################################################################
# Monte-Carlo valuation
###############################################################################
# N
no_times = len(pay_times)

# integrated variance
gt = sigma ** 2 * times
dgt = gt[1:] - gt[:-1]

# generate normal random variables
eps = np.random.normal(0.0, 1.0, (num_sims, no_times))

# generate  equity returns
returns = (mu - 0.5 * gt[1:]) * pay_times + dgt * eps
sim_log_return = np.mean(returns - K, 0)

# compute the value of the swap
# sim_value_swap = sum(sim_log_return * DF[1:])
for i in range(0,5):
    value[i] = sim_log_return[i] * DF[i+1]

for i in range(0, 5):
    value_swap[i] = sum(value[i:]) / DF[i + 1]
# compute the value of the swap

print(value_swap)