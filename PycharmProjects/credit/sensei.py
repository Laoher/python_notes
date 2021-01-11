# -*- coding: utf-8 -*-
"""
Script for XVA related computations 

written for SMU Credit Risk II in the MQF

This is a draft script that you need to alter for the assignment
It only provides a starting point

author: Jeroen Kerkhof
"""

# include libraries
import numpy as np # vector / matrix algebra
import matplotlib.pyplot as plt # plotting
import pandas as pd # indexed data
import statsmodels.api as sm # econometric models
import numpy.linalg as la # solving matrix equations
import seaborn as sns # pretty plotting

###############################################################################
# model parameters and simulation parameter definitions
# This section should have all the parameters that you need to change
###############################################################################

# number of simulations
num_sims = 2**17

# model parameters
# expected market return 
mu = 0.06

# initial yield curve with a constant interest rate
r = 0.02
# initial survival curve with a constant hazard rate
q = 0.02
# volatility parameters for the Black-Scholes Model
# dS(t) = S(t)[r dt + sigma dW(t)]
sigma = 0.16

# fixing and payment times
num_years = 5

fixed_rate = 0.02

###############################################################################
# Create the time grid
###############################################################################

# T_0, ..., T_N
times = np.arange(0.0, num_years+0.1)
# T_0, ..., T_{N-1}
fix_times = times[:-1]
# T_1, ..., T_{N}
pay_times = times[1:]

###############################################################################
# Compute the discount factors and surivival probabilities on the relevant
# time points
###############################################################################

#initial yield curve
# Z(0,T_0),..., Z(0,T_N)
Z_0T = np.exp(-r*times)

#credit / survival curve 
# Q(0,T_0),..., Q(0,T_N)
Q_0T = np.exp(-q*times)

###############################################################################
# Simulating the value of the equity return 
###############################################################################

# N
no_times = len(pay_times)

# integrated variance
gt = sigma**2 * times
dgt = gt[1:] - gt[:-1]

# generate normal random variables
eps = np.random.normal(0.0, 1.0, (num_sims, no_times))

# generate  equity returns
returns = (mu - 0.5 *gt[1:])*pay_times + dgt * eps

###############################################################################   
# Stage 2: Valuation
###############################################################################

# compute the value of the swap     
# We compute the value at times T_0,...,T_{N-1}
# TO DO: Value the swap for each future scenario
# value_swap =     


###############################################################################   
# Stage 2: Valuation
###############################################################################

# expected exposures
# TO DO: Compute the Expected Exposures (Positive, Negative)
# EE = np.mean( ?? , 0)
# EPE = np.mean( ?? ,0)
# ENE = np.mean( ?? ,0)
# PFE = ??
   
# Plot Exposure profiles
#plt.plot(pay_times, EE, label='EE')
#plt.plot(pay_times, EPE, label='EPE')
#plt.plot(pay_times, ENE, label='ENE')
#plt.grid()
#plt.legend()
#plt.title('Expected Exposures')
#plt.show()

# marginal default probabilities
# TO DO: Compute default probability in each period    
# dQ = 

# brute-force method cva
# TO DO: Compute cva, dva 
# cva_bfm = 
# dva_bfm = 

