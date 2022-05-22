import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as sps
import math


# Call price BS 

def d1(K,sig,T,x,r) : 
    return 1/(sig*np.sqrt(T)) *(np.log(x/K) + (r+sig**2/2) * T)

def C(K, sig, T, x,r):
    '''
    Returns the price of a Call in the Black-Scholes model with parameters : 
        - K : strike
        - sig : volatility
        - T : maturity
        - x: price of the underlying asset
        - r : interest rate
    '''
    d = d1(K,sig,T,x,r)
    return(x * sps.norm.cdf(d) - np.exp(-r*T)*K*sps.norm.cdf(d-sig*np.sqrt(T)))

def P(K, sig, T,x, r): #European Put Price BS
    '''
    Returns the price of a European Put in the Black-Scholes model with parameters : 
        - K : strike
        - sig : volatility
        - T : maturity
        - x: price of the underlying asset
        - r : interest rate
    '''
    d_1 = d1(K,sig,T,x,r)
    d_2 = d_1 - sig*np.sqrt(T)
    return np.exp(-r*T)*K*sps.norm.cdf(-d_2) - x*sps.norm.cdf(-d_1)
    
def call(s, K): #call payoff
    '''returns the payoff of a call
    '''
    return np.maximum(s-K, 0)

def put(s, K): #put payoff
    '''returns the payoff of a put
    '''
    return np.maximum(K-s, 0)

def put_min(S,K):
    return np.maximum(0, K-np.min(S))
