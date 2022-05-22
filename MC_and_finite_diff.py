import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

import utils

def BS_new_val(old_val, t_old, t_new, r, sigma,nb_assets):
    ''' returns a realisation of S_t_new given S_t_old=old_val
    in a BS setting with volatility sigma and interest rate r
    '''
    return old_val * np.exp((r-sigma**2/2) * (t_new-t_old) + sigma*np.sqrt(t_new-t_old)*np.random.randn(nb_assets))
    
def node_value_low(pres_val, future_vals, time, r,t):
    ''' returns the value of a node for the low MC estimator
    '''
    b = future_vals.size
    eta = np.zeros(b)
    actual_fact = np.exp(-r*(t[time+1]-t[time]))
    for i,val in enumerate(future_vals):
        if pres_val > (np.sum(future_vals)-val) * actual_fact/(b-1):
            eta[i] = pres_val
        else:
            eta[i] = val * actual_fact

    return np.mean(eta)

def node_value_high(pres_val, future_vals, time,r,t):
    ''' returns the value of a node for the low MC estimator
    '''
    b = future_vals.size
    actual_fact = np.exp(-r*(t[time+1]-t[time]))
    return np.maximum(pres_val, np.mean(future_vals) * actual_fact)

def MC_estimator(S0, r, sigma,t,K,b, h,):
    '''
    Gives MC estimators for the option with payoff h with one underlying asset following a BS dynamics
    parameters : 
        - S0 asset price
        - r interest rate
        - sigma : volatility 
        - t : exercise dates
        - K strike
        - b : number of branches
        - h : payoff
    returns low estimator, high estimator, execution time
    '''
#allocate storage
    nb_assets=1
    d = len(t)-1
    start = time.time()
    w = np.ones(d+1, np.uint8)
    v = np.zeros((b,d+1, 2)) #0 : low esimator 1:high estimator

    #initialize param
    v[0,0] = S0
    for j in range(1,d+1) :
        v[0,j] = BS_new_val(v[0,j-1], t[j-1], t[j], r,sigma,nb_assets)
    #tree
    j = d
    while j>=0:
        if j==d:
            v[w[j]-1, j] = h(v[w[j]-1, j], K)
            #v[w[j]-1, j,1] = h(v[w[j]-1, j], K)
            if w[j]<b:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma, nb_assets)
                w[j] = w[j]+1
            elif w[j]==b:
                w[j]=0
                j=j-1
        elif j<d and w[j]<b:
            h_temp = h(v[w[j]-1,j, 0],K)
            v[w[j]-1, j,1]= node_value_high(h_temp, v[:, j+1,1],j,r,t)
            v[w[j]-1, j,0]= node_value_low(h_temp, v[:, j+1,0],j,r,t)
            if j>0:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma, nb_assets)
                w[j]+=1
                w[j+1:]=1
                v[0,j+1] = BS_new_val(v[w[j]-1, j], t[j], t[j+1], r,sigma, nb_assets)
                for k in range(j+2,d+1) : 
                    v[0,k] = BS_new_val(v[0,k-1], t[k-1], t[k], r,sigma, nb_assets)
                j=d
            else:
                j=-1
        elif j<d and w[j]==b:
            h_temp =h(v[w[j]-1,j,0],K)
            v[w[j]-1, j,1]= node_value_high(h_temp, v[:, j+1,1],j,r,t)
            v[w[j]-1, j,0]= node_value_low(h_temp, v[:, j+1, 0],j,r,t)
            w[j]=0
            j=j-1
    #tree estimate
    return v[0,0,0], v[0,0,1], time.time()-start
    
    
def MC_estimator_euro_pruning(S0, r, sigma,t,K,b, h, euro_val):
    '''
    Gives MC estimators for the option with payoff h with one underlying asset following a BS dynamics and with pruning of last layer with european value
    parameters :
        - S0 asset price
        - r interest rate
        - sigma : volatility
        - t : exercise dates
        - K strike
        - b : number of branches
        - h : payoff
    returns low estimator, high estimator, execution time
    '''
#allocate storage
    nb_assets=1
    d = len(t)-1
    start = time.time()
    w = np.ones(d, np.uint8)
    v = np.zeros((b,d)) #low esimator
    u = np.zeros((b,d)) #high estimator
    #initialize param
    v[0,0] = S0
    for j in range(1,d) :
        v[0,j] = BS_new_val(v[0,j-1], t[j-1], t[j], r,sigma,nb_assets)
    #tree
    j = d-1
    while j>=0:
        if j==d-1:
            p = euro_val(K,sigma, t[-1]-t[-2],v[w[j]-1, j], r)
            u[w[j]-1, j] = np.maximum(p, h(v[w[j]-1, j], K))
            v[w[j]-1, j] = np.maximum(p, h(v[w[j]-1, j], K))
            if w[j]<b:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma,nb_assets)
                w[j] = w[j]+1
            elif w[j]==b:
                w[j]=0
                j=j-1
        elif j<d-1 and w[j]<b:
            u[w[j]-1, j]= node_value_high(h(v[w[j]-1,j],K), u[:, j+1],j,r,t)
            v[w[j]-1, j]= node_value_low(h(v[w[j]-1,j], K), v[:, j+1],j,r,t)
            if j>0:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma,nb_assets)
                w[j]+=1
                w[j+1:]=1
                v[0,j+1] = BS_new_val(v[w[j]-1, j], t[j], t[j+1], r,sigma,nb_assets)
                for k in range(j+2,d) :
                    v[0,k] = BS_new_val(v[0,k-1], t[k-1], t[k], r,sigma,nb_assets)
                j=d-1
            else:
                j=-1
        elif j<d-1 and w[j]==b:
            u[w[j]-1, j]= node_value_high(h(v[w[j]-1,j],K), u[:, j+1],j,r,t)
            v[w[j]-1, j]= node_value_low(h(v[w[j]-1,j],K), v[:, j+1],j,r,t)
            w[j]=0
            j=j-1
    #tree estimate
    return v[0,0], u[0,0], time.time()-start
    
    
def MC_estimator_2assets(S0, r, sigma,t,K,b, h):
    '''
    Gives MC estimators for the option with payoff h with two independant underlying asset following a BS dynamics
    parameters :
        - S0 asset price
        - r interest rate
        - sigma : volatility
        - t : exercise dates
        - K strike
        - b : number of branches
        - h : payoff
    returns low estimator, high estimator, execution time
    '''
#allocate storage
    nb_assets=2
    d = len(t)-1
    start = time.time()
    w = np.ones(d+1, np.uint8)
    v = np.zeros((b,d+1, 2)) #0 : low esimator 1: high estimator

    #initialize param
    v[0,0, :] = S0
    for j in range(1,d+1) :
        v[0,j] = BS_new_val(v[0,j-1], t[j-1], t[j], r,sigma,nb_assets)
    #tree
    j = d
    #print('v1',v)
    while j>=0:
        #print('v2', v)
        if j==d:
            #u[w[j]-1, j] = h(v[w[j]-1, j], K)
            v[w[j]-1, j,:] = h(v[w[j]-1, j], K)
            if w[j]<b:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma, nb_assets)
                w[j] = w[j]+1
            elif w[j]==b:
                w[j]=0
                j=j-1
        elif j<d and w[j]<b:
            h_temp = h(v[w[j]-1,j],K)
            v[w[j]-1, j,1]= node_value_high(h_temp, v[:, j+1,1],j,r,t)
            v[w[j]-1, j,0]= node_value_low(h_temp, v[:, j+1,0],j,r,t)
            if j>0:
                prev_val = v[w[j-1]-1, j-1]
                v[w[j], j] = BS_new_val(prev_val, t[j-1], t[j], r,sigma,nb_assets)
                w[j]+=1
                w[j+1:]=1
                v[0,j+1] = BS_new_val(v[w[j]-1, j], t[j], t[j+1], r,sigma, nb_assets)
                for k in range(j+2,d+1) :
                    v[0,k] = BS_new_val(v[0,k-1], t[k-1], t[k], r,sigma, nb_assets)
                j=d
            else:
                j=-1
        elif j<d and w[j]==b:
            h_temp = h(v[w[j]-1,j],K)
            v[w[j]-1, j,1]= node_value_high(h_temp, v[:, j+1,1],j,r,t)
            v[w[j]-1, j,0]= node_value_low(h_temp, v[:, j+1,0],j,r,t)
            w[j]=0
            j=j-1
    #tree estimate
    return v[0,0,0], v[0,0,1], time.time()-start


    
###### Finite diff
    
def finite_differences(r, sigma, t,K, g,theta ,beta, alpha, m=1000, l=300):
    '''Compute theta sceme for a Bermuda option (Black Scholes model, one underlying asset)
    parameters :  
        r : interest rate 
        sigma volatility 
        t : exercise dates 
        K : strike
        g : payoff
        theta : theta in the theta scheme
        beta : high boudary
        alpha : low boundary
        m : number of time steps 
        l : number of space steps
    returns: s: discretized space values, u: values at time 0, execution time

    '''
    m = m-m%(len(t)-1) #if the exercise times are equidistant, ensure that they are in the discretization scheme
    start = time.time()
    T = t[-1]
    t_ex = len(t)-2
    h = T/m
    delta = (beta-alpha) / (l+1)
    x = alpha + delta * np.arange(1,l+1)
    s = np.exp(x)

    a = -sigma**2/delta**2 - r
    b = 1/2 * (sigma**2/delta**2 + (r - sigma**2/2)/delta)
    c = 1/2 * (sigma**2/delta**2 - (r - sigma**2/2)/delta)


    A = np.diag(a*np.ones(l)) + np.diag(b * np.ones(l-1) , 1) + np.diag(c * np.ones(l-1), -1)
    A[0,0]+=c
    A[l-1,l-1]+=b
    u = np.zeros(l)
    u = g(s,K)

    for n in range(m, 0,-1):
        u = np.linalg.inv(np.eye(l) - h*theta * A) @ ((np.eye(l) + h*(1-theta) * A) @ u)
        if np.abs((n-1)*T/m - t[-t_ex])<=h/2:
            payoff = g(s,K)
            u = np.maximum(payoff, u)
            t_ex+=1
    return s,u, time.time()-start
    
def get_interpolated_value(S_0, s, u, l, alpha, beta):
    ''' Gives a value when initial price = S_0 when this point is not considered in the finite difference scheme
    '''
    x_0 = np.log(S_0)
    i = int((x_0 - alpha) * (l+1)/(beta-alpha))
    return ((u[i]-u[i-1])/(s[i]-s[i-1]) * (S_0-s[i-1]) + u[i-1])



def finite_differences_2assets(r, sigma, t,K, g ,beta1, alpha1,beta2, alpha2,  m=1000, l1=300, l2=300):
    '''Compute theta sceme for a Bermuda option (Black Scholes model, 2 independant underlying assets)
    parameters :
        r : interest rate
        sigma volatility
        t : exercise dates
        K : strike
        g : payoff
        theta : theta in the theta scheme
        beta1 (resp. beta2) : high boudary for asset 1 (resp. asset 2)
        alpha1 (resp. alpha2) : low boundary for asset 1 (resp. asset 2)
        m : number of time steps
        l1 (resp. l2): number of space steps for asset 1 (resp. asset 2)
    returns: s1, s2: discretized space values, u: values at time 0, execution time

    '''
    m = m-m%(len(t)-1) #if the exercise times are equidistant, ensure that they are in the discretization scheme
    start = time.time()
    T = t[-1]
    t_ex = len(t)-2
    h = T/m
    delta1 = (beta1-alpha1) / (l1+1)
    x1 = alpha1 + delta1 * np.arange(1,l1+1)
    s1 = np.exp(x1)
    
    delta2 = (beta2-alpha2) / (l2+1)
    x2 = alpha2 + delta2 * np.arange(1,l2+1)
    s2 = np.exp(x2)

    a = -sigma**2/delta1**2 - r
    b = 1/2 * (sigma**2/delta1**2 + (r - sigma**2/2)/delta1)
    c = 1/2 * (sigma**2/delta1**2 - (r - sigma**2/2)/delta1)
    
    a_star = -sigma**2/delta2**2
    b_star = 1/2 * (sigma**2/delta2**2 - (r - sigma**2/2)/delta2)
    c_star = 1/2 * (sigma**2/delta2**2 + (r - sigma**2/2)/delta2)

    A = np.diag(a*np.ones(l1)) + np.diag(b * np.ones(l1-1) , 1) + np.diag(c * np.ones(l1-1), -1)
    A[0,0]+=c
    A[l1-1,l1-1]+=b
    B = np.diag(a_star*np.ones(l2)) + np.diag(b_star * np.ones(l2-1) , 1) + np.diag(c_star * np.ones(l2-1), -1)
    B[0,0]+=b_star
    B[l2-1,l2-1]+=c_star
    u = np.zeros((l1,l2))
    payoffs = np.zeros((l1,l2))
    for i in range (l1):
        for j in range(l2):
            u[i,j] = g([s1[i], s2[j]],K)
            payoffs[i,j] = g([s1[i], s2[j]],K)

    for n in range(m, 0,-1):
        #u = np.linalg.inv(np.eye(l) - h*theta * A) @ ((np.eye(l) + h*(1-theta) * A) @ u)
        u = h* (A@u + u@B) + u
        if np.abs((n-1)*T/m - t[-t_ex])<=h/2:
            u = np.maximum(payoffs, u)
            t_ex+=1
    return s1,s2,u, time.time()-start
