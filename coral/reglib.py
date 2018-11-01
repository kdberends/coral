# Bayesian Regression Functions
# This file: Preferences for plotting
# 
# Author: Koen Berends
# Contact: k.d.berends@utwente.nl 
# Copyright (c) 2017 University of Twente
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

# =============================================================================
# Imports
# =============================================================================
import theano
import numpy as np
import pymc3 as pm
import random
from coral.statsfunc import get_empirical_cdf, empirical_ppf
import matplotlib.pyplot as plt 

# =============================================================================
# Functions to model response given a trace
# =============================================================================


def regression_interval(predictor, response_modelled, ci_interval):
    """
    Does something
    """
    ci_qtiles = np.abs(np.array([0, 100]) - (100 - ci_interval) / 2.)
    x_interval = np.linspace(*(np.append(np.sort(predictor)[[0, -1]], 101)))
    interval = list()
    for column in np.array(response_modelled).T:
        interval.append(empirical_ppf(ci_qtiles, column))
    interval = np.array(interval).T
    mean = np.mean(response_modelled, axis=0)
    return dict(x=x_interval, mean=mean, cilow=interval[0], cihigh=interval[1])

def cdf_interval(data, ci_interval):
    """
    x = list of probabilities (0-1)
    mean = mean values belonging to x
    qtiles = 2.5 and 97.5 quantiles (for 95% interval) belonging to x
    """
    cdfs = list()
    qtiles = list()
    prob_x = np.linspace(0, 1, 101)
    ci_qtiles = np.abs(np.array([0, 100]) - (100 - ci_interval) / 2.)

    # Retrieve a cdf for each modelled response
    for rm in data:
        p, val = get_empirical_cdf(rm)
        cdfs.append(np.interp(prob_x,  p, val))

    def get_cdf_intervals(cdfs): 
        """For each cdf, return the ci quantiles"""
        qtiles = list()
        for cdf in cdfs:
            qtiles.append(empirical_ppf(ci_qtiles, cdf))
        return qtiles

    cdfs = np.array(cdfs).T
    qtiles = np.array(get_cdf_intervals(cdfs)).T
    cdfmean = np.mean(cdfs, axis=1)

    return dict(x=prob_x, cdfs=cdfs, mean=cdfmean, quantiles=qtiles)

def mtrace_glm(x, trace, number_of_draws):
    """
    Arguments:
        x: list
        trace: loaded trace (pandas dataframe)
        number_of_draws: int

    returns:
        np array
    """
    a = trace['a']
    b = trace['b']
    sigma = trace['sigma']
    a, b, sigma = map(np.array, [a, b, sigma])

    modelled_response = list()
    for i in range(number_of_draws):
        # Draws from MCMC Trace
        modelled_response.append(a[i] + b[i] * x + np.random.normal(loc=0, scale=sigma[i], size=len(x)))

    return np.array(modelled_response)

def mtrace_glmrbf(x, trace, number_of_draws):
    """
    Arguments:
        x: list
        trace: loaded trace (pandas dataframe)
        number_of_draws: int

    returns:
        np array
    """
    number_of_draws = int(number_of_draws)
    theta = [trace['a1'], trace['a2']]
    w = [trace['a3'], trace['a4']]
    loc = [trace['v0'], trace['v1']]
    gamma = [trace['t0'], trace['t1']]
    sigma = trace['sigma']
    theta, w, loc, gamma, sigma = map(np.array, [theta, w, loc, gamma, sigma])

    modelled_response = list()

    for i in range(number_of_draws):
        # Draws from MCMC Trace
        rbf = glm_rbf(x, 
                      theta = theta.T[i], 
                      w= w.T[i], 
                      loc= loc.T[i], 
                      gamma=gamma.T[i])
        # add noise
        modelled_response.append(np.array(rbf) + np.random.normal(loc=0, scale=sigma[i], size=len(x)))

    return np.array(modelled_response)

def mtrace_dirichlet(x, trace, number_of_draws):
    
    with model:
        trace = pm.backends.text.load('.')
        return pm.sample_ppc(trace, 5000)

def ppcgpwn(x, model, number_of_draws):
    print (model)
    with model:
        f_pred = gp.conditional("f_pred", x)

# =============================================================================
# Kernel Functions
# =============================================================================

def glm_rbf(x, theta=[0, 1], w=[0], loc=[1], gamma=[2]):
    """
    Linear combination of GLM and RBF. 

    For a large number of runs, glm_rbf_theano is more efficient

    Arguments:
        x: list of predictor values

    Keyword arguments:
        theta: list, [intercept, slope]
        w: list of kernel weights
        loc: list of kernel positions
        gamma: list of kernel precisions

    Returns:
        y = f(x | theta, w, loc, gamma)

    """

    output = list()
    theta, w, loc, gamma = map(np.array, [theta, w, loc, gamma])
    for ix in x:
        glm = theta[0] + theta[1] * ix
        gauss_rbfk = np.sum(w * np.exp(-gamma * np.abs(loc - ix) ** 2))
        output.append(glm + gauss_rbfk)

    return output

def glm_rbf_theano(x, theta=[0, 1], w=[0], loc=[1], gamma=[2]):
    """
    Linear combination of GLM and RBF. 
    
    For a small number of runs, glm_rbf is more efficient

    This function uses Theano for use in a PyMC3 inference context.

    Arguments:
        x: list of predictor values

    Keyword arguments:
        theta: list, [intercept, slope]
        w: list of kernel weights
        loc: list of kernel positions
        gamma: list of kernel precisions

    Returns:
        y = f(x | theta, w, loc, gamma)

    """
    linear_part = theta[0] + theta[1] * x
    
    kmax = len(w)
    kernel_part = np.zeros(len(x))
    for i in range(kmax):
        if theano.tensor.lt(np.array(i), kmax):
            kernel_part = kernel_part + w[i] * np.exp(-gamma[i] * np.abs(loc[i] - x) ** 2)
    
    return linear_part + kernel_part

# =============================================================================
# Standalone
# ============================================================================= 


def general_linear(predictor_sample, response_sample, predictor, params):
    """

    """

    X = predictor_sample
    y = response_sample
    X_new = predictor

    with pm.Model():

        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
        intercept = pm.Normal('Intercept', 0, sd=20)
        slope_coeff = pm.Normal('Slope', 0, sd=20)

        # Define likelihood
        likelihood = pm.Normal('y', mu=intercept + slope_coeff * X,
                                sd=sigma, observed=y)

        # Draw samples using NUTS sampler
        trace = pm.sample(draws=params.draws, chains=params.chains, cores=params.cores, tune=params.burn_in)  

    #pm.traceplot(trace)
    #plt.savefig('{}_trace.png'.format(random.getrandbits(48)))
    #plt.close()
    # After burn-in MCMC should sample from the poster predictive
    a = trace.get_values('Intercept', burn=params.burn_in, combine=True)  # trace['Intercept'][params.burn_in:]
    b = trace.get_values('Slope', burn=params.burn_in, combine=True)  # trace['Slope'][params.burn_in:]
    sigma = trace.get_values('sigma', burn=params.burn_in, combine=True)  # trace['sigma'][params.burn_in:]
    a, b, sigma = map(np.array, [a, b, sigma])

    response_modelled = list()
    for i in range(len(a)):
        response_modelled.append(a[i] + b[i] * X_new + np.random.normal(loc=0, scale=sigma[i], size=len(X_new)))
    
    # summarize trace for 95, 89, 80, 50, 20 and 10 % ci
    
    prob_x = np.array([2.5, 5, 10, 25, 40, 45, 55, 60, 75, 90, 95, 97.5])/100
    def sumtrace(data):
        p, val = get_empirical_cdf(data)
        return {'p':prob_x.tolist(), 'val': np.interp(prob_x,  p, val).tolist()}
    #trace_summary = {'intercept':sumtrace(a), 'slope':sumtrace(b), 'sigma':sumtrace(sigma)}
    trace_summary = {'intercept':a[::10].tolist(), 'slope':b[::10].tolist(), 'sigma':sigma[::10].tolist()}
    return np.array(response_modelled), trace_summary


def gaussian_process(predictor_sample, response_sample, predictor, params):
    """
    Args:
    predictor_sample : subsample of predictor
    response_sample : subsample of response
    predictor: full sample of predictor
    params: ParameterContainer object

    Returns:
    modelled response,
    inference trace (if mcmc)
    inference estimates (if map)
    """
    X = predictor_sample[:, None]
    y = response_sample
    X_new = predictor[:, None]
    with pm.Model() as model:

        # length scale factor
        L = pm.Gamma("L", alpha=2, beta=1)

        # Covariance scale factor
        eta = pm.HalfCauchy("eta", beta=5)

        # todo: set by parameters?
        kernel = 'radialbasis'
        if kernel == 'matern':
            # Matern kernel
            cov = eta**2 * pm.gp.cov.Matern52(1, L)
        elif kernel == 'radialbasis':
            # Radial basis kernel
            cov = eta**2 * pm.gp.cov.ExpQuad(1, L)

        gp = pm.gp.Marginal(cov_func=cov)

        sigma = pm.HalfCauchy("sigma", beta=15)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)     

        if params.inference == 'map':
            mp = pm.find_MAP()
        elif params.inference == 'mcmc':
            mp = pm.sample(10000)

    with model:
        y_pred = gp.conditional("y_pred", X_new, pred_noise=True)
        response_modelled = pm.sample_ppc([mp],
                                          vars=[y_pred],
                                          samples=params.ppc_draws)

    return response_modelled['y_pred'], mp
# =============================================================================
# Markov-Chain Monte Carlo Functions
# =============================================================================

def mcmc_glmrbf(X, Y, **parameters):
    """
    Linear regression model with Gaussian likelihood

    Arguments:
        k: int, number of gaussian kernels
        mcmc_steps: int, steps in mcmc 
        burn_in: float, fraction of mcmc steps to throw away
        outputpath: str, path to trace dir
    Returns:
        model
    """
    #for key, val in parameters.items():
    #    exec(key + '=val')
    steps = parameters.get('steps')
    burnin = parameters.get('burnin')
    k = parameters.get('k')
    outputpath = parameters.get('outputpath')
    traceplot = parameters.get('traceplot')
    burn_in = int(np.ceil(steps * burnin))

    kernel_model = pm.Model()
    kmax = k
    with kernel_model:

        theta = [pm.Normal('a1', mu=0, sd=10), 
                 pm.Normal('a2', mu=0, sd=10)]

        w = list()
        for i in range(kmax):
            w.append(pm.Normal('a{}'.format(i + 3), mu=0, sd=10))
        
        v = list()
        for i in range(kmax):
            v.append(pm.Uniform('v{}'.format(i), lower=12, upper=20))
        t = list()
        for i in range(kmax):
            t.append(pm.Gamma('t{}'.format(i), alpha=0.001, beta=0.001))

        sigma = pm.InverseGamma('sigma', alpha=0.1, beta=0.01)
        
        # Expected value of outcome
        mu = glm_rbf_theano(X, theta, w, v, t)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

        # Draw samples using Metropolis sampler
        step = pm.Metropolis()
        trace = pm.sample(steps, step)
        #trace = pm.sample(steps)

    pm.backends.text.dump(outputpath, trace[burn_in:], chains=None)
    if traceplot:
        pm.traceplot(trace[burn_in:])

    return kernel_model

def mcmc_glm(X, Y, **parameters):

    """
    Linear regression model with Gaussian likelihood

    Arguments:
        df: int, degrees of freedom
        mcmc_steps: int, steps in mcmc 
        burn_in: float, fraction of mcmc steps to throw away
        outputpath: str, path to trace dir
    Returns:
        model
    """

    for key, val in parameters.items():
        exec(key + '=val')

    burnin = int(np.ceil(steps * burnin))

    with pm.Model() as dh_model:

        # Priors for unknown model parameters
        a = pm.Flat('a')
        b = pm.Flat('b')
        
        # Prior for gaussian noise
        esp = pm.InverseGamma('sigma', alpha=0.1, beta=0.1)
        
        # Expected value of outcome
        mu_est = a + b * X
         
        # Likelihood
        Y_obs = pm.Normal('Y_obs', mu=mu_est, sd=esp, observed=Y)

        # Draw samples using NUTS sampler
        trace = pm.sample(steps)
    
    trace = trace[burnin:]
    if traceplot:
        pm.traceplot(trace)   

    pm.backends.text.dump(outputpath, trace, chains=None)
    return dh_model

def dirichletprocess(X, Y, **parameters):
    """
    Dirichlet process regression
    http://docs.pymc.io/notebooks/dependent_density_regression.html
    Arguments:
        df: int, degrees of freedom
        mcmc_steps: int, steps in mcmc 
        burn_in: float, fraction of mcmc steps to throw away
        outputpath: str, path to trace dir
    Returns:
        model
    """
    kmax = parameters.get('k')
    steps = parameters.get('steps')
    outputpath = parameters.get('outputpath')
    
    BURN = int(steps / 2.)

    # convert X for theano
    X = X[:, np.newaxis]
    X = theano.shared(X, broadcastable=(False, True))

    def norm_cdf(z):
        return 0.5 * (1 + theano.tensor.erf(z / np.sqrt(2)))

    def stick_breaking(v):
        return v * theano.tensor.concatenate([theano.tensor.ones_like(v[:, :1]),
                                              theano.tensor.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                                              axis=1)

    with pm.Model() as dirprocessmodel:
        alpha = pm.Normal('alpha', 0., 5., shape=kmax)
        beta = pm.Normal('beta', 0., 5., shape=kmax)
        v = norm_cdf(alpha + beta * X)
        w = pm.Deterministic('w', stick_breaking(v))

        gamma = pm.Normal('gamma', 0., 10., shape=kmax)
        delta = pm.Normal('delta', 0., 10., shape=kmax)
        mu = pm.Deterministic('mu', gamma + delta * X)

        tau = pm.Gamma('tau', 1., 1., shape=kmax)
        obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=Y)

        step = pm.Metropolis()
        trace = pm.sample(steps, step, tune=BURN)

        pm.backends.text.dump(outputpath, trace)
    return dirprocessmodel

def dpmodel(X, Y, **parameters):
    """
    Dirichlet process regression
    http://docs.pymc.io/notebooks/dependent_density_regression.html
    Arguments:
        df: int, degrees of freedom
        mcmc_steps: int, steps in mcmc 
        burn_in: float, fraction of mcmc steps to throw away
        outputpath: str, path to trace dir
    Returns:
        model
    """
    kmax = parameters.get('k')
    steps = parameters.get('steps')
    outputpath = parameters.get('outputpath')
    
    BURN = int(steps / 2.)

    # convert X for theano
    X = X[:, np.newaxis]
    X = theano.shared(X, broadcastable=(False, True))

    def norm_cdf(z):
        return 0.5 * (1 + theano.tensor.erf(z / np.sqrt(2)))

    def stick_breaking(v):
        return v * theano.tensor.concatenate([theano.tensor.ones_like(v[:, :1]),
                                              theano.tensor.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                                              axis=1)

    with pm.Model() as dirprocessmodel:
        alpha = pm.Normal('alpha', 0., 5., shape=kmax)
        beta = pm.Normal('beta', 0., 5., shape=kmax)
        v = norm_cdf(alpha + beta * X)
        w = pm.Deterministic('w', stick_breaking(v))

        gamma = pm.Normal('gamma', 0., 10., shape=kmax)
        delta = pm.Normal('delta', 0., 10., shape=kmax)
        mu = pm.Deterministic('mu', gamma + delta * X)

        tau = pm.Gamma('tau', 1., 1., shape=kmax)
        obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=Y)

    return dirprocessmodel

# =============================================================================
# Process models
# =============================================================================

def gpwn(X, Y, **parameters):
    """
    Gaussian process with white noise
    """

    # to column vector
    X = X[:, None]
    with pm.Model() as model:
        L = pm.Gamma("L", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)
        
        cov = eta**2 * pm.gp.cov.Matern52(1, L)
        gp = pm.gp.Marginal(cov_func=cov)
         
        #σ = pm.HalfCauchy("σ", beta=15)
        sigma = pm.HalfCauchy("sigma", beta=15)

        y_ = gp.marginal_likelihood("y", X=X, y=Y, noise=sigma)
        
        mp = pm.find_MAP()

        return model