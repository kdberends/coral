""" Regression library """

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
# Regression functions
# ============================================================================= 


def general_linear(predictor_sample, response_sample, predictor, params):
    """
    General linear model of the form y=ax + b + e
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
