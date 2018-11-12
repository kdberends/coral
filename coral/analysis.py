""" High-level objects & functions """

# =============================================================================
# Imports
# =============================================================================

import sys, os
import json
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset
from scipy import stats
from coral.containers import ParameterContainer, RegressionResults
from coral import reglib, utils, settings
from coral.statsfunc import invboxcox, get_empirical_cdf, empirical_ppf
from coral import settings
import matplotlib.pyplot as plt
import random 

if (sys.version_info > (3, 0)):
    from io import StringIO
else:
    from cStringIO import StringIO

# =============================================================================
# Main function
# =============================================================================


def uncertainty_estimation(predictor, response, subsample, **kwargs):
    """
    Readme
    predictor, response: pandas dataframe with:
    column = location
    index = ensemble indices

    """
    # get Logger
    logger = utils.get_logger(name=str(random.getrandbits(128)))

    # Flags
    isMultivariable = False
    locations = [1]

    # Input check
    if predictor.ndim == 2:
        isMultivariable = True
        locations = np.array(predictor.columns)

    # Data
    predictor = predictor.values
    response = response.values

    # Set parameters
    # ------------------
    params = ParameterContainer()
    for (key, value) in kwargs.items():
        setattr(params, key, kwargs.get(key, value))

    logger.info(params)

    # Prepare output folder
    if not(os.path.isdir(params.outputpath)):
        os.mkdir(params.outputpath)

    # Data transformation
    # =============================================================================
    if params.logtransform:
        predictor = np.log10(predictor)
        response = np.log10(response)
    if params.origintransform:
        dt_shift = np.min(predictor, axis=0)
        predictor = predictor - dt_shift
        response = response - dt_shift

    # Function for reverse transformation
    def rev_dt(data, iloc=None):
            if params.origintransform:
                if iloc is None:
                    data = data + dt_shift
                else:
                    data = data + dt_shift[iloc]

            if params.logtransform:
                data = 10 ** data
            return data

    # Subsample
    # =============================================================================
    logger.info('Subsample size is {}'.format(len(subsample)))

    predictor_sample = predictor[subsample]
    response_sample = response[subsample]

    # Regression
    # =============================================================================
    if isMultivariable:
        response_matrix = []
        for i, dc in enumerate(locations):
            logger.info("Now At Location: {}".format(dc))
            response_modelled_transf, trace_summary = params.statmodel(predictor_sample[:, i],
                                                               response_sample[:, i],
                                                               predictor[:, i],
                                                               params)
            # save trace_summary to file (json)
            data_out = os.path.join(params.tracesumpath, '{}_{}.json'.format(params.name, dc))
            with open(data_out, 'w') as f:
                json.dump(trace_summary, f, sort_keys=True, indent=4)


            # Reverse transformation of modelled response     
            response_modelled = np.array([rev_dt(rm, i) for rm in response_modelled_transf])

            response_matrix.append(response_modelled)

    else:
        response_modelled_transf, trace_summary = params.statmodel(predictor_sample,
                                                           response_sample,
                                                           predictor,
                                                           params)
        
        # Reverse transformation of modelled response            
        response_modelled = np.array([rev_dt(rm) for rm in response_modelled_transf])
    
    # Reverse transformation of predictor and response
    [predictor, response] = map(rev_dt, [predictor, response])
    
    if isMultivariable:
        response_modelled = np.array(response_matrix)
    
    return RegressionResults(predictor=predictor,
                             response=response,
                             subsample=subsample,
                             response_modelled=response_modelled,
                             params=params,
                             locations=locations)


