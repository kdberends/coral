""" High-level objects & functions """

# =============================================================================
# Imports
# =============================================================================

import sys, os
import json
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from coral.statsfunc import get_empirical_cdf, empirical_ppf
from coral import reglib, utils

# =============================================================================
# Main
# =============================================================================

class ParameterContainer:
    """ Custom class for default parameters """
    def __init__(self):
        self.outputpath = os.getcwd()
        self.statmodel = reglib.gaussian_process
        self.inference = "map"
        self.ppc_draws = 2000
        self.logtransform = False
        self.origintransform = False
        self.draws = 10000
        self.burn_in = 5000
        self.cores = 2
        self.chains = 2
        self.confidence_interval = 95
        self.tracesumpath = os.getcwd()
        self.name = 'noname'

    def __str__(self):
        outputstring = "Parameter values \n"
        for prop, value in self.__dict__.items():
            outputstring += "{}: {}\n".format(prop, value)
        return outputstring


class RegressionResults:
    """ Read/Write/Plot results of regression analysis"""
    def __init__(self, predictor=None, response=None, subsample=None, 
                       response_modelled=None, locations=None, params=None):
        self.predictor = predictor
        self.response = response
        self.response_modelled = response_modelled
        self.subsample = subsample
        self.locations = locations
        self.logger = utils.get_logger()
        self.cdfs = {'predictor': [], 'response': [], 'diff': []}

        self.colormap = 'PuBu'
        if params is None:
            params = ParameterContainer()
        self.params = params 

    def save_to_file(self, filename='results.nc'):
        """
        save output to ... netcdf

        Will overwrite any file in its path
        """
        filepath = os.path.join(self.params.outputpath, filename)
        mode = "w"
        writeCDF = False
        with Dataset(filepath, mode, format="NETCDF4") as f:
            if self.response_modelled.ndim > 2:
                ppc = self.response_modelled.shape[1]
            else:
                ppc = self.response_modelled.shape[0]
            print('Full sample size: {}\nSubsample size: {}\n Locations: {}\n Posterior draws:{}'.format(
                self.predictor.shape[0], len(self.subsample), len(self.locations), ppc))
            print("RESPMMOD: {}".format(self.response_modelled.shape))
            
            f.createDimension('sample_index', self.predictor.shape[0])
            f.createDimension('subsample_size', len(self.subsample))
            f.createDimension('nlocations', len(self.locations))
            f.createDimension('ppc_index', ppc)
            f.createDimension('cdf_probabilities', 100)
            f.createDimension('cdf_values', 100)
            f.createDimension('cdf_ci', 3)  # 2.5, 50, 97.5

            # the indices of the full sample 
            subs = f.createVariable('subsample', "i8", ("subsample_size"))

            # Predictor results at samples
            pred = f.createVariable('predictor', "f8", ("sample_index",
                                                        "nlocations"))

            # Response results at samples 
            #(note: only subsample location will have values, otherwise NaN)
            resp = f.createVariable('response', "f8", ("sample_index",
                                                       "nlocations"))

            # Locations
            locs = f.createVariable('locations', "f8", ("nlocations"))

            # Modelled response for full set
            respm = f.createVariable('response_modelled', "f8", ("nlocations",
                                                                 "ppc_index",
                                                                 "sample_index"))

            # cdfs (optional)
            if writeCDF:
                cdf_pred = f.createVariable('cdf_predictor',"f8", ("location",
                                                                   "cdf_probabilities",
                                                                   "cdf_values"))
                cdf_resp = f.createVariable('cdf_predictor',"f8", ("location",
                                                                   "cdf_probabilities",
                                                                   "cdf_values",
                                                                   'cdf_ci'))
                cdf_diff = f.createVariable('cdf_predictor',"f8", ("location",
                                                                   "cdf_probabilities",
                                                                   "cdf_values",
                                                                   'cdf_ci'))


            # Write data to arrays
            # ----------------------------------
            subs[:] = self.subsample
            pred[:] = self.predictor
            resp[:] = self.response
            respm[:] = self.response_modelled
            locs[:] = self.locations

            if writeCDF:
                cdf_pred[:] = self.cdfs['predictor']
                cdf_resp_m[:] = self.cdfs['response']
                cdf_diff_m[:] = self.cdfs['response']

    def load(self, filename='results.nc'):
        ncfilepath = os.path.join(self.params.outputpath, filename)

        with Dataset(ncfilepath, "r", format="NETCDF4") as f:
            self.predictor = f.variables['predictor'][:]
            self.response = f.variables['response'][:]
            self.subsample = f.variables['subsample'][:]
            self.locations = f.variables['locations'][:]
            self.response_modelled = f.variables['response_modelled'][:]

        print("Data loaded succesfully from: {}\n".format(ncfilepath))


