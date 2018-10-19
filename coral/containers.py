import configparser
import sys
import os
import time
import json
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt
from mflib.statsfunc import get_empirical_cdf, empirical_ppf
import random 



class RegressionResults:
    def __init__(self, predictor=None, response=None, subsample=None, 
                       response_modelled=None, locations=None, params=None):
        self.predictor = predictor
        self.response = response
        self.response_modelled = response_modelled
        self.subsample = subsample
        self.locations = locations
        
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
            # Locations
            locs = f.createVariable('locations', "f8", ("nlocations"))

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

    def _cdf_at_location(self, location=0, diff=False):
        """
        Returns array with cdfs for each location in modelled response
        """
        cdfs = list()
        qtiles = list()

        prob_x = np.linspace(0, 1, 101)
        ci_qtiles = np.abs(np.array([0, 100]) - (100 - self.params.confidence_interval) / 2.)

        # Retrieve a cdf for each modelled response 
        if diff:
            data = self.response_modelled[location] - self.predictor.T[location]
        else:
            data = self.response_modelled[location]

        for rm in data:
            p, val = get_empirical_cdf(rm)
            cdfs.append(np.interp(prob_x,  p, val))

        if not diff:
            p, val = get_empirical_cdf(self.predictor)
            cdf_predictor = np.interp(prob_x,  p, val)

        def get_cdf_intervals(cdfs): 
            """
            For each cdf, return the ci quantiles
            """ 
            qtiles = list()
            for cdf in cdfs:
                qtiles.append(empirical_ppf(ci_qtiles, cdf))
            return qtiles

        cdfs = np.array(cdfs).T
        qtiles = np.array(get_cdf_intervals(cdfs)).T
        cdfmean = np.mean(cdfs, axis=1)

        if diff:
            return prob_x, cdfs, cdfmean, qtiles, None
        else:
            return prob_x, cdfs, cdfmean, qtiles, cdf_predictor

    def calculate_cdf_at_locations(self, location=0, diff=False, overwrite=False):
        """
        Calculate the cdf at given location(s)
        """
        [cdf_mean, cdf_low, cdf_high, cdf_pred] = [np.nan * np.zeros(100, len(self.locations)) for i in range(4)]
        
        for iloc, location in enumerate(tqdm(self.locations)):
            if iloc % every == 20:
                prob_x, cdfs, cdfmean, qtiles, cdf_predictor = self._cdf_at_location(location=iloc, diff=diff)
                cdf_mean[iloc] = cdfmean
                cdf_low[iloc] = qtiles[0]
                cdf_high[iloc] = qtiles[1]
                cdf_pred[iloc] = cdf_predictor

        #[cdf_mean, cdf_low, cdf_high, cdf_pred] = [np.array(l) for l in [cdf_mean, cdf_low, cdf_high, cdf_pred]]
        
        if diff:
            self.cdfs['diff'] = np.array([cdf_mean, cdf_low, cdf_high])
        else:
            self.cdfs['predictor'] = np.array(cdf_predictor)
            self.cdfs['response'] = np.array([cdf_mean, cdf_low, cdf_high])

    def plot_effect_multidim(self, location=None, ax=None, every=1):
        """
        Plot effect (response - predictor) along the location dimension.

        Model uncertainty is visualised as the shaded area, using the expected
        cdf.

        Estimation uncertainty is visualised through a dashed red line, showing
        the ECI at the 0.5 median.
        """
        fig, ax = plt.subplots(1)
        cmap = plt.get_cmap(self.colormap)
        cdflist = list()
        eci = list()
        x = list()
        # Get expected ecdf at each location
        for iloc, location in enumerate(tqdm(self.locations)):
            if iloc % every == 0:
                prob_x, cdfs, cdfmean, qtiles = self._cdf_at_location(location=iloc, diff=True)
                eci.append(qtiles[0][50] - qtiles[1][50])
                cdflist.append(cdfmean)
                x.append(location)

        ilist = [90, 80, 70, 60]
        cdfs = np.array(cdflist)

        for j, i in enumerate(ilist):
            color_val = int(j / len(ilist) * 255)
            ax.fill_between(x, cdfs[:, i], cdfs[:, 100-i], color=cmap(color_val), alpha=0.8, label='')
            ax.plot(x, eci, '--r', label="ECI$_{0.5}$")
        settings.gridbox_style(ax)
        plt.show()

    def plot_effect_at_location(self, location=0, ax=None):
        fig, ax = plt.subplots(1)
        prob_x, cdfs, cdfmean, qtiles, forget = self.cdf_at_location(location=location, diff=True)
        ax.plot(cdfmean, prob_x, '-k', label='Effect (mean)')
        ax.plot(qtiles[0], prob_x, '--k')
        ax.plot(qtiles[1], prob_x, '--k')
        settings.gridbox_style(ax)
        plt.show()

    def plot_regression(self, location=0, ax=None):
        return self.plot_regression_at_location(location=0, ax=ax)

    def plot_regression_at_location(self, location=0, ax=None):
        """
    
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        cmap = plt.get_cmap(self.colormap)

        # The predictor is sorted from low to high. Response is sorted using 
        # same sorting indexes
        isort = np.argsort(self.predictor.flatten())
        samples = [i[isort].flatten() for i in self.response_modelled[location]]
        samples = np.array(samples).T
        x = self.predictor[isort].flatten()
        
        percs = np.linspace(51, 99, 10)
        colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(samples, p, axis=1)
            lower = np.percentile(samples, 100-p, axis=1)
            color_val = colors[i]
            ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=0.8)
        
        ax.plot(x, np.percentile(samples, 2.5, axis=1), '--k', linewidth=0.5)
        ax.plot(x, np.percentile(samples, 97.5, axis=1), '--k', linewidth=0.5)
        ax.plot(self.predictor[self.subsample], self.response[self.subsample], '.k')

        # Identity line
        xlim = ax.get_xlim()

        ax.plot(xlim, xlim, '-', color=[0.7] * 3, linewidth=0.5)
        ax.set_xlabel('Predictor [m]')
        ax.set_ylabel('Response [m]')
        settings.gridbox_style(ax)

        if self.params.logtransform:
            ax.set_xscale("log")
            ax.set_yscale("log")

        plt.show()
        return ax

    def plot_cdf_at_location(self, location=0, ax=None):
        fig, ax = plt.subplots(1)
        prob_x, cdfs, cdfmean, qtiles, cdf_predictor = self.cdf_at_location(location=location)
        ax.plot(cdf_predictor, prob_x, '-r', label='predictor')
        ax.plot(cdfmean, prob_x, '-k', label='Response (mean)')
        ax.plot(qtiles[0], prob_x, '--k')
        ax.plot(qtiles[1], prob_x, '--k')
        settings.gridbox_style(ax)
        plt.show()