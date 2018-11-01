#
# This file: Sampling and sub-sampling function
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

import os
import sys
import numpy as np 
from coral.statsfunc import empirical_ppf, get_empirical_cdf
from SALib.sample import sobol_sequence

# =============================================================================
# Functions
# =============================================================================

def subsample(dataset, 
              subsample_size=10, 
              subsample_method=1, 
              path=os.getcwd(), 
              outputpath=os.getcwd(),
              verbose=True,
              subsample=[]):
    """
    Creates a subsample from a given dataset. If an existing subsample is
    provided, it will either extend the subsample or shrink the subsample to
    given subsample size. 

    Note: in stratified subsample, the returned size of the subsample can be 
    smaller than the given subsample size. This happens if strata do not 
    contain datapoints. 
    
    Arguments:
        subsample_size: int
        subsample_method: 
                        1 - Unweighted stratified
                        2 - Weighted stratified (NOT FULLY IMPLEMENTED)
                        3 - Random subsample (NOT FULLY IMPLEMENTED)
                        4 - No subsample (returns dataset)
        path: str
        outputpath: str
        verbose: bool
        subsample: list

    Returns:
        subsample - list of values
        path to outputfiles
            [dir]/[case]_sample.csv: list of indices
            [dir]/[case]_bins.csv: list of stratified bins

    """

    # Variables
    # -------------------------------------------------------------------------
    bins = list()
    subsample_path = '{}/subsample.csv'.format(path)
    bins_path = '{}/bins.csv'.format(path)
    existing_sample_given = bool(subsample)
    output_subsample = []

    # Methods
    # -------------------------------------------------------------------------
    if subsample_method == 1:
        """
        Unweighted stratified subsampling. Method will draw a single sample from 
        each strata. If a strata does not contain samples, no samples will be drawn.
        """

        # Create bins
        bins = np.linspace(np.min(dataset), np.max(dataset), subsample_size + 1)

        # Make subsample
        for i in range(len(bins) - 1):
            if existing_sample_given: 
                # There already exists a subsample
                where_ss_in_bin = (dataset[np.array(subsample)] >= bins[i]) &\
                                  (dataset[np.array(subsample)] <= bins[i + 1])
                in_bin = np.where(where_ss_in_bin)[0]
                subsample = np.array(subsample)
                in_bin = subsample[in_bin]
            else:
                in_bin = np.argwhere((dataset >= bins[i]) & (dataset <= bins[i + 1])) 

            # pick random from selected (values)
            random_number = int(np.random.uniform(low=0, high=len(in_bin)))
            
            try:
                output_subsample.append(in_bin[random_number])
            except IndexError:
                sys.stdout.write('No values in bin [{} - {}], or bin already filled\n'.format(bins[i], bins[i + 1]))

    elif subsample_method == 2:
        """
        Stratified subsampling (weighted)
        """

        # Crate bins
        p, val = get_empirical_cdf(dataset, method=0)
        bins = np.linspace(0, 100, subsample_size + 1)
        bins_out = []

        # make subsample
        for i in range(len(bins) - 1):
            window = empirical_ppf([bins[i], bins[i + 1]], dataset)
            in_bin = np.argwhere((dataset >= window[0]) & (dataset <= window[1])) 
            
            # pick random from selected
            random_number = int(np.random.uniform(low=0, high=len(in_bin)))
            try:
                subsample.append(in_bin[np.round(random_number)][0])
            except IndexError:
                sys.stdout.write('No values in bin [{} - {}]\n'.format(in_bin[i], in_bin[i + 1]))
            bins_out.append(window[0])

        bins_out.append(window[1])
        bins = bins_out
  
    elif subsample_method == 3:
        """
        Random subsampling
        """
        subsample = np.random.uniform(low=0, 
                                      high=len(dataset), 
                                      size=subsample_size).astype(int)

    elif subsample_method == 4: 
        """
        No subsampling 
        """
        output_subsample = range(len(dataset))

    # output

    np.savetxt(subsample_path, output_subsample, delimiter=",", fmt="%i")
    np.savetxt(bins_path, bins, delimiter=",")
    print ("SUBSAMPLE SAVED TO {}".format(subsample_path))
    return subsample, subsample_path, bins_path

def sample(n, dim=1, cdf=[None], method='random'):
    """
    Samples using chosen method from a distribution

    Arguments:
    n: int, number of samples
    dim: int, number of dimensions
    cdf: [1xd] array of arrays, e.g. [cdf_d1, cdf_d2]. Each cdf should have 
         be either:
            - an empirical cdf of array [values, probabilities]
            - a function, e.g. for normal distribution:
        def norm(sample_in):
            return [stats.norm.ppf(s, loc=0.03, scale=0.002) for s in sample_in]
     
        if no cdf is defined, sampling will be from the uniform [0-1] distribution


    method: str, options:
            'random' - simple random sampling
            'sobol' - quasi random using SAlib's generator   

    returns:
        [nxd] array of sample points 

    """

    # =========================================================
    # Make uniform sample
    # =========================================================

    if method.lower() == 'random':
        unf_sample = np.random.uniform(size=[dim, n])
    elif method.lower() == 'sobol':
        unf_sample = sobol_sequence.sample(n + 1, dim)
        unf_sample = unf_sample[1:]  # skip first row (zeros)
        unf_sample = unf_sample.T

    if len(cdf) != dim:
        cdf = [None] * dim
        print ('Number of defined cdfs unequal to number of dimensions.' + 
               'Sampling from uniform distribution for every dimension')
    
    # =========================================================
    # Transform sample to desired distribution
    # =========================================================
    def unfdummy(sample_in):
        """dummy function for uniform transformation"""
        return sample_in

    def emp(sample_in, empirical_cdf):
        """ Empirical function"""
        return [statsfunc.empirical_ppf(s, empirical_cdf[1], empirical_cdf[0]) for s in sample_in]
    
    samplefuncs = []
    for i in range(dim):
        if callable(cdf[i]):
            samplefuncs.append(cdf[i])
        elif cdf[i] is None:
            samplefuncs.append(unfdummy)
        else:
            samplefuncs.append(empfunc(cdf[i]).run)

    return [samplefunc(unf_sample[i]) for i, samplefunc in enumerate(samplefuncs)]