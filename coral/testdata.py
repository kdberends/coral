""" testdata for testcases """
import pandas as pd
import numpy as np 
import os

testdata_path = os.path.join(__file__, "../../tests/data")

def linear_univariable(ss=True):
    predictor, response, subsample = __retrieve_linear_data(location="40.00km")
    if ss:
        return predictor, response, subsample
    else:
        return predictor, response

def __retrieve_linear_data(location=None):
    """ Testdata from intervention cases. Univariable (here: at one location)"""
    
    subsample = list(map(int, np.loadtxt('{}/intervention/subsample.csv'.format(testdata_path), delimiter=",")))
    predictor = pd.read_csv('{}/intervention/reference.csv'.format(testdata_path), index_col=0, header=0)
    response = predictor.copy()
    response.iloc[:] = np.nan
    response_data = pd.read_csv('{}/intervention/intervention.csv'.format(testdata_path), index_col=0, header=0)

    for column in response_data.columns:
        response.loc[:, column] = response_data.loc[:, column]

    # Now, select only one station
    if location is None:
        predictor = predictor.T
        response = response.T
    else:
        predictor = predictor.T[location]
        response = response.T[location]

    return predictor, response, subsample


