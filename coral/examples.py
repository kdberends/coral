# Provides example data
#
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

import numpy as np 
import pandas as pd 
import reglib

examples = dict()

# Example 01 - Dike relocation (50 km)
# -------------------------------------
predictor = pd.read_csv('example_data/01/reference.csv', index_col=0)
response = pd.read_csv('example_data/01/intervention.csv', index_col=0)
predictor = predictor.T['50.00km'].values
response = response.T['50.00km'].values

regressionfunc = reglib.mcmc_glm
modelresponsefunc = reglib.mtrace_glm

examples['01'] = dict(predictor=predictor,
                      response=response,
                      path='example_data/01',
                      regressionfunc=regressionfunc,
                      modelresponsefunc=modelresponsefunc)

# Example 02 - Vegetation removal (50 km)
# -------------------------------------
predictor = pd.read_csv('example_data/02/reference.csv', index_col=0)
response = pd.read_csv('example_data/02/intervention.csv', index_col=0)
predictor = predictor.T['50.00km'].values
response = response.T['50.00km'].values

regressionfunc = reglib.mcmc_glm
modelresponsefunc = reglib.mtrace_glm

examples['02'] = dict(predictor=predictor,
                      response=response,
                      path='example_data/02',
                      regressionfunc=regressionfunc,
                      modelresponsefunc=modelresponsefunc)