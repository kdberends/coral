.. image:: coralfig.svg
    :width: 400px
    :align: center
    :alt: coralfig


**[!!!Note: be aware that the code is currently under development. Namespaces and function names may be subject to change. Do get in touch if you want to use it :) !!!]**

Version: 0.1-alpha 


About
===============================================================================
``coral`` is a Python package for analysis of the output of two correlated models and to leverage such correlation for efficient uncertainty quantification. I have used this method in my PhD research (e.g. [#r1]_) and I'm now trying to clean up the code sufficiently to be useful for others :) :) If you found this helpful or want to help out, let me know!

Why would I use this?
--------------------------------------------------------------------------------
Wouldn't know. But here's why I needed something like this: I work a lot with environmental models in research and consulting work. These models are not only resource intensive (runtime of several hours or days is not uncommon) but also difficult to validate (to tell how accurate they are) [#fn1]_ . One way of dealing with this is to calculate how uncertainties in assumptions we make in setting up these models translates to uncertainty in model output. The go-to method for this is `Monte Carlo similation <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_ --- which is just too expensive to be useful in practice. 
``coral`` provides an alternative for Monte Carlo simulation. 

What's the catch?
--------------------------------------------------------------------------------
``coral`` provides an approximation of Monte Carlo results and is not a universal method. However, for some applications the approximation can be more than adequate (and better than no uncertainty quantification). The greatest catch is that you always need at least two models and that Monte Carlo simulation has to be done with one of them. This might sound like no improvement at all, but can actually lead to significant cost reduction if the 'Monte Carlo' model is smartly chosen. See [#r1]_ for the methodological background. One of the examples of [#r1]_ is implemented in a jupyter notebook (see docs). 


What's in a name
--------------------------------------------------------------------------------
Coral stands for 'correlated output regression analysis'. Coral is also fish-eggs, which with some imagination can be thought of as complicated 3D dotty plots. And dotty plots are the real stuff of science, aren't they. 

Coral is based on an earlier ``mfps`` package [#r1]_ and can be seen as the successor to that code. It is entirely rewritten for Python 3.6+, adds support for non-linear response, more analysis tools and some functionality tests. 

Development is fully done within the research programme RiverCare_ (project F1), at the University of Twente. 


To-do's
--------------------------------------------------------------------------------

- add tutorial/notebooks
- hook up to testing/docs
- more pre-analysis tools
- optimise package loading


Installation
===============================================================================

Quick start
--------------------------------------------------------------------------------
To install, clone the repository to your local machine and use setup.py to install. 


Dependencies
--------------------------------------------------------------------------------
 built on top of the excellent 'PyMC3 <https://github.com/pymc-devs/pymc3>'_. 


Literature
===============================================================================

To cite this package
--------------------------------------------------------------------------------
To cite, please refer to [#r1]_. 

.. rubric:: Footnotes

.. [#fn1] For an excellent introductory write-up I recommend the now-classical Science article by Oreskes et al. [#r2].

References
--------------------------------------------------------------------------------
.. [#r1] Berends, K.D., Warmink, J.J., Hulscher, S.J.M.H., 2018, Efficient uncertainty quantification for impact analysis of human interventions in rivers, Environmental Modelling & Software 107, 50-58. doi: https://doi.org/10.1016/j.envsoft.2018.05.021 

.. [#r2] Oreskes, N., Shrader-Frechtette, K., Belitz, K., 1994, Verification, Validation and Confirmation of Numerical Models in the Earth Sciences, Science, 263, 641-646, doi: https://doi.org/10.1126/science.263.5147.641

.. _RiverCare: https://kbase.ncr-web.org/rivercare
.. _PyMC3: https://docs.pymc.io/