.. image:: coralfig.svg
    :width: 400px
    :align: center
    :alt: coralfig



*coral* is a Python package for efficient uncertainty analysis of model output. The efficiency comes from leveraging a second model, whose output is correlated with the model you are interested in. This may sound like a lot of work, but it can lead to significantly faster uncertainty quantification. And for some applications (like effect studies), it does not actually require any extra modelling work. 

Example usage:

- effect / impact studies. (ADD REFERENCE + notebook)
- multifidelity methods (ADD REFERENCE + notebook)

What's in a name
===============================================================================
Coral stands for 'correlated output regression analysis'. Coral is also fish-eggs, which with some imagination can be thought of as complicated 3D dotty plots. And dotty plots are the real stuff of science, aren't it. 

Coral is based on an earlier 'mfps' package [#r1]_ and can be seen as the successor to that code. Changes include a refactor to Python 3.6+, support for non-linear response, more analysis tools and some functionality tests. 

Development is fully done within the research programme RiverCare_ (project F1), at the University of Twente. 

Todo
===============================================================================

- hook up to travis for testing
- hook up to coveralls
- hook up to rtd
- add example/how-to-use/notebooks


References
===============================================================================


.. rubric:: Footnotes

.. [#r1] Berends, K.D., Warmink, J.J., Hulscher, S.J.M.H., 2018, Efficient uncertainty quantification for impact analysis of human interventions in rivers, Environmental Modelling & Software 107, 50-58. doi: https://doi.org/10.1016/j.envsoft.2018.05.021 

:: _RiverCare: https://kbase.ncr-web.org/rivercare