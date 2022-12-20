# Comparison of different integration methods in the statistical analysis of supernova and baryon acoustic oscillations

This repository contains the codes to analyze the ΛCDM model using the scipy.integrate.cumtrapz, scipy.integrate.quad and F2py implementations.

In [codes](https://github.com/williamjouse/LCDM/tree/main/codes) directory contains the scripts to plot the distance modulus and BAO angular scale using scipy.integrate.cumtrapz, scipy.integrate.quad and F2py implementations.

- BAO2D.f: F2py integration implementation of BAO
- Data.py: auxiliar python script to load SNeIa and BAO datasets
- LCDM-BAO.py: others implementations of BAO and main script
- LCDM-SN.py: others implementations of SNeIa and main script
- SN.f: F2py integration implementation of SNeIa 

The statistical analysis is achieved using [pymultinest](https://github.com/JohannesBuchner/PyMultiNest) and confidence levels are obtained from [Getdist](https://getdist.readthedocs.io/en/latest/), see the [statistical-codes](https://github.com/williamjouse/LCDM/tree/main/statistical-codes) directory. 

- BAO2D.f: F2py integration implementation of BAO
- Data.py: auxiliar python script to load SNeIa and BAO datasets
- LCDM-run.py: others implementations and main script
- SN.f: F2py integration implementation of SNeIa
- plot.py: script to plot the confidence levels and distributions
- chains/: folder that contains chains computed by pymultinest
- figures/: contains confidence levels and distributions figures

----

We ran the code five times for SNeIa and BAO. The results, parameter mean/error, and Bayesian evidence are in a folder named chains, and the time run we show in the time.txt file. We concluded that cumtrapz ran faster than the others for SNeIa and, F2py for BAO.
