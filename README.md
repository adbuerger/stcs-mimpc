stcs-mimpc
==========

This repository provides a demo implementation of a mixed-integer nonlinear Model Predictive Control (MPC) algorithm for the Solar Thermal Climate System (STCS) at Karlsruhe University of Applied Sciences. Both the MPC and the Moving Horizon Estimation (MHE) algorithms can be run for a set of generic input data.


Prerequesites
=============

Python version >= 3.5 is required. The Python packages required for running the algorithms are listed in the file `requirements.txt`. In addition to these, [pycombina](https://github.com/adbuerger/pycombina) is required. The packages listed in `requirements_extra.txt` are required for plotting.

Prior to the first run, for some parts of the algorithms, C code needs to be automatically generated and compiled. For this to work, either `clang` or `gcc` must be available.

The implementations have been tested on a system running Debian 10.


Preparation
===========

Prior to the first run, the file `nlpsetup.py` must be executed once by running

```
python3 nlpsetup.py
```

from within the containing directory. This will generate C code and compile two libraries that contain the Non-Linear Programs (NLPs) for the MPC and MHE routines.


Running the algorithms
======================

The MPC algorithm can be started by running

```
python3 mpc.py
```

and the MHE algorithm by running 

```
python3 mhe.py
```

from within the containing directory.


Structure of the software implementation
========================================

This section is a work in progess.


