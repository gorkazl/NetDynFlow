# -*- coding: utf-8 -*-
# Copyright (c) 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
# <galib@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Network Dynamic Communicability and Flow
========================================

A package to study complex networks based on the temporal evolution of their
Dynamic Communicability and Flow.
Compatible with Python 2.7 and 3.X.

- 'Dynamic Flow' characterises the transient network response over time, as the
network dynamics relax towards their resting-state after a pulse perturbation
(either independent or correlated Gaussian noise) has been applied to selected
nodes.
- 'Dynamic Communicability' corresponds to the special case where uncorrelated
Gaussian noise has initially been applied to all nodes.

NetDynFlow treats networks as connectivity matrices, represented as 2D NumPy
arrays. Given (i) a connectivity matrix and (ii) a an input covariance matrix,
dynamic communicability and flow return a set of matrices (a 3D NumPy array),
each describing the state of the state of the pairwise node interations at
at consecutive time points.

Reference and Citation
**********************

1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).

2. M. Gilson, N. Kouvaris, et al. "Analysis of brain network dynamics estimated
from fMRI data: A new framework based on communicability and flow"
bioRxiv (2018). DOI: https://doi.org/10.1101/421883.

Available functions
*******************

The package is organised into two modules:

core.py
    Functions to obtain the temporal evolution of dynamic communicability and flow.
metrics.py
    Network descriptors to analyse the temporal evolution of the dynamic
    communicability and flow.

To see the list of all functions available use the standard help in an
interactive session, for both modules ::

>>> import netdynflow as ndf
>>> ndf.core?
>>> ndf.metrics?

    **NOTE:**
    Importing NetDynflow brings all functions in the two modules into its
    local namespace. Thus, functions in each of the two modules are called as
    `ndf.func()` instead of `ndf.core.func()` or `ndf.metrics.func()`. Details
    of each function is also found using the usual help, e.g.,

	>>> ndf.DynCom?
	>>> ndf.Diversity?

In an IPython interactive session, or in a Jupyter Notebook, typing ``netdynflow``
and then pressing <tab> will show all the modules and functions available in
the package.

Using NetDynFlow
^^^^^^^^^^^^^^^^
Since NetDynFlow depends on NumPy, it is recommended to import NumPy first,
although this is not necessary for loading the package: ::

>>> import numpy as np
>>> import netdynflow as ndf

    **Note**:
    Importing netdynflow imports also all functions in module *core.py*
    into its namespace. Module *metrics.py* is imported separately. Therefore,
    if the import is relative those functions can be called as, e.g.,  ::

    >>> import netdynflow as ndf
    >>> ...
    >>> dyncom = ndf.DynCom(net, tau)

We did not have to call ``netdynflow.core.DynCom()``. In the case of an absolute
import (using an asterisk ``*``) all functions in *core.py* are imported to the
base namespace:  ::

    >>> from netdynflow import *
    >>> ...
    >>> dyncom = DynCom(net, tau)

Getting started
***************
Create a simple weighted network of N = 4 nodes (a numpy array) and compute its
dynamic communicability over time: ::

>>> net = np.array(((0, 1.2, 0, 0),
                    (0, 0, 1.1, 0),
                    (0, 0, 0, 0.7),
                    (1.0, 0, 0, 0)), float)

>>> tau = 0.8
>>> dyncom = ndf.DynCom(net, tau, tmax=15, timestep=0.01)

The resulting variable ``dyncom`` is an array of rank-3 with dimensions
((tmax / tstep) x N x N) containing tmax / tstep = 1500 matrices of size 4 x 4,
each describing the state of the network at a given time step.

    **NOTE**:
    NetDynFlow employs the convention in graph theory that rows of the
    connectivity matrix encode the outputs of the node. That is, `net[i,j] = 1`
    implies that the node in row ``i`` projects over the node in column ``j``.

Now we calculate the *total communicability* and the ``diversity`` of the
network over time as:  ::

>>> totalcom = ndf.TotalEvolution(dyncom)
>>> divers = ndf.Diversity(dyncom)

``totalcom`` and ``divers`` are two arrays of length (tmax / tsteps) = 1500.


License
-------
Copyright (c) 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris

Released under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

or see the LICENSE.txt file.

"""
from __future__ import absolute_import

# Branch to test different options to organise the core.py module
from . import core
from . import core2
from . import core3
# from .core import *

from . import metrics
from .metrics import *


__author__ = "Gorka Zamora-Lopez, Matthieu Gilson and Nikos E. Kouvaris"
__email__ = "galib@Zamora-Lopez.xyz"
__copyright__ = "Copyright 2019"
__license__ = "Apache License version 2.0"
__version__ = "1.0.xx"
__update__="26/10/2019"


#
