# -*- coding: utf-8 -*-
# Copyright (c) 2021, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
# <galib@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Analysis of dynamic communicability and flow
============================================

Functions to analyse the dynamic communicability or flow "tensors", which have
been previously calculated from a given network.

Metrics derived from the tensors
--------------------------------
TotalEvolution
    Calculates total communicability or flow over time for a network.
NodeEvolution
    Temporal evolution of all nodes' input and output communicability or flow.
Diversity
    Temporal diversity for a networks dynamic communicability or flow.
TimeToPeak
    The time links, nodes or networks need to reach peak flow.
TimeToDecay
    The time pair-wise interaction, nodes or networks need to decay back to zero.
TotalFlow
    The total accumulated flow of pair-wise, nodes or network flow over time.


Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

"""
# Standard library imports
from __future__ import division, print_function
# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg


## METRICS EXTRACTED FROM THE FLOW AND COMMUNICABILITY TENSORS ################
def TotalEvolution(tensor):
    """
    Calculates total communicability or flow over time for a network.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape n_nodes x n_nodes x timesteps , where n_nodes is the number of nodes.

    Returns
    -------
    totaldyncom : ndarray of rank-1
        Array containing temporal evolution of the total communicability.
    """

    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x n_t) expected'

    totaldyncom = tensor.sum(axis=(0,1))

    return totaldyncom

def NodeEvolution(tensor, directed=False):
    """
    Temporal evolution of all nodes' input and output communicability or flow.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape n_nodes x n_nodes x timesteps, where n_nodes is the number of nodes.

    Returns
    -------
    nodedyncom : tuple.
        Temporal evolution of the communicability or flow for all nodes.
        The result consists of a tuple of two ndarrays of shape (n_nodes x timesteps)
        each. The first is for the sum of communicability interactions over all
        inputs of each node and the second for its outputs.
    """
    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x timesteps) expected'

    # 1) Calculate the input and output node properties
    innodedyn = tensor.sum(axis=0)
    outnodedyn = tensor.sum(axis=1)
    nodedyn = ( innodedyn, outnodedyn )

    return nodedyn

def Diversity(tensor):
    """
    Temporal diversity for a networks dynamic communicability or flow.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability or flow. A
        tensor of shape N x N x timesteps, where N is the number of nodes.

    Returns
    -------
    diversity : ndarray of rank-1
        Array containing temporal evolution of the diversity.
    """
    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x n_t) expected'

    diversity = np.zeros(n_t, np.float)
    diversity[0] = np.nan
    for i_t in range(1,n_t):
        temp = tensor[:,:,i_t]
        diversity[i_t] = temp.std() / temp.mean()

    return diversity

def Time2Peak(arr, timestep):
    """
    The time links, nodes or networks need to reach peak flow.

    In terms of binary graphs, time-to-peak is equivalen to the pathlength
    between two nodes.

    The function calculates the time-to-peak for either links, nodes or the
    whole network, depending on the input array given.
    - If 'arr' is the (N x N x nt) tensor flow, the output 'ttp_arr' will be
    an N x N matrix with the ttp between every pair of nodes.
    - If 'arr' is the (N x nt) temporal flow of the N nodes, the output
    'ttp_arr' will be an array of length N, containing the ttp of the N nodes.
    - If 'arr' is the array of length nt for the network flow, then 'ttp_arr'
    will be a scalar, indicating the time at which whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape N X N X nt for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.

    Returns
    -------
    ttp_arr : ndarray of variable rank
        The time(s) taken for links, nodes or the network to reach peak flow.
        Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.
    arr_shape = np.shape(arr)
    if arr_shape==3:
        assert arr_shape[0] == arr_shape[1], \
            'Input not aligned. Shape (n_nodes x n_nodes x n_time) expected'

    # 1) Get the indices at which every element peaks
    ttp_arr = arr.argmax(axis=-1)
    # 2) Convert into simulation time
    ttp_arr = timestep * ttp_arr

    return ttp_arr

def Time2Decay(arr, dt, fraction=0.99):
    """
    The time pair-wise interaction, nodes or networks need to decay back to zero.

    Strictly speaking, this function measures the time that the cumulative
    flow (area under the curve) needs to reach x% of the total (cumulative)
    value. Here 'x%' is controled by the optional parameter 'fraction'.
    For example, 'fraction = 0.99' means the time needed to reach 99%
    of the area under the curve, given a response curve.

    The function calculates the time-to-decay either for all pair-wise
    interactions, for the nodes or for the whole network, depending on the
    input array given.
    - If 'arr' is the (N x N x nt) tensor flow, the output 'ttd_arr' will be
    an N x N matrix with the ttd between every pair of nodes.
    - If 'arr' is the (N x nt) temporal flow of the N nodes, the output
    'ttd_arr' will be an array of length N, containing the ttd of the N nodes.
    - If 'arr' is the array of length nt for the network flow, then 'ttd_arr'
    will be a scalar, indicating the time at which whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape N X N X nt for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    fraction : scalar, optional
        The fraction of the total area-under-the-curve to be reached.
        For example, 'fraction = 0.99' means the time the flow needs to
        reach 99% of the area under the curve.

    Returns
    -------
    ttd_arr : ndarray of variable rank
        The time(s) needed by the flows of pair-wise interactions, nodes or
        the network to decay back to zero. Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve(s) has (have) really decayed back
    ## to zero. At this moment, it is the user's responsability to guarantee
    ## that all the curves have decayed reasonably well.
    ## The check should rise a warning to simulate for longer time.

    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.array(np.shape(arr), np.int)
    if len(arr_shape) == 3:
        assert arr_shape[0] == arr_shape[1], \
            'Input not aligned. Shape (n_nodes x n_nodes x n_time) expected'

    # 1) Set the level of cummulative flow to be reached over time
    targetcflow = fraction * arr.sum(axis=-1)

    # 2) Calculate the time the flow(s) need to decay
    # Initialise the output array, to return the final time-point
    ## NOTE: This version iterates over all the times. This is not necessary.
    ## We could start from the end and save plenty of iterations.
    ttd_shape = arr_shape[:-1]
    nsteps = arr_shape[-1]
    ttd_arr = nsteps * np.ones(ttd_shape, np.int)

    # Iterate over time, calculating the cumulative flow(s)
    cflow = arr[...,0].copy()
    for t in range(1,nsteps):
        cflow += arr[...,t]
        ttd_arr = np.where(cflow < targetcflow, t, ttd_arr)

    # Finally, convert the indices into integration time
    ttd_arr = ttd_arr.astype(np.float) * dt

    return ttd_arr

def TotalFlow(arr, timestep, timespan='full'):
    ## TODO: The name of this function needs good thinking. Different options
    ## are possible depending on the interpretation and naming of other
    ## variables or metrics.
    ## The most explicit would be to call it "AreaUnderCurve()" because that is
    ## exactly what it does. But that doesn't sound very sexy nor hints on the
    ## interpretation of what it measures, given that 'arr' will be the
    ## temporal evolution of a flow, response curve or dyncom ... which are
    ## indeed the same !! Maybe.
    """
    The total accumulated flow of pair-wise, nodes or network flow over time.

    The function calculates the area-under-the-curve for the flow curves over
    time. It does so for all pair-wise interactions, for the nodes or for
    the whole network, depending on the input array given.
    - If 'arr' is the (N x N x nt) flow tensor, the output 'totalflow' will be
    an N x N matrix with the ttp between every pair of nodes.
    - If 'arr' is the (N x nt) temporal flow of the N nodes, the output
    'totalflow' will be an array of length N, containing the ttp of the N nodes.
    - If 'arr' is the array of length nt for the network flow, then 'totalflow'
    will be a scalar, indicating the time at which whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape N X N X nt for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    timespan : string, optional
        If timespan = 'full', the function calculates the area under the
        curve(s) along the whole time span (nt) that 'arr' contains, from t0 = 0
        to tfinal.
        If timespan = 'raise', the function calculates the area-under-the-
        curve from t0 = 0, to the time the flow(s) reach a peak value.
        If timespan = 'decay', it returns the area-under-the-curve for the
        time spanning from the time the flow peaks, until the end of the signal.

    Returns
    -------
    totalflow : ndarray of variable rank
        The accumulated flow (area-under-the-curve) between pairs of nodes,
        by nodes or by the whole network, over a period of time.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.

    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.shape(arr)
    if len(arr_shape) == 3:
        if arr_shape[0] != arr_shape[1]:
            raise ValueError( "Shape (N x N x nt) expected. When 'arr' is a 3D array, first two dimensions must be square. If your 'arr' has shape (nt x N x N), see function tNN2NNt() for reshaping." )

    # Validate options for optional variable 'timespan'
    caselist = ['full', 'raise', 'decay']
    if timespan not in caselist :
        raise ValueError( "Optional parameter 'timespan' requires one of the following values: %s" %str(caselist) )

    # 1) DO THE CALCULATIONS
    # 1.1) Easy case. Integrate area-under-the-curve along whole time interval
    if timespan == 'full':
        totalflow = timestep * arr.sum(axis=-1)

    # 1.2) Integrate area-under-the-curve until or from the peak time
    else:
        # Get the temporal indices at which the flow(s) peak
        tpidx = arr.argmax(axis=-1)

        # Initialise the final array
        tf_shape = arr_shape[:-1]
        totalflow = np.zeros(tf_shape, np.float)

        # Sum the flow(s) over time, only in the desired time interval
        nsteps = arr_shape[-1]
        for t in range(1,nsteps):
            # Check if the flow at time t should be accounted for or ignored
            if timespan == 'raise':
                counts = np.where(t < tpidx, True, False)
            elif timespan == 'decay':
                counts = np.where(t < tpidx, False, True)
            # Sum the flow at the given iteration, if accepted
            totalflow += (counts * arr[...,t])

        # Finally, normalise the integral by the time-step
        totalflow *= timestep

    return totalflow



##
