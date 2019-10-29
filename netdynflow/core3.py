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
Calculation of dynamic communicability and flow
===============================================

This module contains functions to calculate the temporal evolution of the
dynamic communicability and flow. Networks are treated as connectevity
matrices, represented as 2D NumPy arrays. Given a connectivity matrix, dynamic
communicability and flow return a series of matrices arranged into a tensor
(a numpy array of rank-3), each describing the state of the network at
consecutive time points.

Helper functions
----------------
JacobianMOU
    Calculates the Jacobian matrix for the MOU dynamic system.

Generation of main tensors
--------------------------
GenerateTensors
    Generic function to create time evolution of communicability or flow.
DynCom
    Returns the temporal evolution of a network's dynamic communicability.
DynFlow
    Returns the extrinsinc flow on a network over time for a given input.
FullFlow
    Returns the complete flow on a network over time for a given input.
IntrinsicFlow
    Returns the intrinsic flow on a network over time for a given input.

Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Analysis of brain network dynamics estimated
from fMRI data: A new framework based on communicability and flow"
bioRxiv (2018). DOI: https://doi.org/10.1101/421883.


...moduleauthor:: Gorka Zamora-Lopez <galib@zamora-lopez.xyz>

"""
from __future__ import division, print_function

import numpy as np
import numpy.linalg
import scipy.linalg

# __all__ = ['JaccobianMOU', 'DynCom', 'DynFlow', 'FullFlow', 'IntrinsicFlow']


## USEFUL FUNCTIONS ##########################################################
def JacobianMOU(con_matrix, tau_const):
    """Calculates the Jacobian matrix for the MOU dynamic system.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.

    Returns
    -------
    jaccobian : ndarray of rank-2
        The Jaccobian matrix of shape N x N for the MOU dynamical system.
    """
    # 0) SECURITY CHECKS
    # Check the input connectivity matrix
    con_shape = np.shape(con_matrix)
    if len(con_shape) != 2:
        raise ValueError( "con_matrix not a matrix." )
    if con_shape[0] != con_shape[1]:
        raise ValueError( "con_matrix not a square matrix." )
    # Make sure con_matrix is a ndarray of dtype = np.float64
    con_matrix = np.array(con_matrix, dtype=np.float)
    n_nodes = con_shape[0]

    # Check the tau constant, in case it is a 1-dimensional array-like.
    tau_shape = np.shape(tau_const)
    if tau_shape:
        if len(tau_shape) != 1:
            raise ValueError( "tau_const must be either a float or a 1D array." )
        if tau_shape[0] != n_nodes:
            raise ValueError( "con_matrix and tau_const not aligned." )
        # Make sure tau_const is a ndarray of dytpe = np.float64
        tau_const = np.array(tau_const, dtype=np.float)
    else:
        tau_const = tau_const * np.ones(n_nodes, dtype=np.float)

    # 1) CALCULATE THE JACCOBIAN MATRIX
    jacobian_diag = -np.ones(n_nodes, dtype=np.float) / tau_const
    jacobian = np.diag(jacobian_diag) + con_matrix

    return jacobian


## GENERATION OF THE MAIN TENSORS #############################################
def GenerateTensors(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1,
                    normed=True, case='DynCom'):
    ## GORKA: Function probably needs another name.
    """Generic function to create time evolution of communicability or flow.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The covariance matrix of fluctuating inputs.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic flow or communicability.
        A tensor of shape (tmax*timestep) x n_nodes x n_nodes, where n_nodes is
        the number of nodes.
    """
    # 0) SECURITY CHECKS
    caselist = ['DynCom', 'DynFlow', 'FullFlow', 'IntrinsicFlow']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")

    # 1) CALCULATE THE JACOBIAN MATRIX
    jacobian = JacobianMOU(con_matrix, tau_const)
    jacobian_diag = np.diagonal(jacobian)
    n_nodes = len(jacobian)

    # 2) CALCULATE THE DYNAMIC FLOW
    # 2.1) Calculate the extrinsic flow over integration time
    n_t = int(tmax / timestep) + 1
    sigma_sqrt_mat = scipy.linalg.sqrtm(sigma_mat)

    flow_tensor = np.zeros((n_t,n_nodes,n_nodes), dtype=np.float)
    ## GORKA: This may look more elegant or clear because it puts all options
    ## into a single loop over time. But I don't think this is will be efficient.
    ## I bet it is notably slower than the separate loops in GenerateTensors()
    ## of module core2.py.
    for i_t in range(n_t):
        t = i_t * timestep
        if case == 'FullFlow':
            flow_tensor[i_t] = np.dot( sigma_mat, scipy.linalg.expm(jacobian * t) )
        else:
            # Calculate the term for jacobian_diag without using expm(), to speed up
            jacobian_diag_t = np.diag( np.exp(jacobian_diag * t) )
            if case == 'DynCom':
                flow_tensor[i_t] = scipy.linalg.expm(jacobian * t) - jacobian_diag_t
            elif case == 'DynFlow':
                flow_tensor[i_t] = np.dot( sigma_mat, \
                                (scipy.linalg.expm(jacobian * t) - jacobian_diag_t) )
            elif case == 'IntrinsicFlow':
                flow_tensor[i_t] = np.dot( sigma_mat, jacobian_diag_t)

    # 2.2) Normalise by the scaling factor
    if normed:
        scaling_factor = (-1./jacobian_diag).sum()
        flow_tensor /= scaling_factor

    return flow_tensor


## Wrappers using GenerateTensors() ___________________________________________
def DynFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1, normed=True):
    """Returns the extrinsinc flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The covariance matrix of fluctuating inputs.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    dynflow_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x n_nodes x n_nodes, where n_nodes is the number of nodes.
    """
    dynflow_tensor = GenerateTensors(con_matrix, tau_const, sigma_mat, tmax=tmax,
                    timestep=timestep, normed=normed, case='DynFlow')

    return dynflow_tensor

def DynCom(con_matrix, tau_const, tmax=20, timestep=0.1, normed=True):
    """Returns the temporal evolution of a network's dynamic communicability.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    dyncom_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    n_nodes = len(con_matrix)
    sigma_mat = np.identity(n_nodes, dtype=np.float)
    dyncom_tensor = GenerateTensors(con_matrix, tau_const, sigma_mat, tmax=tmax,
                    timestep=timestep, normed=normed, case='DynCom')

    return dyncom_tensor

def IntrinsicFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1, normed=True):
    """Returns the intrinsic flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.dyncom
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    flow_tensor = GenerateTensors(con_matrix, tau_const, sigma_mat, tmax=tmax,
                    timestep=timestep, normed=normed, case='IntrinsicFlow')

    return flow_tensor

def FullFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1, normed=True):
    """Returns the complete flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step. NOT an integration step, but the sampling step.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's flow. A tensor of shape
        (tmax*timestep) x N x N, where N is the number of nodes.
    """
    flow_tensor = GenerateTensors(con_matrix, tau_const, sigma_mat, tmax=tmax,
                            timestep=timestep, normed=normed, case='FullFlow')

    return flow_tensor


##
