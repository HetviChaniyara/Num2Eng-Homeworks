"""
Numerical Mathematics for Engineers II WS 25/26
Homework 08 Exercise 8.2
Helper functions for the 1D FEM -- template
"""
import numpy as np

def get_M_ref():
    """
    returns the ref. mass matrix
    """
    ### TODO
    M  = np.array([[1/3, 1/6], [1/6, 1/3]])
    return M


def get_S_ref():
    """
    returns the ref. stiffness matrix
    """
    ### TODO
    S = np.array([[1, -1],[-1,1]])
    return S

def get_M_loc(Fdet):
    """
    returns the local mass matrix
    """
    ### TODO
    M_loc = Fdet * get_M_ref()
    return M_loc

def get_S_loc(Fdet, Finv):
    """
    returns the local stiffness matrix
    """
    ### TODO
    S_loc = Finv * get_S_ref()
    return S_loc

def get_trafo(k, e, xh):
    """
    returns (Fdet, Finv) for the specified element transformation Tk
    """
    ### TODO

    # getting indices of nodes for element k
    node_left_idx = e[k, 0]
    node_right_idx = e[k, 1]
    hk = xh[node_right_idx] - xh[node_left_idx]
    Fdet = hk
    Finv = 1/hk 

    return Fdet, Finv

def get_elements(xh):
    """
    returns the number of elements n_e, nodes n_p and the index arrays e
    """
    n_p = len(xh)   # nb of nodes without boundary nodes due to bc
    n_e = n_p - 1   # nb of elements

    # construction of the index arrays e
    e1 = np.arange(0, n_p-1)   # left vertex
    e2 = np.arange(1, n_p)   # right vertex

    e = np.vstack((e1, e2)).T

    return (n_e, n_p, e)