"""
Numerical Mathematics for Engineers II WS 25/26
Homework 08 Exercise 8.2
Helper functions for the 1D FEM -- solution
"""
import numpy as np


def get_S_ref():
    """
    returns the ref. stiffness matrix
    """
    Sxx = 0.5 * np.matrix('1 -1 0; -1 1 0; 0 0 0')
    Sxy = 0.5 * np.matrix('2 -1 -1; -1 0 1; -1 1 0')
    Syy = 0.5 * np.matrix('1 0 -1; 0 0 0; -1 0 1')
    return Sxx, Sxy, Syy

def get_S_loc(d_l,  b_l_11, b_l_12, b_l_22 ):
    """
    returns the local stiffness matrix
    """

    Sxx, Sxy, Syy = get_S_ref()

    # Assemble the local stiffness matrix of the current triangle (here, we do not multiply by
    # d_l, but we divide by it since the calculation of the entries b_l_11 etc. already involved
    # a factor of d_l**2)
    S_l = np.asarray((b_l_11 * Sxx + b_l_22 * Syy + b_l_12 * Sxy) / d_l)
    
    return S_l

def get_trafo(Bkl):
    """
    returns (Fdet, Finv) for the specified element transformation Tk
    """
    # Compute the determinant of the matrix F_l which is part of the affine linear transformation
    # T_l from the reference triangle to the lth triangle (analytic computation of the
    # determinant is preferred here for performance reasons)
    d_l = np.abs((Bkl[1, 0] - Bkl[0, 0]) * (Bkl[2, 1] - Bkl[0, 1]) - (Bkl[2, 0] - Bkl[0, 0]) * (Bkl[1, 1] - Bkl[0, 1]))

    # Compute the entries of the matrix d_l**2*B_l with B_l = inv(F_l)*inv(F_l.T) (analytic
    # computation is preferred here for performance reasons)
    b_l_11 = (Bkl[2, 0] - Bkl[0, 0])**2 + (Bkl[2, 1] - Bkl[0, 1])**2
    b_l_12 = -(Bkl[2, 0] - Bkl[0, 0]) * (Bkl[1, 0] - Bkl[0, 0]) - (Bkl[2, 1] - Bkl[0, 1]) * (Bkl[1, 1] - Bkl[0, 1])
    b_l_22 = (Bkl[1, 0] - Bkl[0, 0])**2 + (Bkl[1, 1] - Bkl[0, 1])**2
    
    return d_l, b_l_11, b_l_12, b_l_22 

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