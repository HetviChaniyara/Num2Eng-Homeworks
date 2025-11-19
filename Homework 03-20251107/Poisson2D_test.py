"""
Numerical Mathematics for Engineers II WS 25/26
Homework 03 Exercise 1.3
2D Poisson equation - Test 
"""


import numpy as np
import scipy 
import Poisson2D_template as program  

def test_get_mesh():
    h, X, Y, XFull, YFull = program.get_mesh(7, 0, 1, 0, 1)
    res = np.array([h, X[0,0], XFull[0,0], X[-1,-1], XFull[-1,-1], Y[0,0], YFull[0,0], Y[-1,-1], YFull[-1,-1]])
    res_true = np.array([0.125, 0.125, 0, 0.875, 1, 0.125, 0, 0.875, 1])
    assert np.all(np.isclose(res,res_true)) 


def test_get_matrix_rhs(): 
    h, X, Y, _, _ = program.get_mesh(3, 0, 1, 0, 1)
    f = lambda x,y : x + y 
    matrix, rhs = program.get_matrix_rhs(3, h, X, Y, f)
    test_vec = np.zeros((9,1), dtype=float)
    test_vec[3]=1.0
    res = np.dot(matrix.toarray(),test_vec)
    res_true = np.reshape(np.array([-16,0,0,64,-16,0,-16,0,0]), (9,1))
    assert np.allclose(res, res_true)  and rhs.shape == (9,1)


def test_get_indexing():
    h, X, Y, X_full, Y_full = program.get_mesh(3, 0, 1, 0, 1)
    inner, boundaries = program.get_indexing(0,1,0,1, X_full, Y_full)
    test_m = np.zeros((5,5))
    test_m[inner] = 1 
    test_m[boundaries] = 2

    bc0 = test_m[0,0:5] # bottom boundary
    bc1 = test_m[:,4] # right boundary
    bc2 = test_m[4,0:5] # top boundary
    bc3 = test_m[:,0] # left boundary
    res_b = 2*np.ones(5)

    inner = test_m[1:4,1:4]
    res_i = np.ones((3,3))
    assert np.allclose(bc0,res_b) and np.allclose(bc1,res_b) \
        and np.allclose(bc2,res_b) and np.allclose(bc3,res_b) \
            and np.allclose(inner,res_i)
