"""
Numerical Mathematics for Engineers II WS 25/26
Homework 04 Exercise 4.2
2D Poisson equation with non-homogeneous Dirichlet boundary conditions - Test 
"""


import numpy as np
import Poisson2D_bc_template as program  

def test_get_boundary_values():
    xL, xR, yB, yT = 0,1,0,1
    h, X, Y, X_full, Y_full = program.get_mesh(3, xL, xR, yB, yT)
    gL = lambda y: y
    gR = lambda y: 1-y
    gB = lambda x: x
    gT = lambda x: 1-x
    aL, aR, aB, aT = program.get_boundary_values(h, xL, xR, yB, yT, X_full, Y_full, gL, gR, gB, gT)
    assert aL[0]==gL(xL) and aR[1] == gR(h) and aB[-2]==gB(xR-h) and aT[-1]==gT(xR)
    

def test_get_matrix_rhs(): 
    xL, xR, yB, yT = 0,1,0,1
    h, X, Y, _, _ = program.get_mesh(3, xL, xR, yB, yT)
    f = lambda x,y : 0*x + 0*y 
    aL = np.array([1, 0.1, 0.2, 0.3, 1])
    aR = np.array([2,   1,   2,   3, 2])
    aB = np.array([3,  10,  20,  30, 3])
    aT = np.array([4, 100, 200, 300, 4])
    _, rhs = program.get_matrix_rhs(3, h, X, Y, f, xL, xR, yB, yT, aL, aR, aB, aT)
    print(rhs)

    result = np.array([161.6, 320, 496, 3.2, 0, 32, 1604.8, 3200, 4848])

    assert np.allclose(rhs.reshape(9), result.reshape(9))


def test_driver():
    testfunction = 'poly'
    testproblem = program.get_testproblem(testfunction)
    parameters = program.define_default_parameters()
    parameters["plot_freq"]=0

    error = program.my_driver(testproblem, parameters, 5)
    assert np.isclose(error, 0.)