import numpy as np

import heat_template as program


def test_matrices():
    N = 2
    dt = 0.5
    dx = 2 

    matrix_left, matrix_right = program.create_matrices(N,dt,dx)

    assert np.all(np.isclose(matrix_left.toarray(), np.array([[1.125,-0.0625],[-0.0625,1.125]]))) and \
          np.all(np.isclose(matrix_right.toarray(), np.array([[0.875,0.0625],[0.0625,0.875]])))
    

def test_get_rhs():
    U = np.ones(4)
    dt = 0.5
    dx =  2
    g = np.ones(2)
    matrix_right = np.eye(2)
    rhs = program.get_rhs(matrix_right, U, dt, dx, g)

    assert np.all(np.isclose(rhs, np.array([1.0625,1.0625])))



def test_update_boundaries(): 
    dt = 0.2
    time = 0.8
    U = np.zeros(4)
    g = np.zeros(2)
    testproblem = program.get_testproblem()
    program.update_boundaries(U,g,time,dt,testproblem)

    result = np.array([U[0], U[-1], g[0], g[-1]])
    values = np.array([0, - np.exp(-0.8), 0, - np.exp(-0.8) - np.exp(-0.6)])
    assert np.all(np.isclose(result, values))
    