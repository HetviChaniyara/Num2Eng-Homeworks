import numpy as np
import Galerkin_template as program

## Tests for b)
# checks the Galerkin matrix 
def test_get_galerkin_matrix():
    n = 5
    # choose constant derivatives of test functions
    dv = lambda i, x : x*0 + i 

    A = program.get_galerkin_matrix(n, dv)
    A_true = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            A_true[i,j] = 2*(i+1)*(j+1)
            
    assert np.all(np.isclose(A,A_true))

# checks the right hand side 
def test_getrhs():
    n = 5
    # choose constant f
    f = lambda x : x*0 + 3
    # choose constant test functions
    v = lambda i, x: x*0 + i 

    b = program.get_rhs(f, v, n)
    b_true = np.zeros(n)
    for i in range(n):
        b_true[i] = 6*(i + 1)
    
    assert np.all(np.isclose(b,b_true))

## Test for c)
# checks FE- solution of Poisson equation for rhs f(x) = 6*x 
def test_my_driver():
    testproblem = {}
    testproblem["xL"] = -1.0
    testproblem["xR"] =  1.0
    testproblem["f"] = lambda x: 6*x
    testproblem["uexact"] = lambda x : (1-x**2)*x
    parameters = program.define_default_parameters()
    n = 3
    x = np.linspace(-1., 1., 100)
    errL2 = program.my_driver(testproblem, parameters, n, x)

    assert(np.isclose(errL2, 0))

