import numpy as np 
import Poisson_template as program


## Tests for (a)
def test_exact_solution_const():
    testproblem = program.get_testproblem("const")
    points = np.array([0,0.25,0.75,1])
    assert np.all(np.isclose(testproblem["uexact"](points), np.array([0, 3./32., 3/32., 0])))


def test_exact_solution_sin():
    testproblem = program.get_testproblem("sin")
    points = np.array([0,0.25,0.75,1])
    assert np.all(np.isclose(testproblem["uexact"](points), np.array([0, np.sqrt(2)/(2*np.pi**2), np.sqrt(2)/(2*np.pi**2), 0])))

## Test for (b)
# Check the rhs and diagonal
def test_rhs():
    testproblem = program.get_testproblem("const")
    parameters = program.define_default_parameters()
    N = parameters["N"]
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    f = testproblem["f"]

    # Grid generation
    # calculate dx: cell length in space
    dx = (xR - xL) / (N + 1)

    # create mesh with boundary nodes
    x = np.linspace(xL, xR, N + 2)

    rhs, l, d, r = program.get_rhs_diag(f,N,x,dx)

    assert  np.isclose(rhs[0],f(xL + dx)) and np.isclose(rhs[-1], f(xR-dx)) 

    assert l.shape == r.shape and l.shape == (N-1,) and d.shape == (N,)

## Test for (c)
def test_full_matrix(): 
    testproblem = program.get_testproblem("const")
    parameters = program.define_default_parameters()
    N = parameters["N"]
    err_max, _ = program.my_driver("full", testproblem, parameters, N)  
    assert np.isclose(err_max, 0.)


## Test for (e)
def test_sparse_matrix(): 
    testproblem = program.get_testproblem("const")
    parameters = program.define_default_parameters()
    N = parameters["N"]
    err_max, _ = program.my_driver("sparse", testproblem, parameters, N)  
    assert np.isclose(err_max, 0.)