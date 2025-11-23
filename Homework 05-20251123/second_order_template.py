"""
Numerical Mathematics for Engineers II WS 25/26
Homework 05 exercise 5.3
General boundary conditions - Template 
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def get_testproblem(testfunction):
    testproblem = {}

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0

    # choice
    testproblem["choice"] = testfunction
    
    # right-hand-side function and exact solution
    if (testfunction == "A"):
        testproblem["g"] = lambda x: x * 0 
        testproblem["f"] = lambda x: x**2 + 2*x - 1
        testproblem["alpha"] = 0.
        testproblem["beta"] = 1.
        testproblem["delta"] = 0.
        testproblem["gamma"] = 1
        testproblem["uexact"] = lambda x: 0*x 

    elif (testfunction == "B"): 
        testproblem["g"] = lambda x: x * 0 + 1 
        testproblem["f"] = lambda x: x**2 - 1
        testproblem["alpha"] = 1.
        testproblem["beta"] = 0.
        testproblem["delta"] = 0.5
        testproblem["gamma"] = 0
        testproblem["uexact"] = lambda x: x**2 + 1

    elif (testfunction == "C"): 
        testproblem["g"] = lambda x: x * 0 
        testproblem["f"] = lambda x: x * 0
        testproblem["alpha"] = 0.
        testproblem["beta"] = 1.
        testproblem["delta"] = 1.
        testproblem["gamma"] = 2.
        testproblem["uexact"] = lambda x: x - 2

    elif (testfunction == "D"): 
        testproblem["g"] = lambda x: x*x 
        testproblem["f"] = lambda x: -6*(1+x) + ((1+x)**3 + 1) * x*x
        testproblem["alpha"] = 1.
        testproblem["beta"] = -1./3.
        testproblem["delta"] = -11/9.
        testproblem["gamma"] = 1.
        testproblem["uexact"] = lambda x: (1+x)**3 + 1.

    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    return testproblem



# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of inner grid points (on coarsest grid)
    parameters["N"] = 80

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


# plot computed and exact solution
def graph(U_comp, x_plot, uexact, xL, xR):
    # evaluate true solution
    U_true = uexact(x_plot)

    plt.figure(1)
    plt.ion()

    plt.figure(1)
    plt.plot(x_plot, U_comp, 'r.', markersize=4, label='computed solution')
    plt.plot(x_plot, U_true, 'k-', label='true solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)
    plt.legend()
    save_str = 'results/solution.png'
    plt.savefig(save_str)

    plt.show()
    plt.pause(2.0)
    plt.clf()


def get_rhs_diag(testproblem,N,x,dx):
    # TODO 
   
    # obtain the parameters from the test problem
    alpha = testproblem["alpha"]
    beta = testproblem["beta"]
    delta = testproblem["delta"]
    gamma = testproblem["gamma"]
    g_func = testproblem["g"]
    f_func = testproblem["f"]
    
    # Initialize components
    rhs = np.zeros(N+2)
    d = np.zeros(N+2)  # diag 
    l = np.zeros(N+1) # diag offset -1
    r = np.zeros(N+1) # diag offset 1
    
    # filling in the interior nodes
    # -u_i-1 + (2h^2g(xi))u_i - u_i+1    
    x_interior = x[1:N+1]
    d[1:N+1] = 2.0 + dx**2 * g_func(x_interior)
    l[:N] = -1.0
    r[1:N+1] = -1.0
    rhs[1:N+1] = dx**2 * f_func(x_interior)


    # left boundary
    d[0] = alpha - beta / dx
    r[0] = beta / dx
    rhs[0] = 1.0

    # right boundary
    d[N+1] = delta + gamma / dx
    l[N] = -gamma / dx
    rhs[N+1] = 1.0
    
    # end TODO
    return rhs, l, d, r

def my_driver(testproblem, parameters, N):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    # Grid generation
    # calculate dx: cell length in space
    dx = (xR - xL) / (N + 1)

    # create mesh with boundary nodes
    x = np.linspace(xL, xR, N + 2)

    rhs, l, d, r = get_rhs_diag(testproblem,N,x,dx)

    # create spare matrix
    matrix = scipy.sparse.csr_matrix(
        scipy.sparse.diags([l, d, r], [-1, 0, 1]))   

    # solve
    full_sol = scipy.sparse.linalg.spsolve(matrix, rhs)  


    # plot
    if plot_freq != 0:
        graph(full_sol, x, uexact, xL, xR)

    # compute error
    true_sol = uexact(x)
    err = full_sol - true_sol
    err_max = np.max(np.abs(err))

    print('Error in Max norm:\t %3.2e\n' % err_max)

    return err_max

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    testfunction = 'D'

    # read in test problem:
    testproblem = get_testproblem(testfunction)

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    errLmax_vec = np.zeros(parameters["Nrefine"] + 1)
    N_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different grid sizes
    N = parameters["N"]
    for k in range(parameters["Nrefine"] + 1):
        errLmax = my_driver(testproblem, parameters, N)

        N_vec[k] = N
        errLmax_vec[k] = errLmax

        N = N * 2
