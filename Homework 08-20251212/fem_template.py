"""
Numerical Mathematics for Engineers II WS 25/26
Homework 08 Exercise 8.2
1D FEM - template
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from femlib_template import get_elements, get_M_loc, get_S_loc, get_M_ref, get_S_ref, get_trafo

# Define the test problem:
# only -u''(x) = f(x)

def get_testproblem(testfunction):
    testproblem = {}

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0

    # choice
    testproblem["choice"] = testfunction
    
    # right-hand-side function and exact solution
    if (testfunction == "cst"):
        testproblem["f"] = 1 # constant function 
        testproblem["uexact"] = lambda x: (1 + np.exp(1) - np.exp(1 - x) - np.exp(x)) / (1 + np.exp(1))
    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    return testproblem



# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 5

    # N: dimension of test space (the smallest one)
    parameters["n"] = 5

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


# plot computed and exact solution
def graph(x, uh, uexact, xL, xR, n):

    # evaluate the exact solutions
    U_true = uexact(x)

    plt.figure(1)
    plt.ion()

    plt.figure(1)
    plt.plot(x, uh, 'r.', markersize=4, label='computed solution')
    plt.plot(x, U_true, 'k-', label='true solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)
    plt.legend()
    plt.savefig(f'results/solution{n}.png')

    plt.show()
    plt.pause(2.0)
    plt.clf()


def get_matrix_rhs(n, f):
    ### TODO
    xL, xR = 0.0, 1.0 # boundary conditions
    xh = np.linspace(xL, xR, n + 2) # as we include the boundaries
    
    # use helper function from femlib
    n_e, n_p, e = get_elements(xh)
    
    A_full = np.zeros((n_p, n_p))
    b_full = np.zeros(n_p)
    
    for k in range(n_e):
        # transformation 
        Fdet, Finv = get_trafo(k, e, xh)
        
        # local matrices
        Mk = get_M_loc(Fdet)
        Sk = get_S_loc(Fdet, Finv)
        Ak_loc = Sk + Mk
        
        # global matrix
        nodes = e[k] # gives [node_left_idx, node_right_idx]
    
        # add local matrices into the global positions
        for i in range(2):
            for j in range(2):
                A_full[nodes[i], nodes[j]] += Ak_loc[i, j]
                
        # bi = f * hk / 2 for each node in the element
        b_full[nodes[0]] += f * np.abs(Fdet) / 2.0
        b_full[nodes[1]] += f * np.abs(Fdet) / 2.0

    # solve for internal nodes only
    A_reduced = A_full[1:-1, 1:-1]
    b_reduced = b_full[1:-1]
    
    return A_reduced, b_reduced, xh

def my_driver(testproblem, parameters, n):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    f = testproblem["f"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    A, b, xh = get_matrix_rhs(n, f)
    
    # solve
    # TODO 
    u_internal = np.linalg.solve(A, b)
    # add boundaries
    uh_full = np.zeros(len(xh))
    uh_full[1:-1] = u_internal

    # plot
    if plot_freq != 0 and n in [1,2,5,10,30,40]:
        graph(xh, uh_full, uexact, xL, xR,n)

    # error 
    errmax = np.max(np.abs(uh_full - uexact(xh).flatten()))
    return errmax 


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    # Options:
    testfunction = 'cst'

    # read in test problem:
    testproblem = get_testproblem(testfunction)

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    errLmax_vec = np.zeros(parameters["Nrefine"] + 1)
    t_solution_vec = np.zeros(parameters["Nrefine"] + 1)
    n_vec = np.zeros(parameters["Nrefine"] + 1)
    errmax_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different space dimensions 
    n = parameters["n"]
    for k in range(parameters["Nrefine"] + 1):
        errmax = my_driver(testproblem, parameters, n)

        n_vec[k] = n
        errmax_vec[k] = errmax

        n = n*2

    if parameters["Nrefine"] > 0:
        plt.semilogy(n_vec, errmax_vec)
        plt.grid()
        plt.xlabel("$n$")
        plt.ylabel("$e_n$")
        plt.savefig(f'results/error.png')
        
        plt.show()
        plt.pause(2.0)
        plt.clf()
