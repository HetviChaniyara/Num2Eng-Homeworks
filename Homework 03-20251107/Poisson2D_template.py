"""
Numerical Mathematics for Engineers II WS 25/26
Homework 03 Exercise 1.3
2D Poisson equation - Template 
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
import os
from matplotlib import cm

# Define the test problem:
# only -u''(x) = f(x)


def get_testproblem(testfunction):
    testproblem = {}

    # xL, xR, yL, yR : domain [xL,xR] x [yL,yR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0
    testproblem["yL"] = 0.0
    testproblem["yR"] = 1.0

    # choice
    testproblem["choice"] = testfunction
    
    # right-hand-side function and exact solution
    if (testfunction == "poly"):
        testproblem["f"] = lambda x, y: -6.0 * x**3.0 * y - 6.0 * x * y**3 + 12.0 * x * y
        testproblem["uexact"] = lambda x, y: x * y - x * y**3 - x**3 * y + x**3 * y**3
    elif (testfunction == "sin"):
        testproblem["f"] = lambda x, y: 10.0 * np.pi**2 * np.sin(3.0 * np.pi * x) * np.sin(np.pi * y)
        testproblem["uexact"] = lambda x, y: np.sin(3. * np.pi * x) * np.sin(np.pi * y)
    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    return testproblem



# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 4

    # N: number of inner grid points (on coarsest grid)
    parameters["N"] = 40

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters

# plot computed and exact solution
def graph(solution_full, Xfull, Yfull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surfaceplot = ax.plot_surface(Xfull, Yfull, np.reshape(solution_full, (N+2,N+2)), cmap=cm.jet)

    # Add colorbar to the plot
    fig.colorbar(surfaceplot, shrink=0.7, aspect=20, pad=0.1)

    # Add labels to the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('approximate solution u')

    plt.title(f'Solution Poisson problem', fontweight='bold')
    plt.show()

def get_mesh(N, xL, xR, yL, yR): 
    ### TODO
    h = (xR-xL)/(N+1)

    x_full_1D = np.linspace(xL, xR, N+2)
    y_full_1D = np.linspace(yL, yR, N+2)

    X_full, Y_full = np.meshgrid(x_full_1D, y_full_1D)

    X,Y= np.meshgrid(x_full_1D[1:-1], y_full_1D[1:-1])

    return h, X, Y, X_full, Y_full

    ### END TODO


def get_matrix_rhs(N, h, X, Y, f):
    ### TODO 
    S = scipy.sparse.diags([1,-2,1], [-1,0,1], shape=(N,N))
    S = (-1/(h**2)) * S
    I_N = scipy.sparse.identity(N)

    A = scipy.sparse.kron(I_N, S) + scipy.sparse.kron(S, I_N)
    rhs = f(X,Y).flatten()

    return A, rhs
    ### END TODO

def get_indexing(xL,xR,yL,yR,X_full,Y_full):
    ### TODO 
    is_boundary = lambda x, y: (np.isclose(x, xL) | np.isclose(x, xR) |
                                np.isclose(y, yL) | np.isclose(y, yR))
    boundary = is_boundary(X_full, Y_full)
    inner = ~ boundary

    return inner, boundary
    ### END TODO


def my_driver(testproblem, parameters, N):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    yL = testproblem["yL"]
    yR = testproblem["yR"]
    f = testproblem["f"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    # We assume that the domain is square
    assert np.isclose(xR-xL, yR-yL)

    # Grid generation
    h, X, Y, X_full, Y_full = get_mesh(N, xL, xR, yL, yR)

    # Assemble the reduced matrix and the right-hand side 
    matrix, rhs =  get_matrix_rhs(N, h, X, Y, f)

    # Create the full solution vector a
    solution_full = np.zeros((N+2,N+2))

    # Solve the reduced problem
    ## TODO
    solution_reduced = scipy.sparse.linalg.spsolve(matrix, rhs)
    
    # Fill the full solution vector with the boundary conditions
    ## TODO
    inner_mask, boundary_mask = get_indexing(xL, xR, yL, yR, X_full, Y_full)
    solution_full[inner_mask] = solution_reduced
    solution_full[boundary_mask] = 0.0


    # plot
    if plot_freq != 0:
        graph(solution_full, X_full, Y_full)

    # compute error
    true_sol = uexact(X_full,Y_full)
    err_max = np.max(np.abs(solution_full - true_sol))

    print('Error in Max norm:\t %3.2e\n' % err_max)

    return err_max

# Main file for solving the 2D Poisson equation -u''(x) = f(x)
# With homogeneous Dirichlet B.C. 

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    #
    # Options:
    # 'poly': polynomial right-hand-side function
    # 'sin': sine right-hand-side function
    testfunction = 'poly'

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

    # plot convergence curve 
    if (parameters["Nrefine"]>0):
        plt.loglog(N_vec, errLmax_vec, "x-", label="max error")
        plt.loglog(N_vec, np.power(N_vec,-2), "--", color='black', label="$1/N^2$")
        plt.grid()
        plt.legend()
        plt.xlabel("$N$")
        plt.savefig("results/cv_curve_"+testfunction+".png")
        plt.show()


