"""
Numerical Mathematics for Engineers II WS 25/26
Homework 04 Exercise 4.2
2D Poisson equation with non-homogeneous Dirichlet boundary conditions - Template
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

    # xL, xR, yB, yT : domain [xL,xR] x [yB,yT]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0
    testproblem["yB"] = 0.0
    testproblem["yT"] = 1.0

    # choice
    testproblem["choice"] = testfunction
    
    # right-hand-side function and exact solution
    if (testfunction == "poly"):
        testproblem["f"] = lambda x, y: -6.0 * x**3.0 * y - 6.0 * x * y**3 + 12.0 * x * y
        testproblem["uexact"] = lambda x, y: x * y - x * y**3 - x**3 * y + x**3 * y**3
        testproblem["gL"] = lambda y: 0*y
        testproblem["gR"] = lambda y: 0*y
        testproblem["gT"] = lambda x: 0*x
        testproblem["gB"] = lambda x: 0*x

    elif (testfunction == "sin"):
        testproblem["f"] = lambda x, y: 10.0 * np.pi**2 * np.sin(3.0 * np.pi * x) * np.sin(np.pi * y)
        testproblem["uexact"] = lambda x, y: np.sin(3. * np.pi * x) * np.sin(np.pi * y)
        testproblem["gL"] = lambda y: 0*y
        testproblem["gR"] = lambda y: 0*y
        testproblem["gT"] = lambda x: 0*x
        testproblem["gB"] = lambda x: 0*x

    elif (testfunction == "sin_bc"):
        testproblem["f"] = lambda x, y: np.pi**2/2 * np.sin(np.pi/2*(x+1)) * np.sin(np.pi/2*(y+1))
        testproblem["uexact"] = lambda x, y: np.sin(np.pi/2*(x+1)) * np.sin(np.pi/2*(y+1))
        testproblem["gL"] = lambda y: np.sin(np.pi/2.*(y+1)) 
        testproblem["gR"] = lambda y: 0*y
        testproblem["gB"] = lambda x: np.sin(np.pi/2.*(x+1))
        testproblem["gT"] = lambda x: 0*x

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
def graph(solution_full, X_full, Y_full, N):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surfaceplot = ax.plot_surface(X_full, Y_full, np.reshape(solution_full, (N+2,N+2)), cmap=cm.jet)

    # Add colorbar to the plot
    fig.colorbar(surfaceplot, shrink=0.7, aspect=20, pad=0.1)

    # Add labels to the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('approximate solution u')

    plt.title(f'Solution Poisson problem', fontweight='bold')
    plt.show()

def get_mesh(N, xL, xR, yB, yT):
    # calculate h: cell length in space
    # Here we assume that there is the same cell length in both directions
    h = (xR - xL) / (N+1)

    # create mesh with boundary points
    x_1D = np.linspace(xL, xR, N+2)
    y_1D = np.linspace(yB, yT, N+2)
    X_full, Y_full = np.meshgrid(x_1D, y_1D)

    # create mesh with inner points only 
    X, Y = np.meshgrid(x_1D[1:-1], y_1D[1:-1])

    return h, X, Y, X_full, Y_full


def get_boundary_separate(h,xL,xR,yB,yT,X,Y,flag):
    if flag == "interior":
        is_boundaryL = lambda x,y: np.isclose(x,xL+h)
        is_boundaryR = lambda x,y: np.isclose(x,xR-h)
        is_boundaryB = lambda x,y: np.isclose(y,yB+h)
        is_boundaryT = lambda x,y: np.isclose(y,yT-h)
    
    if flag == "boundary":
        is_boundaryL = lambda x,y: np.isclose(x,xL)
        is_boundaryR = lambda x,y: np.isclose(x,xR)
        is_boundaryB = lambda x,y: np.isclose(y,yB)
        is_boundaryT = lambda x,y: np.isclose(y,yT)
    
    boundaryL = is_boundaryL(X, Y)
    boundaryR = is_boundaryR(X, Y)
    boundaryT = is_boundaryT(X, Y)
    boundaryB = is_boundaryB(X, Y)

    return boundaryL, boundaryR, boundaryB, boundaryT

def get_boundary_values(h, xL, xR, yB, yT, X_full, Y_full, gL, gR, gB, gT):
    ### TODO 
    x_coords = X_full[0, :] # x-coordinates (0 to 1) for B/T boundaries
    y_coords = Y_full[:, 0] # y-coordinates (0 to 1) for L/R boundaries

    # Evaluate boundary functions gL/gR on y-coordinates (size N+2)
    aL = gL(y_coords) 
    aR = gR(y_coords)

    # Evaluate boundary functions gB/gT on x-coordinates (size N+2)
    aB = gB(x_coords)
    aT = gT(x_coords)
    ### end TODO

    return aL, aR, aB, aT


def get_matrix_rhs(N, h, X, Y, f, xL, xR, yB, yT, aL, aR, aB, aT):
    # Assemble the reduced matrix using the Kronecker product
    S = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format="csr")
    I = scipy.sparse.identity(N)
    matrix = -1 /(h**2) * (scipy.sparse.kron(S, I) + scipy.sparse.kron(I, S))
    
    # Initialize right hand side for the discrete system 
    # Since the Dirichlet B.C are homogeneous, rhs = source
    rhs = f(X, Y)

    # Add the boundary conditions where needed = at points near the boundary
    ### TODO     
    # left boundary (k=1)
    rhs[:, 0] += aL[1:N+1] / (h**2)

    # right boundary (k=N)
    rhs[:, N-1] += aR[1:N+1] / (h**2) 
    
    # bottom boundary (l=1)
    rhs[0, :] += aB[1:N+1] / (h**2)
    
    # top boundary (l=N)
    rhs[N-1, :] += aT[1:N+1] / (h**2) 
    #### end TODO

    # Transform rhs into lexicographically ordered vector
    rhs = np.reshape(rhs, (N**2, 1))

    return matrix, rhs 


def my_driver(testproblem, parameters, N):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    yB = testproblem["yB"]
    yT = testproblem["yT"]
    f = testproblem["f"]
    gL = testproblem["gL"]
    gR = testproblem["gR"]
    gB = testproblem["gB"]
    gT = testproblem["gT"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    # We assume that the domain is square
    assert np.isclose(xR-xL, yT-yB)

    # Grid generation
    h, X, Y, X_full, Y_full = get_mesh(N, xL, xR, yB, yT)

    # Evaluate the boundary functions at the boundary points
    aL, aR, aB, aT = get_boundary_values(h, xL, xR, yB, yT, X_full, Y_full, gL, gR, gB, gT) 

    # Assemble the reduced matrix and the right-hand side 
    matrix, rhs = get_matrix_rhs(N, h, X, Y, f, xL, xR, yB, yT, aL, aR, aB, aT)

    # Create the full solution vector
    solution_full = np.zeros((N+2,N+2))

    # Fill the full solution vector with the inner solution
    solution = scipy.sparse.linalg.spsolve(matrix, rhs)
    is_boundary = lambda x,y: np.isclose(x,xL) | np.isclose(x,xR) | np.isclose(y,yB) | np.isclose(y,yT) 
    boundaries = is_boundary(X_full, Y_full)
    inner = ~boundaries 
    solution_full[inner] = solution.reshape(N**2)

    # Fill the full solution vector with the boundary condition
    ## TODO 
    # Left boundary (x=xL, column 0)
    solution_full[:, 0] = aL 
    # Right boundary (x=xR, column N+1)
    solution_full[:, N+1] = aR 
    # Bottom boundary (y=yB, row 0)
    solution_full[0, :] = aB
    # Top boundary (y=yT, row N+1)
    solution_full[N+1, :] = aT
    ## End TODO 

    # plot
    if plot_freq != 0:
        graph(solution_full, X_full, Y_full, N)

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
    # 'sin': sine right-hand-side function with zero bc
    # 'sin_bc': sine right-hand-side function with non-zero bc
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


