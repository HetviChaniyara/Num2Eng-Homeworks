"""
Numerical Mathematics for Engineers II WS 25/26
Homework 06 Exercise 6.3
Theta schemes - Template
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

# Define the test problem:
# only -u''(x) = f(x)


def get_testproblem():
    testproblem = {}

    # xL, xR, yL, yR : domain [xL,xR] x [yL,yR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0
    testproblem["yL"] = 0.0
    testproblem["yR"] = 1.0

    testproblem["t0"] = 0.0
    testproblem["tend"] = 0.05
    testproblem["u0"] = lambda x, y: 0*x + 0*y
    
    # right-hand-side function and exact solution
    testproblem["f"] = lambda x, y: np.where(np.logical_and(0.05<=(x-0.5)**2+(y-0.5)**2,(x-0.5)**2+(y-0.5)**2<=0.07), 1000., 0.)

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # theta parameter for the theta scheme
    parameters["theta"] = 0.

    # N: number of inner grid points (on coarsest grid)
    parameters["N"] = 80

    # N_time: number of time steps
    parameters["N_time"] = 1400

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 10000

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1000

    return parameters


def graph(solution_full, X_full, Y_full, time, theta, done):
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    minTemperature = 0
    maxTemperature = np.max(solution_full)
    ax.pcolor(X_full,Y_full,solution_full,cmap='hot',vmin=minTemperature,vmax=maxTemperature)

    plt.title(f'Solution heat equation at t={time:.3f} for theta={theta:.2f}', fontweight='bold')

    if(done):
        plt.savefig(f"results/heat2D_theta{int(theta*10)}.png")
    plt.show()


def get_mesh(N, xL, xR, yL, yR): 
    # calculate h: cell length in space
    # Here we assume that there is the same cell length in both directions
    h = (xR - xL) / (N+1)

    # create mesh with boundary points
    x_1D = np.linspace(xL, xR, N+2)
    y_1D = np.linspace(yL, yR, N+2)
    X_full, Y_full = np.meshgrid(x_1D, y_1D)

    # create mesh with inner points only 
    X, Y = np.meshgrid(x_1D[1:-1], y_1D[1:-1])

    return h, X, Y, X_full, Y_full


def get_indexing(xL,xR,yL,yR,X_full,Y_full):
    is_boundary = lambda x,y: np.isclose(x,xL) | np.isclose(x,xR) | np.isclose(y,yL) | np.isclose(y,yR) 
    boundaries = is_boundary(X_full, Y_full)
    inner = ~boundaries 
    return inner, boundaries 


def get_fh(N,X,Y,f):
    ### TODO 
    # Initialize right hand side for the discrete system 
    # Since the Dirichlet B.C are homogeneous, rhs = source
    ### end TODO
    

def advance_time(N, h, dt, theta, fh, uh_old):
    ### TODO 
    # Assemble the reduced matrix using the Kronecker product
    ### end TODO


def my_driver(testproblem, parameters, N, N_time):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    yL = testproblem["yL"]
    yR = testproblem["yR"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    f = testproblem["f"]
    
    theta = parameters["theta"]
    max_steps = parameters["max_steps"]
    plot_freq = parameters["plot_freq"]

    dt = (tend-t0)/N_time 
    
    # We assume that the domain is square
    assert np.isclose(xR-xL, yR-yL)

    # Grid generation
    h, X, Y, X_full, Y_full = get_mesh(N, xL, xR, yL, yR)
    inner, _ = get_indexing(xL, xR, yL, yR, X_full, Y_full)

    # Initialization

    # initialize U
    solution_full = u0(X_full, Y_full)

    # Evaluate the source term at grid points 
    fh = get_fh(N, X, Y, f)

    # start time marching
    time = t0
    done = 0

    # do output
    print('\n# START PROGRAM')
    
    # Time stepping
    for j in range(1, max_steps + 1):

        # check that time 'tend' has not been exceeded
        if (time + dt) > tend:
            done = 1
        time = time + dt

        # do output to screen if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            print('Taking time step %i: \t update from %f \t to %f' %
                  (j, time - dt, time))

        # solution update
        # update the solution at inner points
        ## TODO

        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph(solution_full, X_full, Y_full, time, theta, done)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph(solution_full, X_full, Y_full, time, theta, done)


# Main file for solving the 2D Poisson equation -u''(x) = f(x)
# With homogeneous Dirichlet B.C. 

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # read in test problem:
    testproblem = get_testproblem()

    # read in problem parameters:
    parameters = define_default_parameters()

    # call the driver routine for the given grid sizes
    N = parameters["N"]
    N_time = parameters["N_time"]
    my_driver(testproblem, parameters, N, N_time)
    


