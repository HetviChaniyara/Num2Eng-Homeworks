"""
Numerical Mathematics for Engineers II WS 25/26
Homework 02 Exercise 2.3
1D heat equation
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

# Define the test problem:
# only u_t - u_xx = f


def get_testproblem():
    testproblem = {}

    # t_0: initial time
    testproblem["t0"] = 0.0

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 3*np.pi/2.

    # tend: final time
    testproblem["tend"] = 1.0

    # uexact: exact solution
    testproblem["uexact"] = lambda x, t: np.exp(-t)*np.sin(x)

    # u0: initial data
    testproblem["u0"] = lambda x: testproblem["uexact"](x, testproblem["t0"])

    # ga, gb: boundary conditions:
    testproblem["ga"] = lambda x: 0
    testproblem["gb"] = lambda t: - np.exp(-t) 

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 40

    # N_time: numer of time steps
    parameters["N_time"] = 20

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 100

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


def graph(U_comp, x_plot, time, uexact, xL, xR, initialize, final_time):
    # evaluate true solution
    U_true = uexact(x_plot, time)

    if initialize:
        plt.figure(1)
        plt.ion()

    plt.figure(1)
    plt.plot(x_plot, U_comp, 'r.', markersize=4, label='computed solution')
    plt.plot(x_plot, U_true, 'k-', label='true solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)
    plt.ylim(-1, +1)

    if final_time == 1:
        plt.figure(2)
        plt.plot(x_plot, U_comp, 'r.', markersize=4,
                 label='computed solution')
        plt.plot(x_plot, U_true, 'k-', label='true solution')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.xlim(xL, xR)
        plt.ylim(-1, +1)
        plt.legend()
        save_str = 'results/heat.png'
        plt.savefig(save_str)

    plt.show()
    plt.pause(0.1)
    plt.clf()

def create_matrices(N, dt, dx):   
    ### TODO 
    l = np.ones(N-1) /(dx**2)
    d = np.ones(N)*-2 /(dx**2)
    A = scipy.sparse.diags([l,d,l], offsets=[-1, 0, 1], shape=(N, N), format='csr') 
    A = A*dt/2
    I = scipy.sparse.diags([np.ones(N)], offsets=[0], shape=(N, N), format='csr')
    matrix_left = I - A
    matrix_right = I + A

    return matrix_left, matrix_right

def get_rhs(matrix_right, U, dt, dx, g): 
    ## TODO 
    U_interior = U[1:-1]
    rhs = matrix_right @ U_interior
    rhs = rhs + ((dt/(2*dx**2))*g)
    return rhs


def update_boundaries(U,g,time,dt,testproblem): 
    ##TODO
    # functions for boundary condition
    ga = testproblem['ga']
    gb = testproblem['gb']

    # boundary values at time t
    # We increment time before passing it into the function to we need to go back for t_n
    ga_n = ga(time-dt)
    gb_n = gb(time-dt)

    ga_n1 = ga(time)
    gb_n1 = gb(time)

    # updated boundaries of U
    U[0] = ga_n1
    U[-1] = gb_n1    

    # Update g - boundary contribution
    g[0] = ga_n + ga_n1
    g[-1] = gb_n + gb_n1


# Driver
#
# Output:
# - errLmax: error in maximum norm


def my_driver(testproblem, parameters):

    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    uexact = testproblem["uexact"]

    dt = (testproblem["tend"] - testproblem["t0"]) / parameters["N_time"]
    N = parameters["N"]
    max_steps = parameters["max_steps"]
    plot_freq = parameters["plot_freq"]

    # Grid generation

    # calculate dx: cell length in space
    dx = (xR - xL) / (N + 1)

    # create mesh with ghost cells
    x = np.linspace(xL, xR, N + 2)

    # Initialization

    # initialize U
    U = u0(x)

    # setup matrices
    matrix_left, matrix_right = create_matrices(N, dt, dx)

    # setup vector of boundary condition 
    g = np.zeros(N)

    # start time marching
    time = t0
    done = 0

    # do output
    print('\n# START PROGRAM')

    # Plot initial data
    if plot_freq != 0:
        graph(U, x, time, uexact, xL, xR, True, False)

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
        # update the vector of boundary conditions and the boundary values of U
        update_boundaries(U,g,time,dt,testproblem)
        rhs = get_rhs(matrix_right, U, dt, dx, g)

        # update the solution at inner points 
        ## TODO
        U_new = scipy.sparse.linalg.spsolve(matrix_left, rhs)
        U[1:-1] = U_new

        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph(U, x, time, uexact, xL, xR, False, False)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph(U, x, time, uexact, xL, xR, False, True)

    # compute error
    true_sol = uexact(x, time)
    err = U - true_sol
    err_max = np.max(np.abs(err))

    print('Error:\t %3.2e\n' % err_max)

    return err_max

# Main file for solving
# heat equation u_t - u_xx = f using zero DBC.


def main():
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # read in test problem:
    testproblem = get_testproblem()

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    err_vec = np.zeros(parameters["Nrefine"] + 1)
    N_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different grid sizes
    for k in range(parameters["Nrefine"] + 1):
        err_max = my_driver(testproblem, parameters)

        N_vec[k] = parameters["N_time"]
        err_vec[k] = err_max

        parameters["N_time"] = parameters["N_time"] * 2
        parameters["N"] = parameters["N"] * 2

    # compute convergence rate
    rate = np.diff(np.log(err_vec)) / np.diff(np.log(1. / N_vec))

    # print convergence rate
    print('\n N \t| \t err \t | p \n')
    print('%i \t %5.3e \t NaN\n' % (N_vec[0], err_vec[0]))
    for i in range(1, parameters["Nrefine"] + 1):
        print('%i \t %5.3e \t %3.2f\n' %
            (N_vec[i], err_vec[i], rate[i - 1]))

    # write results to file for use in latex table
    with open("results/error_latex.txt", "w") as f:
        f.write('\\begin{tabular}{ccc}\n')
        f.write('\\hline N  & err & p \\\\ \n\\hline \n')
        f.write('%i & %5.3e & -- \\\\ \n' %
                (N_vec[0], err_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i & %5.3e & %3.2f \\\\ \n' % (
                N_vec[i], err_vec[i], rate[i - 1]))
        f.write("\\hline\n")
        f.write('\\end{tabular}\n')


if __name__ == '__main__':
    main()
