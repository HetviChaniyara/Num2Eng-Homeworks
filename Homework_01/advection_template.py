"""
Numerical Mathematics for Engineers II WS 25/26
Homework 01 Exercise 1.3
1D Linear advection
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the test problem:
# u_t + a u_x = 0
def get_testproblem():
    testproblem = {}

    # t_0: initial time
    testproblem["t0"] = 0.0

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0

    # advection speed
    testproblem["a"] = 1.0

    # u0: initial data
    testproblem["u0"] = lambda x: np.sin(2 * math.pi * x)

    # uexact: exact solution
    testproblem["uexact"] = lambda x, t: testproblem["u0"]((x-testproblem["a"]*t) % testproblem["xR"])

    # tend: final time
    testproblem["tend"] = 1.0

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 40

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 10000

    # CFL: CFL number used
    parameters["CFL"] = 0.8 # Change Here for question d

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


def graph_solution(U, x, time, uexact, xL, xR, method, initialize, final_time):
    # remove the ghost cells at the boundaries
    x_plot = x[1:-1]
    U_comp = U[1:-1]

    # evaluate true solution
    U_true = uexact(x_plot, time)

    if initialize:
        plt.figure(1)
        plt.ion()

    plt.figure(1)
    plt.plot(x[1:-1], U_comp, 'r.', markersize=4, label='computed solution')

    plt.plot(x[1:-1], U_true, 'k-', label='true solution')
    plt.title(method)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)

    if final_time == 1:
        plt.figure(2)
        plt.plot(x[1:-1], U_comp, 'r.', markersize=4, label='computed solution')

        plt.plot(x[1:-1], U_true, 'k-', label='true solution')
        plt.title(method)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.xlim(xL, xR)
        plt.legend()
        save_str = 'results/' + method + '.eps'
        plt.savefig(save_str)

    plt.show()
    plt.pause(0.01)
    plt.clf()



#### Question (a) 
def compute_dt(CFL, a, dx):
    dt = CFL*dx/ abs(a)## TODO - Only a is okay since it is positive but this is better if a is negative
    return dt



#### Question (b)
#
def update_ftcs(U, a, dt, dx):
    ## TODO 
    Unew = np.copy(U)
    for i in range(1, len(U)-1):
        Unew[i] = U[i] - ((a*dt/(2*dx))*(U[i+1]-U[i-1]))


    U[:]= Unew
    return U

def update_upwind(U, a, dt, dx):
    ## TODO
    Unew = np.copy(U)
    coeff = a*dt/(dx)
    for i in range(1, len(U)-1):
        Unew[i] = U[i] - (coeff*(U[i]-U[i-1]))
    
    U[:] = Unew
    return U

def compute_error(x, time, uexact, U):
    ## TODO
    U_exact = uexact(x, time) # Obtain the exact solution at x,t
    error = np.abs(U - U_exact)
    err_max = np.max(error)# get max of abs error
    return err_max
#
#######

# Driver
#
# Output:
# - errL1: error in L1 norm
# - errLmax: error in maximum norm
def my_driver(method, testproblem, parameters, N):

    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    a  = testproblem["a"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    uexact = testproblem["uexact"]

    max_steps = parameters["max_steps"]
    CFL = parameters["CFL"]
    plot_freq = parameters["plot_freq"]

    # Grid generation

    # calculate dx: cell length in space
    dx = (xR - xL) / N

    # create mesh with ghost cells
    x = np.linspace(xL - dx, xR, N + 2)

    # Initialization

    # initialize U
    U = u0(x)

    # start time marching
    time = t0
    done = 0

    # do output
    print('\n# START PROGRAM')

    # Plot initial data
    if plot_freq != 0:
        graph_solution(U, x, time, uexact, xL, xR, method, True, False)

    # Time stepping
    for j in range(1, max_steps + 1):

        # set periodic boundary conditions
        U[0] = U[-2]
        U[-1] = U[1]  

        # impose CFL condition to find dt
        dt = compute_dt(CFL, a, dx)

        # check that time 'tend' has not been exceeded
        if (time + dt) > tend:
            dt = tend - time
            done = 1
        time = time + dt

        # do output to screen if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            print('Taking time step %i: \t update from %f \t to %f' % (j, time - dt, time))

        # take a time step
        if method == 'FTCS':
            update_ftcs(U, a, dt, dx)

        elif method == 'upwind':
            update_upwind(U, a, dt, dx)
        else:
            raise ValueError('Stop in my_driver. No appropriate method chosen.')

        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph_solution(U, x, time, uexact, xL, xR, method, False, False)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph_solution(U, x, time, uexact, xL, xR, method, False, True)

    # Compute error
    U[-1] = U[1]
    err_max = compute_error(x[1:], time, uexact, U[1:])

    print('Error in maximum norm:\t %3.2e\n' % err_max)

    return err_max

# Main file for solving
# the linear advection equation u_t + a u_x = 0
# using FD schemes:
#   -- FTCS scheme
#   -- upwind scheme
# using periodic b.c.


def main():
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose method:
    #
    # Options:
    # 'FTCS'  : forward time central space
    # 'upwind': upwind scheme 
    method = 'FTCS'

    # read in test problem:
    testproblem = get_testproblem()

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
        errLmax = my_driver(method, testproblem, parameters, N)

        N_vec[k] = N
        errLmax_vec[k] = errLmax

        N = N * 2

    # compute convergence rate
    rateLmax = np.diff(np.log(errLmax_vec)) / np.diff(np.log(1. / N_vec))

    # write results to file
    with open("results/error.txt", "w") as f:
        f.write('\nLmax error:\n')
        f.write(f' {N_vec[0]} \t {errLmax_vec[0]:.3e} \t NaN\n')
        for i in range(1, parameters["Nrefine"] + 1):
            f.write(f'{N_vec[i]} \t {errLmax_vec[i]:.3e} \t {rateLmax[i - 1]:.2f} \n')


if __name__ == '__main__':
    main()