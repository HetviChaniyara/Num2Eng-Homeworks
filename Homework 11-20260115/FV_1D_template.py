"""
Numerical Mathematics for PDEs WS 25/26
Homework 11 Exercise 11.2
Implementation of 1D FVM
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Define the test problem:
# only u_t + a u_x = 0
def get_testproblem(testfunction):
    testproblem = {}

    # t_0: initial time
    testproblem["t0"] = 0.0

    # choice
    testproblem["testfunction"] = testfunction

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = 0.0
    testproblem["xR"] = 1.0

    if (testfunction == "smooth"):
        # u0: initial data
        testproblem["u0"] = lambda x: np.sin(2*np.pi*x)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: testproblem["u0"](x-t)

    elif (testfunction == "discontinuous"):
        # u0: initial data
        testproblem["u0"] = lambda x: np.where(abs(x - 0.5) < 0.25, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: testproblem["u0"]((x - t)%1.)

    else:
        raise Exception('Stop in testproblem. Choice of test problem does not exist')
    
    # tend: final time
    testproblem["tend"] = 1.0

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 2

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 40

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 10000

    # CFL: CFL number used
    parameters["CFL"] = 0.8

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 5

    return parameters


def graph_solution(U, x, time, uexact, xL, xR, testfunction, numflux, initialize, final_time):

    # remove the ghost cells at the boundaries
    x_plot = x[1:-1]

    # evaluate true solution
    U_true = uexact(x_plot, time)

    if initialize:
        plt.figure(1)
        plt.ion()

    plt.figure(1)
    plt.plot(x_plot, U[1:-1], 'r.', markersize=4, label='computed solution')
    plt.plot(x_plot, U_true, 'k-', label='true solution')

    plt.title(testfunction)
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.xlim(xL, xR)
    plt.legend()

    if final_time == 1:
        plt.figure(2)
        plt.plot(x_plot, U[1:-1], 'r.', markersize=4, label='computed solution')
        plt.plot(x_plot, U_true, 'k-', label='true solution')

        plt.title(testfunction)
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        plt.xlim(xL, xR)
        plt.legend()
        save_str = 'results/' + testfunction + '_' + numflux + '.jpg'
        plt.savefig(save_str)

    plt.show()
    plt.pause(0.1) 
    plt.clf()

def upwind(U,N):
    # a>0 , information is coming from the left
    # TODO
    return 1.0 * U[0:-1]

def LW(U,N,dt,dx):
    # TODO
    average = 0.5 *(U[0:-1] + U[1:]) 
    term2 = - (dt/(2*dx))*(U[1:]-U[0:-1])
    return average+term2

def LF_FV(U,N,dt,dx,f):
    # TODO
    flux = 0.5*(f(U[0:-1]) + f(U[1:])) - ((dx/(2*dt)) *(U[1:]-U[0:-1]))
    return flux

# Driver
#
# Output:
# - errL1: error in L1 norm
# - errLmax: error in maximum norm
def my_driver(numflux, testproblem, parameters, N):

    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    uexact = testproblem["uexact"]
    testfunction = testproblem["testfunction"]

    max_steps = parameters["max_steps"]
    CFL = parameters["CFL"]
    plot_freq = parameters["plot_freq"]

    def f(u): return u  # flux function for linear advection

    # Grid generation

    # calculate dx: cell length in space
    dx = (xR - xL) / N

    # create mesh with ghost cells
    x = np.linspace(xL - dx, xR, N + 2)
    x = x + 0.5 * dx

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
        graph_solution(U, x, time, uexact, xL, xR, testfunction, numflux, True, False)

    # Time stepping
    for j in range(1, max_steps + 1):

        # set periodic boundary conditions
        # TODO 
        U[0] = U[N]
        U[N+1] = U[1]
        # impose CFL condition to find dt
        # TODO
        a = 1.0
        dt = CFL * dx / a
        # check that time 'tend' has not been exceeded
        if (time + dt) > tend:
            dt = tend - time
            done = 1
        time = time + dt

        # do output to screen 
        if (plot_freq != 0) and (j % plot_freq) == 0:
            print('Taking time step %i: \t update from %f \t to %f' % (j, time - dt, time))

        if(numflux == "upwind"):
            # compute the flux using upwind 
            flux = upwind(U,N)

        if(numflux == "LW"):
            # compute the flux using Lax-Wendroff 
            flux = LW(U,N,dt,dx)

        if(numflux == "LF-FV"):
            # compute the flux using Lax-Friedrichs, FV version
            flux = LF_FV(U,N,dt,dx,f)

        # using conservative formula
        # TODO
        U[1:-1] = U[1:-1] - (dt/dx) * (flux[1:] - flux[:-1])
        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph_solution(U, x, time, uexact, xL, xR, testfunction, numflux, False, False)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph_solution(U, x, time, uexact, xL, xR, testfunction, numflux, False, True)

    # Compute error
    true_sol = uexact(x[1:-1], time)
    err = U[1:-1] - true_sol
    err_L1 = np.sum(np.abs(err) * dx)
    err_max = np.max(np.abs(err))

    print('Error in L1:\t\t %3.2e' % err_L1)
    print('Error in L^infty:\t %3.2e\n' % err_max)

    return err_L1, err_max


def main():
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    #
    # Options:
    # 'smooth' : sine function
    # 'discontinuous' : hat function
    testfunction = 'discontinuous'


    # Choose numerical flux:
    #
    # Options:
    # 'upwind' 
    # 'LF-FV'
    # 'LW'
    numflux ='LF-FV'

    # read in test problem:
    testproblem = get_testproblem(testfunction)

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    errL1_vec = np.zeros(parameters["Nrefine"] + 1)
    errLmax_vec = np.zeros(parameters["Nrefine"] + 1)
    N_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different grid sizes
    N = parameters["N"]
    for k in range(parameters["Nrefine"] + 1):
        [errL1, errLmax] = my_driver(numflux, testproblem, parameters, N)

        N_vec[k] = N
        errL1_vec[k] = errL1
        errLmax_vec[k] = errLmax

        N = N * 2

    # compute convergence rate
    rate = np.diff(np.log(errL1_vec)) / np.diff(np.log(1. / N_vec))
    rateLmax = np.diff(np.log(errLmax_vec)) / np.diff(np.log(1. / N_vec))

    # write results to file
    with open("results/error_" + testfunction + '_' + numflux + ".txt", "w") as f:
        f.write('\nL1 error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errL1_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' % (N_vec[i], errL1_vec[i], rate[i - 1]))

        f.write('\nLmax error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' % (N_vec[i], errLmax_vec[i], rateLmax[i - 1]))

    # write results to file for use in latex table
    with open("results/error_" + testfunction + '_' + numflux + "_latex.txt", "w") as f:
        f.write('N  & L1-err & L1-ord & Lmax-err & Lmax-ord\n')
        f.write('%i & %5.3e & -- & %5.3e & --\n' % (N_vec[0], errL1_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i & %5.3e & %3.2f & %5.3e & %3.2f\n' % (N_vec[i], errL1_vec[i], rate[i - 1], errLmax_vec[i], rateLmax[i - 1]))


if __name__ == '__main__':
    main()
