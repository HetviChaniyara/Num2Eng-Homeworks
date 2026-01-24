import math
import matplotlib.pyplot as plt
import numpy as np
import os


def get_testproblem(testfunction):
    testproblem = {}

    # t_0: initial time
    testproblem["t0"] = 0.0

    # choice
    testproblem["choice"] = testfunction

    if (testfunction == "S"):
        # xL,xR : domain [xL,xR]
        testproblem["xL"] = -1.0
        testproblem["xR"] = 3.0

        # u0: initial data
        testproblem["u0"] = lambda x: np.where(x < 0.0, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: np.where(x < 0.5 * t, 1.0, 0.0)

        # tend: final time
        testproblem["tend"] = 2.0

    elif (testfunction == "R"):
        # xL,xR : domain [xL,xR]
        testproblem["xL"] = -1.0
        testproblem["xR"] = 3.0

        # u0: initial data
        testproblem["u0"] = lambda x: np.where(x > 0.0, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: np.where((x > 0) * (x < t) == 1.0, x / t, 0.0) + np.where(x >= t, 1.0, 0.0)

        # tend: final time
        testproblem["tend"] = 1.5

    elif (testfunction == "RS"):
        # xL,xR : domain [xL,xR]
        testproblem["xL"] = -1.0
        testproblem["xR"] = 3.0

        # u0: initial data
        testproblem["u0"] = lambda x: np.where((x > 0.0) * (x < 1.0) == 1.0, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: np.where((x > 0) * (x < t) == 1.0, x / t, 0.0) + np.where((x >= t) * (x <= (1 + 0.5 * t)) == 1.0, 1.0, 0.0)

        # tend: final time
        testproblem["tend"] = 1.5

    else:
        raise Exception('Stop in testproblem. Choice of test problem does not exist')

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 80

    # max_steps: maximal number of time steps
    parameters["max_steps"] = 10000

    # CFL: CFL number used
    parameters["CFL"] = 0.8

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


def graph_Burger(U, x, time, ind_dof, uexact, xL, xR, method, testfunction, initialize, final_time):

    # remove the ghost cells at the boundaries
    x_plot = x[ind_dof]
    U_comp = U[ind_dof]

    # evaluate true solution
    U_true = uexact(x_plot, time)

    if initialize:
        plt.figure(1)
        plt.ion()

    plt.figure(1)
    plt.plot(x[ind_dof], U_comp, 'r.', markersize=4, label='computed solution')

    if testfunction != "d":
        plt.plot(x[ind_dof], U_true, 'k-', label='true solution')
    plt.title(method)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)

    if final_time == 1:
        plt.figure(2)
        plt.plot(x[ind_dof], U_comp, 'r.', markersize=4, label='computed solution')

        if testfunction != "d":
            plt.plot(x[ind_dof], U_true, 'k-', label='true solution')
        plt.title(method)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.xlim(xL, xR)
        plt.legend()
        save_str = 'results/' + method + '_' + testfunction + '.jpg'
        plt.savefig(save_str)

    plt.show()
    plt.pause(0.1)
    plt.clf()


# Godunov flux for the inviscid Burgers equation
def flux_godunov(U, N):
    # extract left and right velocity
    UL = U[0:N + 1]
    UR = U[1:N + 2]

    flux = 0.5 * np.where(UL<= UR, np.minimum(UR, np.maximum(0,UL))**2, np.maximum(UR**2,UL**2))

    return flux


def update_non_conservative(U, N, dt, dx):
    U[1:N+1] = U[1:N+1] - (dt / dx) * U[1:N+1] * (U[1:N+1] - U[0:N])
    # TODO 


# Driver
#
# Output:
# - errL1: error in L1 norm
# - errLmax: error in maximum norm
def my_driver(method, testproblem, parameters, N):

    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    t0 = testproblem["t0"]
    tend = testproblem["tend"]
    u0 = testproblem["u0"]
    uexact = testproblem["uexact"]
    testfunction = testproblem["choice"]

    max_steps = parameters["max_steps"]
    CFL = parameters["CFL"]
    plot_freq = parameters["plot_freq"]

    def f(x): return 0.5 * x * x  # flux function for Burgers
    def df(x): return  x  # derivative of the flux function for Burgers

    # Grid generation

    # calculate dx: cell length in space
    dx = (xR - xL) / N

    # create mesh with ghost cells
    x = np.linspace(xL - dx, xR, N + 2)
    x = x + 0.5 * dx
    ind_dof = range(1, N + 1)

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
        graph_Burger(U, x, time, ind_dof, uexact, xL, xR, method, testfunction, True, False)

    # Time stepping
    for j in range(1, max_steps + 1):

        # set transmissive boundary conditions
        U[0] = U[1]
        U[N + 1] = U[N]

        # impose CFL condition to find dt
        smax = np.max(np.abs(U))
        dt = CFL * (dx / smax)

        # check that time 'tend' has not been exceeded
        if (time + dt) > tend:
            dt = tend - time
            done = 1
        time = time + dt

        # do output to screen if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            print('Taking time step %i: \t update from %f \t to %f' % (j, time - dt, time))

        if method == 'non_cons':
            update_non_conservative(U, N, dt, dx) 
        else:
            # Default: Godunov flux 
            flux = flux_godunov(U, N)

            # advance using conservative formula
            U[ind_dof] = U[ind_dof] - (dt / dx) * (flux[1: N + 1] - flux[0: N])

        # draw graph if wished
        if (plot_freq != 0) and (j % plot_freq) == 0:
            graph_Burger(U, x, time, ind_dof, uexact, xL, xR, method, testfunction, False, False)

        # if we have done the calculation for tend, we can stop
        if done == 1:
            print('Have reached time tend; stop now')
            break

    if j >= max_steps:
        print('Stopped after %i steps.' % max_steps)
        print('Did not suffice to reach the end time %f.' % tend)

    if plot_freq != 0:
        graph_Burger(U, x, time, ind_dof, uexact, xL, xR, method, testfunction, False, True)

    # Compute error
    true_sol = uexact(x[ind_dof], time)
    err = U[ind_dof] - true_sol
    err_L1 = np.sum(np.abs(err) * dx)
    err_max = np.max(np.abs(err))

    print('Error in L1:\t\t %3.2e' % err_L1)
    print('Error in L^infty:\t %3.2e\n' % err_max)

    return err_L1, err_max, U[ind_dof], x[ind_dof]




def main():
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    #
    # Options:
    # 'S'  : shock solution
    # 'R'  : rarefaction solution
    # 'RS' : rarefaction and shock
    testfunction = 'RS'

    # Choose method:
    #
    # Options:
    # 'Godunov'  : Godunov flux
    # 'non_cons' : non-conservative scheme
    method = 'Godunov'

    # read in test problem:
    testproblem = get_testproblem(testfunction)

    # read in problem parameters:
    parameters = define_default_parameters()

    # call driver
    if parameters["Nrefine"] < 0:
        raise Exception("Stop in main. Nrefine negative!")

    # call the driver routine for different grid sizes
    N = parameters["N"]
    for k in range(parameters["Nrefine"] + 1):
        [errL1, errLmax, _, _] = my_driver(method, testproblem, parameters, N)

        N = N * 2

if __name__ == '__main__':
    main()  

