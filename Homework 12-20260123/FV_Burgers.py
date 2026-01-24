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
        testproblem["tend"] = 1.5

    elif (testfunction == "R"):
        # xL,xR : domain [xL,xR]
        testproblem["xL"] = -1.0
        testproblem["xR"] = 3.0

        # u0: initial data
        testproblem["u0"] = lambda x: np.where(x > 0.0, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: np.where((x > 0) * (x < t) == 1.0, x / t, 0.0) + np.where(x >= t, 1.0, 0.0)

        # tend: final time
        testproblem["tend"] = 0.25

    elif (testfunction == "RS"):
        # xL,xR : domain [xL,xR]
        testproblem["xL"] = -1.0
        testproblem["xR"] = 3.0

        # u0: initial data
        testproblem["u0"] = lambda x: np.where((x > 0.0) * (x < 1.0) == 1.0, 1.0, 0.0)

        # uexact: exact solution
        testproblem["uexact"] = lambda x, t: np.where((x > 0) * (x < t) == 1.0, x / t, 0.0) + np.where((x >= t) * (x <= (1 + 0.5 * t)) == 1.0, 1.0, 0.0)

        # tend: final time
        testproblem["tend"] = 2
    # TODO 
    # Part e
    # Also in the driver function
    elif (testfunction == "sine"):
        testproblem["xL"] = 0.0
        testproblem["xR"] = 1.0
        testproblem["u0"] = lambda x: np.sin(2 * np.pi * x)
        testproblem["tend"] = 0.25
        # no exact solution
        testproblem["uexact"] = lambda x, t: np.zeros_like(x)


    else:
        raise Exception('Stop in testproblem. Choice of test problem does not exist')

    return testproblem


# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 0

    # N: number of grid points (on coarsest grid)
    parameters["N"] = 100

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

    if testfunction != "sine":
        plt.plot(x[ind_dof], U_true, 'k-', label='true solution')
    plt.title(method)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)

    if final_time == 1:
        plt.figure(2)
        plt.plot(x[ind_dof], U_comp, 'r.', markersize=4, label='computed solution')

        if testfunction != "sine":
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


def LF_FV(U,N,dt,dx,f):
    #TODO
    # from hw 11
    UL = U[0:N + 1]
    UR = U[1:N + 2]
    a_max = dx / dt
    flux = 0.5 * (f(UL) + f(UR)) - 0.5 * a_max * (UR - UL)
    return flux


def LLF(U,N,dt,dx,f,df):
    #TODO
    UL = U[0:N + 1]
    UR = U[1:N + 2]
    # local speed
    a_local = np.maximum(np.abs(df(UL)), np.abs(df(UR))) # max(|f'(U_L)|, |f'(U_R)|)
    
    flux = 0.5 * (f(UL) + f(UR)) - 0.5 * a_local * (UR - UL)
    return flux

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

        if testfunction == "S" or testfunction == "R" or testfunction == "RS":
            # set transmissive boundary conditions
            U[0] = U[1]
            U[N + 1] = U[N]
        # TODO
        elif testfunction =="sine":
            # same from last time
            U[0] = U[N]
            U[N+1] = U[1]

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

        if method == 'Godunov':
            flux = flux_godunov(U, N)
        elif method == 'LF_FV':
            flux = LF_FV(U,N,dt,dx,f)
        elif method == 'LLF':
            flux = LLF(U,N,dt,dx,f,df)

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
    # 'Sine' : sine
    testfunction = 'sine'

    # Choose method:
    #
    # Options:
    # 'Godunov'  : Godunov flux
    # 'LF_FV'       : Lax-Friedrichs FV version
    # 'LLF'       : Lax-Friedrichs
    method = 'Godunov'

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
        [errL1, errLmax, _, _] = my_driver(method, testproblem, parameters, N)

        N_vec[k] = N
        errL1_vec[k] = errL1
        errLmax_vec[k] = errLmax

        N = N * 2

    # compute convergence rate
    rate = np.diff(np.log(errL1_vec)) / np.diff(np.log(4. / N_vec))
    rateLmax = np.diff(np.log(errLmax_vec)) / np.diff(np.log(4. / N_vec))

    # write results to file
    with open("results/error.txt", "w") as f:
        f.write('\nL1 error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errL1_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' % (N_vec[i], errL1_vec[i], rate[i - 1]))

        f.write('\nLmax error:\n')
        f.write('%i \t %5.3e \t NaN\n' % (N_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i \t %5.3e \t %3.2f\n' % (N_vec[i], errLmax_vec[i], rateLmax[i - 1]))

    # write results to file for use in latex table
    with open("results/"+testfunction+"_"+method+"_error_latex.txt", "w") as f:
        f.write('N  & L1-err & L1-ord & Lmax-err & Lmax-ord\n')
        f.write('%i & %5.3e & -- & %5.3e & --\n' % (N_vec[0], errL1_vec[0], errLmax_vec[0]))
        for i in range(1, parameters["Nrefine"] + 1):
            f.write('%i & %5.3e & %3.2f & %5.3e & %3.2f\n' % (N_vec[i], errL1_vec[i], rate[i - 1], errLmax_vec[i], rateLmax[i - 1]))





def main_graph():
    # TODO
    # part c
    if not os.path.exists("results"):
        os.makedirs("results")

    parameters = define_default_parameters()

    test_cases = ['S', 'R', 'RS']
    methods = ['Godunov', 'LF_FV', 'LLF']
    colors = ['r', 'b', 'g'] #  corresponding color for each method

    for case in test_cases:
        testproblem = get_testproblem(case)
        plt.figure(figsize=(10, 6))
        
        # iterate through methods and plot on same graph
        for i, method in enumerate(methods):
            # call my_driver
            _, _, U_comp, x_comp = my_driver(method, testproblem, parameters, parameters["N"])
            plt.plot(x_comp, U_comp, colors[i] + '.', markersize=4, label=method)

        # plot exact solution
        uexact = testproblem["uexact"]
        plt.plot(x_comp, uexact(x_comp, testproblem["tend"]), 'k-', label='True Solution', linewidth=1.5)

        plt.title(f"Comparison of Methods: Data {case} (N={parameters['N']})")
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"results/comparison_{case}.png"
        plt.savefig(save_path)
        plt.show()


def run_convergence_study():
    # TODO
    # Part d
    if not os.path.exists("results"):
        os.makedirs("results")

    parameters = define_default_parameters()

    test_cases = ['S', 'R']
    methods = ['Godunov', 'LLF', 'LF_FV']

    file_path = "results/final_convergence_latex.txt"
    
    with open(file_path, "w") as f:
        for case in test_cases:
            f.write(f"\nTEST CASE: {case}\n")
            f.write("="*70 + "\n")
            
            for method in methods:
                f.write(f"Method: {method}\n")
                f.write("N   & L1-err    & L1-ord & Lmax-err  & Lmax-ord\n")
                
                N = parameters["N"]
                errL1_old = None
                errLmax_old = None
                
                for k in range(parameters["Nrefine"] + 1):
                    # driver call
                    errL1, errLmax, _, _ = my_driver(method, testproblem_obj := get_testproblem(case), parameters, N)
                    
                    
                    # convergence order
                    if errL1_old is not None:
                        ordL1 = np.log(errL1_old / errL1) / np.log(2.0)
                        ordLmax = np.log(errLmax_old / errLmax) / np.log(2.0)
                        ordL1_str = f"{ordL1:.2f}"
                        ordLmax_str = f"{ordLmax:.2f}"
                    else:
                        ordL1_str = "--"
                        ordLmax_str = "--"

                    f.write(f"{N:<3} & {errL1:5.3e} & {ordL1_str:<6} & {errLmax:5.3e} & {ordLmax_str}\n")
                    
                    errL1_old = errL1
                    errLmax_old = errLmax
                    N *= 2
                f.write("-" * 70 + "\n")

    print(f"Results written to {file_path}")

if __name__ == '__main__':
    main()  
    # Switch here 
    # main_graph()
    # run_convergence_study()

