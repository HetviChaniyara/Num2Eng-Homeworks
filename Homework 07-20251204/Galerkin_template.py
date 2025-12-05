"""
Numerical Mathematics for Engineers II WS 25/26
Homework 07 Exercise 7.3
Galerkin method - template
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def get_testproblem(testfunction):
    testproblem = {}

    # xL,xR : domain [xL,xR]
    testproblem["xL"] = -1.0
    testproblem["xR"] =  1.0

    # choice
    testproblem["choice"] = testfunction
    
    # right-hand-side function and exact solution
    if (testfunction == "sin"):
        testproblem["f"] = lambda x: np.sin(np.pi * x) * np.pi**2
        testproblem["uexact"] = lambda x: np.sin(np.pi * x) 
    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    return testproblem



# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # v: basis for the test space
    parameters["v"] = lambda i,x: x**i - x**(i + 2)
    parameters["dv"] = lambda i,x: i * x**(i - 1) - (i + 2) * x**(i + 1)

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 9

    # N: dimension of test space (the smallest one)
    parameters["n"] = 1

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


# plot computed and exact solution
def graph(x, uhx, uexact, xL, xR, n):

    # evaluate the true and exact solutions
    U_true = uexact(x)
    U_comp = uhx(x)

    plt.figure(1)
    plt.ion()

    plt.figure(1)
    plt.plot(x, U_comp, 'r.', markersize=4, label='computed solution')
    plt.plot(x, U_true, 'k-', label='true solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.xlim(xL, xR)
    plt.legend()
    plt.savefig(f'results/solution{n}.png')

    plt.show()
    plt.pause(2.0)
    plt.clf()


def get_galerkin_matrix(n, dv):
    # TODO 
    A = np.zeros((n, n))
    xL = -1.0
    xR = 1.0
    order = n + 2 
    
    # interior nodes --> phi starts at 1 so index 1 to n 
    for i in range(1, n + 1):
        for k in range(1, n + 1):
            
            def g(x):
                return dv(i, x) * dv(k, x)
                    
            # store integral at A[i-1, k-1] --> matrix index is phi index - 1 (0 to n-1)
            # output is estimated value , error estimate - we dont care about error
            A[i - 1, k - 1], _ = scipy.integrate.fixed_quad(g, xL, xR, n=order)
    return A


def get_rhs(f, v, n):
    # TODO 
    xL = -1.0
    xR = 1.0
    order = n + 2 
    b = np.zeros(n)
    for k in range(1, n + 1):
        
        def g(x):
            return f(x) * v(k, x)
        
        # F(phi_k) = f(phi_k)v(phi_K)
        b[k - 1], _ = scipy.integrate.fixed_quad(g, xL, xR, n=order)
    return b

def my_driver(testproblem, parameters, n, x):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    f = testproblem["f"]
    v = parameters["v"]
    dv = parameters["dv"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    matrix = get_galerkin_matrix(n, dv)
    rhs = get_rhs(f, v, n)
    
    # TODO 
    alpha = scipy.linalg.solve(matrix,rhs)

    # define function to calculate approximate u
    def uh(x):
        u_val = np.zeros_like(x, dtype=float)
        # The coefficients alpha are 0-indexed, but phi_k is 1-indexed (k=1 to n)
        for k in range(1, n + 1):
            u_val += alpha[k - 1] * v(k, x)
        return u_val
    

    # END TODO 

    # plot
    if plot_freq != 0 and n in [1,2,5,10]:
        graph(x, uh, uexact, xL, xR,n)

    # error 
    # TODO 
    error_integral = lambda x: (uh(x) - uexact(x))**2
    errL2 = np.sqrt(scipy.integrate.fixed_quad(error_integral, xL, xR, n=n + 2)[0])
    return errL2 


# Main file for solving the Poisson equation -u''(x) = f(x)
# using periodic DBC.


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    print("")

    # Choose test problem:
    #
    # Options:
    # 'sin': sine right-hand-side function
    testfunction = 'sin'

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
    errL2_vec = np.zeros(parameters["Nrefine"] + 1)

    # Grid generation (for plotting and error computation)
    # TODO 
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    x = np.linspace(xL, xR, 100)
    # END TODO 

    # call the driver routine for different space dimensions 
    n = parameters["n"]
    for k in range(parameters["Nrefine"] + 1):
        errL2 = my_driver(testproblem, parameters, n, x)

        n_vec[k] = n
        errL2_vec[k] = errL2

        n = n + 1

    if parameters["Nrefine"] > 0:
        plt.semilogy(n_vec, errL2_vec)
        plt.grid()
        plt.xlabel("$n$")
        plt.ylabel("$e_n$")
        plt.savefig(f'results/error.png')
        
        plt.show()
        plt.pause(2.0)
        plt.clf()
