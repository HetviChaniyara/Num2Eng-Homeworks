"""
Numerical Mathematics for Engineers II WS 25/26
Homework 08 Exercise 8.2
1D FEM - solution
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os
import mesh 
import femlib2D 

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
        testproblem["f"] = lambda x, y: 10 * np.pi**2 * np.multiply(np.sin(3 * np.pi * x), np.sin(np.pi * y))
        testproblem["uexact"] = lambda x, y: np.multiply(np.sin(3. * np.pi * x), np.sin(np.pi * y))        
    else:
        raise Exception(
            'Stop in testproblem. Choice of test problem does not exist')

    return testproblem



# Set parameters for solving the problem
def define_default_parameters():
    parameters = {}

    # nrefine: how many refinements do we do?
    parameters["Nrefine"] = 3

    # N: dimension of test space (the smallest one)
    parameters["n"] = 30

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    return parameters


# plot computed and exact solution
def graph(uh, uexact, N):

    h = 1. / (N + 1)
    xh, yh = np.meshgrid(np.linspace(0, 1, N + 2), np.linspace(0, 1, N + 2))
    uh = uh.reshape((N+2,N+2))

    # Create plot with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot of the numerical solution
    contour = axs[0].contourf(xh, yh, uh, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axs[0])
    axs[0].set_title('Numerical solution of the BVP')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Surface plot of the numerical solution
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(xh, yh, uh, cmap='viridis')
    ax.set_title('Surface plot of the numerical solution of the BVP')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')

    plt.tight_layout()
    plt.show()




def get_matrix_rhs(N, f):
    ## TODO 
    # obtain mesh from the function in the mesh file
    B, C, D, E = mesh.getTriangulationMatrices(N)
    
    nNodes = len(B)
    nTriangles = len(C)
    
    # global matrix An and fn
    An = sp.lil_matrix((nNodes, nNodes))
    fn = np.zeros(nNodes)
    
    # looping over all triangles (the summation we see in the formulation in part 1 b)
    for l in range(nTriangles):
        # coord and idxs of vertices of triangle l
        indices = C[l, :]
        Bkl = B[indices, :]
        
        # transformation and local stiffness matrix
        d_l, b_11, b_12, b_22 = femlib2D.get_trafo(Bkl) # mapping reference to physical
        S_l = femlib2D.get_S_loc(d_l, b_11, b_12, b_22)
        
        # Quad approx using hint from document
        # midpoint rule: 1/3 * Area * f(centroid) = 1/2 * dl * 1/3* f(centroid) = 1/6 *dl*f(centroid)
        f_loc = (1/6) * d_l * f(E[l, 0], E[l, 1])
        
        # adding local contributions to global matrix
        for i in range(3):
            fn[indices[i]] += f_loc
            for j in range(3):
                An[indices[i], indices[j]] += S_l[i, j]
    return An, fn, D



def my_driver(testproblem, parameters, n):
    # Extract problem information
    xL = testproblem["xL"]
    xR = testproblem["xR"]
    yL = testproblem["xL"]
    yR = testproblem["xR"]
    f = testproblem["f"]
    uexact = testproblem["uexact"]
    plot_freq = parameters["plot_freq"]

    An, fn, D = get_matrix_rhs(n, f)
    
    # solve
    # TODO 
    # obtaining boundary nodes
    boundary_indices = np.where(D == 1)[0]
    
    # Dirichlet boundary conditions: u_i = 0
    # We zero out the rows and put 1 on the diagonal
    for idx in boundary_indices:
        # zeroes out the entire row
        An.rows[idx] = []
        An.data[idx] = []
        An[idx, idx] = 1.0 
        # we get 0*u_0 + ... + 1*u_i + ... 0*u_n = 0 --> u_i = 0
        fn[idx] = 0.0 # 0 rhs
        
    # Solve the system directly (scipy handles LIL internally)
    uh_full = sp.linalg.spsolve(An, fn)

    # plot
    if plot_freq != 0:
        graph(uh_full, uexact, n)


    # error 
    xh, yh = np.meshgrid(np.linspace(0, 1, n + 2), np.linspace(0, 1, n + 2))
    errmax = np.max(np.abs(uh_full - uexact(xh,yh).flatten()))
    
    # errL2 = np.sqrt(scipy.integrate.nquad(lambda x,y: (uh_full- uexact(x,y))**2, [[-1, 1], [-1,1]] )[0])
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
    errL2_vec = np.zeros(parameters["Nrefine"] + 1)

    # call the driver routine for different space dimensions 
    n = parameters["n"]
    for k in range(parameters["Nrefine"] + 1):
        errL2 = my_driver(testproblem, parameters, n)

        n_vec[k] = n
        errL2_vec[k] = errL2

        n = n*2
    if parameters["Nrefine"] > 0:
        plt.semilogy(n_vec, errL2_vec)
        plt.grid()
        plt.xlabel("$n$")
        plt.ylabel("$e_n$")
        plt.savefig(f'results/error.png')
        
        plt.show()
        
        plt.pause(2.0)
        plt.clf()
