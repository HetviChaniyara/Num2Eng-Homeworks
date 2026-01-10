import numpy as np
import scipy
import time
import matplotlib.pyplot as plt

def define_default_parameters():
    parameters = {}

    # Parameters
    parameters["n_refinements"] = 7
    parameters["N_coarse"] = [1, 1]
    parameters["L_coarse"] = [1.0, 1.0]

    # linear solver settings
    parameters["maxIter"] = 100
    parameters["preconditioner"] = "Jacobi"  # "multigrid" or "Jacobi"

    # plot_freq: how often do we plot?
    parameters["plot_freq"] = 1

    # setup: which source and initial guess do we take?
    # possible choices: 
    # "unit source": source f(x)= 1
    # "low-frequency error": source f(x)=0, lf initial guess
    # "high-frequency error": source f(x)=0, hf initial guess
    parameters["setup"] = "unit source" 

    return parameters


def generate_grid(n_refinements, N_coarse, L_coarse):
    Nx = N_coarse[0] * 2**n_refinements + 1
    Ny = N_coarse[1] * 2**n_refinements + 1

    hx = L_coarse[0] / (Nx - 1)
    hy = L_coarse[1] / (Ny - 1)

    x = np.linspace(0, L_coarse[0], Nx)
    y = np.linspace(0, L_coarse[1], Ny)

    return x, y, hx, hy, Nx, Ny


def apply_operator(u, hx, hy):
    ihx = 1 / hx**2
    ihy = 1 / hy**2
    ihxy = ihx + ihy

    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = -(u[:-2, 1:-1] + u[2:, 1:-1]) * ihx - \
        (u[1:-1, :-2] + u[1:-1, 2:]) * ihy + 2 * u[1:-1, 1:-1] * ihxy

    return laplacian


# Jacobi method
def jacobi(u, f, hx, hy):
    ## TODO
    # in our content u[i,j] where i is x direction and j is y direction
    # and f = b
    u_new = np.copy(u)
    ihx = 1.0 / hx**2
    ihy = 1.0 / hy**2
    
    # simplified
    # 1:-1 in row - all points x except left and right boundaries
    # 1:-1 in column - y direction - all y points except bottom and top boundaries
    # y^(k+1) - all interior nodes where we are solving the poisson equation
    # every point U_ij is calculated using its four neighbors
    u_new[1:-1, 1:-1] = (1.0 / (2*ihx + 2*ihy)) * (
        f[1:-1, 1:-1] + 
        (u[:-2, 1:-1] + u[2:, 1:-1]) * ihx + # shifting entire grid backwards by one row and then forward to get the values from the left and right neighbors
        (u[1:-1, :-2] + u[1:-1, 2:]) * ihy # shifting entire grid columns backwards by one and forward by one to get the values from the top and bottom neighbors 
    )
    return u_new


# relaxtion scheme with a given damping parameter omega
def relax(u, f, hx, hy, omega=0.7):
    ## TODO
    u_jacobi = jacobi(u, f, hx, hy)
    return (1 - omega) * u + omega * u_jacobi

# restrict via injection
def restrict(residual):
    ## TODO
    return residual[::2, ::2]

# prolongate via weighted average of neighboring points
def prolongate(coarse):
    ## TODO
    # fine mesh # of grid points: coarse * 2 - 1
    Nx_fine = 2 * coarse.shape[0] - 1
    Ny_fine = 2 * coarse.shape[1] - 1
    fine = np.zeros((Nx_fine, Ny_fine))
    
    # 3a: points coinciding with coarse points 
    fine[::2, ::2] = coarse
    
    # 3b: horizontal midpoints 
    fine[1:-1:2, ::2] = 0.5 * (coarse[:-1, :] + coarse[1:, :])
    
    # 3c: vertical midpoints
    fine[::2, 1:-1:2] = 0.5 * (coarse[:, :-1] + coarse[:, 1:])
    
    # 3d: center of coarse cells 
    fine[1:-1:2, 1:-1:2] = 0.25 * (
        coarse[:-1, :-1] + coarse[1:, :-1] + 
        coarse[:-1, 1:] + coarse[1:, 1:]
    )
    return fine

def v_cycle(u, f, hx, hy, num_levels, num_smooth=3, num_smooth_cg=10, min_level=0):
    if num_levels == min_level:
        # coarse-grid problem
        for _ in range(num_smooth_cg):
            u = relax(u, f, hx, hy, num_smooth_cg)

    else:
        # presmoothing
        ## TODO
        for _ in range(num_smooth):
            u = relax(u, f, hx, hy)

        # compute residual
        ## TODO
        res_fine = f - apply_operator(u, hx, hy)

        # restrict residual to coarse grid
        ## TODO
        res_coarse = restrict(res_fine)

        # solve on coarse-grid problem (call this function recursively)
        ## TODO
        e_coarse = v_cycle(np.zeros_like(res_coarse), res_coarse, 2*hx, 2*hy, 
                           num_levels - 1, num_smooth, num_smooth_cg, min_level)
        
        # prolongate correction to fine grid
        ## TODO
        u = u + prolongate(e_coarse)

        # postsmoothing
        ## TODO
        for _ in range(num_smooth):
            u = relax(u, f, hx, hy)

    return u


# a dummy class for monitoring iterations of CG
class cg_monitor(object):
    def __init__(self, A, f, x, y, disp=True, plot_freq=0):
        self._disp = disp
        self.niter = 0
        self.A = A
        self.f = f
        self.x = x
        self.y = y
        self.plot_freq = plot_freq

    def __call__(self, rk=None):
        self.niter += 1

        if self._disp:
            if self.niter == 1:
                print(f"  i | Residual")

            if self.plot_freq>0 and (self.niter % self.plot_freq) == 0:
                if False:
                    plt.contourf(self.x, self.y, np.transpose(rk.reshape((self.x.size, self.y.size))), levels=10, cmap='viridis')
                    plt.colorbar()
                else:
                    X, Y = np.meshgrid(self.x, self.y)
                    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, num=3)
                    ax.plot_surface(X, Y, np.transpose(rk.reshape((self.x.size, self.y.size))))
                 
                plt.xlabel("X axis")
                plt.ylabel("Y axis")
                plt.title(r"$-\nabla^2 u = f$ on $\Omega$, $u=0$ on $\partial\Omega$")
                plt.gca().set_aspect('equal')
                plt.show()
                plt.pause(0.1)
                plt.clf()

            error = np.linalg.norm(self.f-self.A * rk)
            print(f"{self.niter:3d} | {error:.6e}")


def main():
    # Set up parameters 
    parameters = define_default_parameters()
    setup = parameters["setup"]
    n_refinements = parameters["n_refinements"]
    N_coarse = parameters["N_coarse"]
    L_coarse = parameters["L_coarse"]
    preconditioner = parameters["preconditioner"]


    # create mesh on finest level
    x, y, hx, hy, Nx, Ny = generate_grid(n_refinements, N_coarse, L_coarse)
    print(f"Number of grid points {Nx}*{Ny}={Nx*Ny}")

    # Right-hand side and initial guess
    if setup == "unit source":
        f = np.zeros((Nx, Ny))
        f[1:-1, 1:-1] = 1.0
        u = np.zeros((Nx, Ny))
    elif setup == "low-frequency error":
        f = np.zeros((Nx, Ny))
        u = np.zeros((Nx, Ny))
        u[1:-1, 1:-1] = 1.0
    elif setup == "high-frequency error":
        f = np.zeros((Nx, Ny))
        u = np.zeros((Nx, Ny))
        u[1:-1, 1:-1] = (np.random.rand(Nx - 2, Ny - 2)- 0.5) * 0.1

    # plot initial guess   
    if parameters["plot_freq"]>0:
        plt.ion() 
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, num=3)
        ax.plot_surface(X, Y, np.transpose(u.reshape((x.size, y.size))))
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title(r"$-\nabla^2 u = f$ on $\Omega$, $u=0$ on $\partial\Omega$")
        #plt.gca().set_aspect('equal')
        plt.show()
        plt.pause(2)
        plt.clf()
    

    # Define the LinearOperator for the Poisson problem
    def matvec(u):
        return apply_operator(u.reshape((Nx, Ny)), hx, hy).flatten()

    def mg_precon(rhs):
        return v_cycle(np.zeros((Nx, Ny)), rhs.reshape((Nx, Ny)), hx, hy, n_refinements).flatten()

    def jacobi_precon(rhs):
        return relax(np.zeros((Nx, Ny)), rhs.reshape((Nx, Ny)), hx, hy).flatten()

    A = scipy.sparse.linalg.LinearOperator(
        shape=(Nx * Ny, Nx * Ny), matvec=matvec)

    if preconditioner == "multigrid":
        M = scipy.sparse.linalg.LinearOperator(
            shape=(Nx * Ny, Nx * Ny), matvec=mg_precon)
    elif preconditioner == "Jacobi":
        M = scipy.sparse.linalg.LinearOperator(
            shape=(Nx * Ny, Nx * Ny), matvec=jacobi_precon)

    # solve
    start_time = time.time()

    monitor = cg_monitor(A, f.reshape(Nx * Ny), x, y, True, parameters["plot_freq"])

    u, info = scipy.sparse.linalg.cg(A, f.reshape(
        Nx * Ny), x0=u.reshape(Nx * Ny), M=M, maxiter=parameters["maxIter"], callback=monitor)
    u = u.reshape((Nx, Ny))
    elapsed_time = time.time() - start_time

    # postprocess
    print(f"Solution norm: {np.linalg.norm(u)}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # Plot results
    if parameters["plot_freq"]>0:
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, num=3)
        ax.plot_surface(X, Y, np.transpose(u.reshape((x.size, y.size))))
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title(r"$-\nabla^2 u = f$ on $\Omega$, $u=0$ on $\partial\Omega$")
        #plt.gca().set_aspect('equal')
        plt.show()
        plt.savefig("gmg.png", dpi=600)
        plt.pause(5)
        plt.clf()


if __name__ == '__main__':
    main()
