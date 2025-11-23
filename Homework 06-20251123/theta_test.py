import numpy as np
import theta_template as program

## Test for a)
# Checks restriction of function f for grid size N = 8
def test_get_fh():
    N = 8 
    # get function f
    f = lambda x, y: np.where(np.logical_and(0.05<=(x-0.5)**2+(y-0.5)**2,(x-0.5)**2+(y-0.5)**2<=0.07), 1000., 0.)
    # Grid generation
    h, X, Y, X_full, Y_full = program.get_mesh(N, 0., 1., 0., 1.) 

    fh = program.get_fh(N,X,Y,f)
    fh_true = np.zeros(N*N)
    fh_true[[18, 21, 42, 45]] = 1000.

    assert np.all(fh == fh_true)


## Test for b)
# Checks solution u after one and two time steps
def test_advance_time():
    dt = 0.0005
    N = 3
    theta = 1.
    h = 1/(N+1)

    # get rhs
    fh = np.zeros(N*N)
    fh[[1,3,5,7]] = 1000.

    # initial condition
    uh0 = np.zeros(N*N)
    uh1 = program.advance_time(N, h, dt, theta, fh, uh0)
    uh1_true = np.array([0.00751518, 0.48472915, 0.00751518, 0.48472915, 0.01503036, 0.48472915,
                            0.00751518, 0.48472915, 0.00751518])
    
    uh2 = program.advance_time(N, h, dt, theta, fh, uh1_true)
    uh2_true = np.array([0.02208649, 0.95487977, 0.02208649, 0.95487977, 0.04417298, 0.95487977,
                            0.02208649, 0.95487977, 0.02208649])
    
    assert np.all(np.isclose(uh1, uh1_true))

    assert np.all(np.isclose(uh2, uh2_true)) 


