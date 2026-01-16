import numpy as np 
import FV_1D_template as program

## Test for (a)
def test_upwind():
    N = 5
    U = np.arange(N+2)
    assert np.all(np.isclose(program.upwind(U,N), np.arange(N+1)))


## Test for (b) 
def test_LW():
    N = 5
    U = np.ones(N+2)
    dt = 100.
    dx = 1.
    F = program.LW(U,N,dt,dx)
    assert np.all(np.isclose(F, np.ones(N+1)))

    U[0::2]=-1
    F = program.LW(U,N,dt,dx)
    assert np.all(np.isclose(F, np.array([-100.,  100., -100.,  100., -100.,  100.])))


## Test for (c) 
def test_LF_FV():
    f = lambda x: 0*x 
    N = 5
    U = np.arange(N+2)
    dt = 1.
    dx = 1.
    F = program.LF_FV(U,N,dt,dx,f)
    assert np.all(np.isclose(F, -0.5*np.ones(N+1)))

    f = lambda x: x
    F = program.LW(U,N,dt,dx)
    assert np.all(np.isclose(F, np.arange(N+1)))