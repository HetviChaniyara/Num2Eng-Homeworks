import numpy as np
import FV_Burgers as program # change later to FV_Burgers

## Tests for functions for flux computation from Exercise 12.1

# checks flux computed by LF-FV (Lax-Friedrichs) scheme
def test_LF_FV():
    # set test parameters
    N = 40
    dt = 0.02
    dx = 0.01
    def f(x): return 0.5 * x * x  
    U = np.zeros(42)
    U[11:21] = 1.
    #get flux
    flux = program.LF_FV(U,N,dt,dx,f)
    fluxtrue = np.zeros(41)
    fluxtrue[11:21] = 0.5

    assert np.all(flux == fluxtrue)

# checks flux computed by LLF (local Lax-Friedrichs) scheme
def test_LLF():
    # set test parameters
    N = 40
    dt = 0.02
    dx = 0.01
    def f(x): return 0.5 * x * x  
    def df(x): return  x
    U = np.zeros(42)
    U[11:21] = 1.
    #get flux
    flux = program.LLF(U,N,dt,dx,f,df)
    fluxtrue = np.zeros(41)
    fluxtrue[10] = -0.25
    fluxtrue[11:20] = 0.5
    fluxtrue[20] = 0.75

    assert np.all(flux == fluxtrue)

