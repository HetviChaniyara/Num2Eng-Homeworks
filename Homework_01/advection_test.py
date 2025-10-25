import numpy as np 
import advection_template as program

## Test for (a)
def test_compute_dt():
    CFL = 0.8
    a = 1.2
    dx = 0.3
    assert np.isclose(program.compute_dt(CFL, a , dx), 0.2)


## Test for (b) 
def test_update():
    U = np.array([0.3, 0.9, 0.25, 1])
    a  = 0.9
    dt = 0.5
    dx = 0.75

    UU = U.copy()
    program.update_ftcs(UU, a, dt, dx)
    assert np.all(np.isclose(UU, np.array([0.3,   0.915, 0.22,  1.])))

    UU = U.copy()
    program.update_upwind(UU, a, dt, dx)
    assert np.all(np.isclose(UU, np.array([0.3,  0.54, 0.64, 1.])))


## Test for (c) 
def test_compute_error():
    time = 12
    x = np.array([0, 0.25, 0.75, 1])
    U = np.array([0, 0.25, 0.75, 1])
    uexact = lambda x, t: -x * t

    err_max = program.compute_error(x, time, uexact, U)
    assert np.isclose(err_max, 13.0)

    print()