import numpy as np
import non_conservative as program


def test_update_non_conservative():
    U = np.array([0.0, 0.5, 1])
    N = 2
    dt = 0.5
    dx = 0.75

    program.update_non_conservative(U, N, dt, dx)

    assert np.all(np.isclose(U, np.array([0, 1 / 3, 2 / 3])))
