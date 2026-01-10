import numpy as np
import multigrid_template as program


def test_restriction():
    fine = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    assert (program.restrict(fine) == np.array([[1, 3], [7, 9]])).all()


def test_prolongation():
    fine = np.array([[1, 3], [7, 9]])

    assert (program.prolongate(fine) == np.array([[1, 2, 3],
                                                  [4, 5, 6],
                                                  [7, 8, 9]])).all()


def main():
    test_prolongation()
    test_restriction()


if __name__ == '__main__':
    main()
