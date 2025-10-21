import numpy as np
from pyGNARSIL import bitBuilder, symplecticMatrix, pyGNARSIL_par

def test_bitbuilder_small():
    arr = bitBuilder(4, 2)
    # there are 6 combos for C(4,2)
    assert arr.shape == (6, 4)
    # each row should have exactly two ones
    assert all(int(x.sum()) == 2 for x in arr)

def test_symplectic_small():
    x = np.array([[1,0,0,1]])  # 2-qubit (X|Z) style vector
    y = np.array([[1,0,0,1]])
    # result is a 1x1 array mod 2
    r = symplecticMatrix(x, y)
    assert r.shape == (1,1)

def test_pygnarsil_par_smoke():
    # small mock: two stabilizers of width 4 bits
    code = np.array([[1,0,1,0],[0,1,0,1]])
    solutions = pyGNARSIL_par(code, [0,1], weight=1, numPieces=2)
    assert isinstance(solutions, list)
    assert len(solutions) == 2
