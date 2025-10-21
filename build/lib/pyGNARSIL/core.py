import numpy as np
import itertools
import math
from numba import njit, prange
from math import comb

@njit(fastmath=True)
def nCr(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    num = 1
    den = 1
    for i in range(1, r + 1):
        num *= n - (r - i)
        den *= i
    return num // den


@njit(fastmath=True)
def unrank_combination(idx, n, k):
    """Return the k-combination (as an array of indices) corresponding to rank idx."""
    combo = np.empty(k, dtype=np.int64)
    x = 0
    for i in range(k):
        while nCr(n - x - 1, k - i - 1) <= idx:
            idx -= nCr(n - x - 1, k - i - 1)
            x += 1
        combo[i] = x
        x += 1
    return combo


@njit(parallel=True,fastmath=True)
def bitBuilder(n, k):
    num_combos = nCr(n, k)
    bitStrings = np.zeros((num_combos, n))

    for idx in prange(num_combos):
        combo = unrank_combination(idx, n, k)
        for j in range(k):
            bitStrings[idx, combo[j]] = 1

    return bitStrings

@njit(parallel=True, fastmath=True)
def depGauge(splitArray):  # finds Linearly Dependent Gauge for each split
    return np.sum(splitArray, axis=0) % 2


@njit(parallel=True, fastmath=True)
def residualWeight(splitArray):  # finds residual weight (heuristic)
    depGauge_vec = np.sum(splitArray, axis=0) % 2
    return np.sum(depGauge_vec)


@njit(parallel=True, fastmath=True)
def symplecticMatrix(x, y):
    num_rows, num_cols = x.shape
    n_qubits = num_cols // 2

    V = np.zeros((num_cols, num_cols))
    for i in range(n_qubits):
        V[i, n_qubits + i] = 1
        V[n_qubits + i, i] = 1

    result = np.empty((num_rows, y.shape[0]))
    for i in prange(num_rows):
        result[i, :] = (x[i, :] @ V @ y.T) % 2
    return result


@njit(parallel=True, fastmath=True)
def fillGauges_numba(gaugeOps, idx, splitArray):
    n_gauge = gaugeOps.shape[0]
    n_split, n = splitArray.shape
    base = splitArray.copy()
    out = np.empty((n_gauge, n_split, n))

    for g in prange(n_gauge):
        arr = base.copy()
        arr[idx, :] = gaugeOps[g, :]
        out[g, :, :] = arr
    return out


def fillGauges(gaugeOps, idx, splitArray):
    newArrays = fillGauges_numba(gaugeOps, idx, splitArray)
    return [newArrays[i] for i in range(newArrays.shape[0])]


@njit(parallel=True, fastmath=True)
def _pyGNARSIL_parallel_core(code, toSplit, gaugeOps, numPieces):
    n = code.shape[1]
    num_stabs = len(toSplit)
    results = np.zeros((num_stabs, numPieces + 2, n))

    for s in prange(num_stabs):
        stabs = toSplit[s]
        splitArray = np.zeros((numPieces + 2, n))
        splitArray[0, :] = code[stabs, :]

        # iterative splitting
        for i in range(1, numPieces):
            n_gauge = gaugeOps.shape[0]
            best_idx = 0
            best_weight = n + 1
            for g in range(n_gauge):
                arr = splitArray.copy()
                arr[i, :] = gaugeOps[g, :]
                w = np.sum(np.sum(arr, axis=0) % 2)
                if w < best_weight:
                    best_idx = g
                    best_weight = w
            splitArray[i, :] = gaugeOps[best_idx, :]

        splitArray[-1, :] = np.sum(splitArray, axis=0) % 2
        results[s, :, :] = splitArray
    return results


def pyGNARSIL_par(code, toSplit, weight, numPieces):  # main runner
    n = code.shape[1]
    binaryStrings = bitBuilder(n, weight)

    findGauges = symplecticMatrix(binaryStrings, code)
    zeros = np.all(findGauges == 0, axis=1)
    gaugeIdx = np.where(zeros)[0]
    gaugeOps = binaryStrings[gaugeIdx]

    
    solutions_array = _pyGNARSIL_parallel_core(code, np.array(toSplit), gaugeOps, numPieces)

   
    solutions = [solutions_array[i] for i in range(solutions_array.shape[0])]

   
    for i, splitArray in enumerate(solutions, 1):
        print(f"The Residual Weight for the Dependent Gauge of Stabilizer {i} is {splitArray[-1].sum()}\n")

    return solutions
