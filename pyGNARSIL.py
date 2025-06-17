

import numpy as np
import itertools
import math


def bitBuilder(n,k): #finds all weight k binary strings
    seed=list(range(n))
    bitStrings=np.zeros((math.comb(n,k),n))
    all_combos = np.array(list(itertools.combinations(seed, k)))
    map=np.eye(n)
    for i in  range(all_combos.shape[0]):
        bitStrings[i,:]=np.sum(map[all_combos[i, :]], axis=0)
    return bitStrings
        
def residualWeight(splitArray): #finds residual weight (heuristic) for each stabilizer split
     depGauge=np.sum(splitArray,axis=0) %2 
    
     return np.sum(depGauge)

def depGauge(splitArray): #finds Linearly Dependent Gauge for each split
    return np.sum(splitArray,axis=0) %2 


def symplecticMatrix(x, y): #compact way to find commutativity relations between Pauli Ops
    num_rows, num_cols = x.shape
    n_qubits = num_cols // 2
    V = np.block([
            [np.zeros((n_qubits, n_qubits)), np.eye(n_qubits)],
            [np.eye(n_qubits), np.zeros((n_qubits, n_qubits))]
    ])
    return (x @ V @ y.T) % 2


def fillGauges(gaugeOps,idx,splitArray):
     newArrays=[]
     for gauge in gaugeOps:
         if not np.any(np.all(splitArray == gauge, axis=1)): #Check for Duplicates
             splitArray[idx,:]=gauge
             newArrays.append(splitArray.copy())
             splitArray[idx,:]=np.zeros(splitArray.shape[1])

     return newArrays
    
def pyGNARSIL(code,toSplit,weight,numPieces): #main runner
    #code assumes [Lx;S;Lz] form
    #toSplit list of rows indicating which Stabilizers to Split
    #weight -> weight of Gauges
    #numPieces-> How many Gauges per Stabilizer
    n=code.shape[1]
    #find Gauges
    solutions=[] #output of stab split arrays
    binaryStrings=bitBuilder(n,weight)
    
    findGauges=symplecticMatrix(binaryStrings, code)
    zeros=np.all(findGauges == 0, axis=1)
    gaugeIdx=np.where(zeros)[0]

    mask = np.zeros(binaryStrings.shape[0], dtype=bool)
    mask[list(gaugeIdx)] = True
    gaugeOps=binaryStrings[mask]
    


    #main loop
    for stabs in toSplit:
        splitArray=np.zeros((numPieces+2,n))
        splitArray[0,:]=code[stabs,:]
        graph=[]
        for i in range(1,numPieces):
            graph=fillGauges(gaugeOps,i,splitArray)+graph
            graph.sort(key=residualWeight)
            splitArray=graph.pop(0)
        splitArray[-1,:]=depGauge(splitArray)

        solutions.append(splitArray)
    return solutions



            
            
        
        
        
    





#