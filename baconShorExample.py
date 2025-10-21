import time
import numpy as np
import itertools
import math
from numba import njit, prange
Sz=np.array([[1,1,1,1,1,1,0,0,0],[0,0,0,1,1,1,1,1,1]])
Sx=np.array([[1,1,0,1,1,0,1,1,0],[0,1,1,0,1,1,0,1,1]])
Lx=np.array([1,0,0,1,0,0,1,0,0])
Lz=np.array([1,1,1,0,0,0,0,0,0])

Sz=np.hstack((np.zeros(Sz.shape),Sz))
Sx=np.hstack((Sx,np.zeros(Sx.shape)))
Lx=np.hstack((Lx,np.zeros(Lx.shape)))
Lz=np.hstack((np.zeros(Lz.shape),Lz))

S=np.vstack((Sx,Sz))
baconShor=np.vstack((Lx,S,Lz))

#single loop versiobn
start_time = time.time()
sols=pyGNARSIL(baconShor,[1,2,3,4],2,3)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution time (time.time()): {elapsed_time:.6f} seconds")
print(sols[0])




#parallelized version in Numba
start_time = time.time()
sols=pyGNARSIL_par(baconShor,[1,2,3,4],2,3)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time (time.time()): {elapsed_time:.6f} seconds")
print(sols[0])

#first time will be slower due to compilation!! 


