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

sols=pyGNARSIL(baconShor,[1,2,3,4],2,3)
print('')
print(sols[0])

