from mpi4py import MPI
import sys, slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

slepc4py.init(sys.argv);
comm = MPI.COMM_WORLD
rank = comm.Get_rank();
size = comm.Get_size();

if rank == 0:
    print('==== parallel eigendecomposition starts ====',flush=True);
else:
    None;
comm.Barrier();
print('processor '+str(rank)+' is working properly ...',flush=True);

#----- load M matrix ------
M = np.loadtxt('M.txt',dtype=np.complex128);
row,col = M.shape;
#--------------------------------------------
#----- declare petsc dense matrix ------ 
data = PETSc.Mat().create(comm=PETSc.COMM_WORLD);
# data.setType(PETSc.Mat.Type.DENSE);
data.setSizes((row,col));
data.setFromOptions(); 
data.setPreallocationNNZ((row,col)); 
data.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR,False);
#---------------------------------------
#----- loading petsc dense matrix -----
Istart, Iend = data.getOwnershipRange()
for I in range(Istart,Iend):
    data.setValues(I,range(col),M[I,:],addv=PETSc.InsertMode.INSERT);
data.assemble();
#--------------------------------------
#----- parallel eigendecomposition ------
E = SLEPc.EPS(); E.create(comm=PETSc.COMM_WORLD); E.setOperators(data);
E.setBalance(2); 
E.setProblemType(3); 
E.setTolerances(tol=1e-9); 
E.setType('krylovschur');
E.setDimensions(nev=row); E.setFromOptions(); 
E.solve(); nconv = E.getConverged(); 
eigVal,eigVec = [],[];
if nconv > 0: 
    vr, wr = data.getVecs();
    vi, wi = data.getVecs(); 
    for t in range(nconv):
        k = E.getEigenpair(t,vr,vi);
        eigVal = eigVal+[k];
        tmp=[complex(vr0, vi0) for vr0, vi0 in zip(vr.getArray(),vi.getArray())];
        eigVec.append(tmp);
else:
    None;
# eigVal = [i[1] for i in sorted(enumerate(eigVal), key=lambda x:x[1].real)]
eigVal = np.asarray(eigVal);
eigVec = np.array(eigVec).transpose();
eigVec = comm.gather(eigVec,root=0);
if rank == 0:
    eigVec = np.concatenate([np.array(item) for item in eigVec],axis=0)
    np.savetxt('./eigVec.txt',eigVec);
    np.savetxt('./eigVal.txt',eigVal);
else:
    None;
#------------------------------------------
data.destroy(); E.destroy();











