import scipy
import scipy.sparse
import numpy
import networkx

a=numpy.ones((2,2))
b=[a,a,a]
c=scipy.linalg.block_diag(*b)
print(c)