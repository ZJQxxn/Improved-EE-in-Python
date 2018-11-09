import numpy as np
import time
from improvedEE import improvedEE
from evaluation import accuracy
from EE import EE

if __name__ == '__main__':
    prePath = 'D:\\programming\\R\\improved_EE\\'
    graph = np.genfromtxt(prePath + 'precision\\balance_2500vars_20blks.csv', delimiter=',')
    data = np.genfromtxt(prePath + 'data\\balance_2500vars_20blks_5000samples.csv', delimiter=',')
    print('Finished reading data.')

    rowNum, colNum = data.shape
    mean=data.mean(axis=0)
    data=data-mean
    cov=np.dot(data.T,data)
    cov/=np.max(abs(cov))
    #print(cov)
    print('Improved EE:')
    start=time.time()
    Omega,var_seq=improvedEE(cov,0.23,core_num=1)
    print(time.time()-start)
    reorder_graph=graph[var_seq]
    reorder_graph=reorder_graph[:,var_seq]
    Omega/=np.max(abs(Omega)) #Normalization
    print(accuracy(reorder_graph,Omega))
    print('EE:')
    start = time.time()
    ee_model=EE(cov,0.23)
    ee_model /= np.max(abs(ee_model))
    print(time.time() - start)
    print(accuracy(graph, ee_model))


