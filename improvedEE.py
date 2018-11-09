from thresholdFunc import hardThreshold
from scipy.sparse.csgraph import connected_components
import scipy.linalg
import numpy.linalg
import pp


def _comps_inv(mat):
    return numpy.linalg.inv(mat)


#TODO:Args for each thresholding function
def improvedEE(cov,thrArgs=0.0,func=hardThreshold,core_num=1):
    var_num=(cov.shape)[0]
    thr_cov=func(cov,thrArgs)
    compNum,bins=connected_components(thr_cov,directed=False)
    compSize=[0 for i in range(compNum)]
    for i in range(len(bins)):
        compSize[bins[i]]+=1
    var_seq=[(i,bins[i]) for i in range(var_num)]
    var_seq=[each[0] for each in sorted(var_seq,key=lambda x:x[1],reverse=False)] # The reordered sequence of vars
    reorder_cov=cov[var_seq]
    reorder_cov=reorder_cov[:,var_seq]
    #Split each connected component
    offset = 0  #TODO:No need to use offset

    # Single core
    if core_num==1:
        comp_Omega = []
        for index in range(compNum):
            start = offset
            end = compSize[index] + offset
            offset = end
            comp_Omega.append(numpy.linalg.inv(reorder_cov[start:end, start:end]))
    elif core_num>1:
        size=compNum//core_num  #TODO:Take care the condition when size is a float number
        part=[sum(compSize[i*size:(i+1)*size]) for i in range(core_num)]
        server = pp.Server(ncpus=core_num)
        core_jobs=[
            server.submit(_comps_inv,
                          (scipy.linalg.block_diag(reorder_cov[i*part[i]:(i+1)*part[i],i*part[i]:(i+1)*part[i]]),),
                          modules=('numpy.linalg',)) for i in range(core_num)
        ]
        comp_Omega=[job() for job in core_jobs]
        server.destroy()
    else:
        print("The number of core should be positive.")
        return None
    Omega = scipy.linalg.block_diag(*comp_Omega)
    return Omega,var_seq

