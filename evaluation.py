from numpy.linalg import norm

def _maxNorm(X):
    dim=X.shape[0]
    m=0
    for i in range(dim):
        for j in range(dim):
            if abs(X[i,j])>m:
                m=abs(X[i,j])
    return m


def accuracy(real,est):
    dim=real.shape[0]
    real_edge=0
    real_nonedge=0
    tp=0
    fp=0
    for i in range(dim):
        for j in range(i,dim):
            if real[i,j] != 0:
                real_edge+=1
                if est[i,j] != 0:
                    tp+=1
            elif real[i,j] == 0:
                real_nonedge+=1
                if est[i,j] != 0:
                    fp+=1
    tpr=tp/real_edge
    fpr=fp/real_nonedge
    frobenius=norm(real - est, ord='fro')
    maximumnorm=_maxNorm(real-est)
    return tpr,fpr,frobenius,maximumnorm