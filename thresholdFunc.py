def hardThreshold(mat,thr):
    dim=(mat.shape)[0]
    for i in range(dim):
        for j in range(i,dim):
            if abs(mat[i,j])<=thr:
                mat[i,j]=mat[j,i]=0
    return mat