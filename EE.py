from thresholdFunc import hardThreshold
import numpy as np

def EE(cov,thrArgs=0.0,func=hardThreshold):
    thr_cov = hardThreshold(cov, thrArgs)
    return np.linalg.inv(thr_cov)