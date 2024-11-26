from numba import jit
import numpy as np


#generators

@jit
def GLCG(n,M=2**10,As=np.array([3,7,68]),Xs = np.array([1,2,4])):
    k = len(Xs)
    X = np.zeros(n+k)
    X[0:k] = Xs
    for i in range(k,k+n):
        s = 0
        for j in range(k):
            s = s+As[j]*Xs[i-(j+1)]
            s = s%M
        X[i] = s
    return X[k:]




@jit
def KSA(m = 32,K = np.array([2,3,4,7])):
    L = len(K)
    S = np.zeros(m)
    for i in range(m):
        S[i] = i
    j = 0
    for i in range(m):
        j = j+S[i]+K[i%L]
        j = j%m
        S[int(j)],S[i] = S[i],S[int(j)]
    return S


@jit
def PRGA(n=10,m = 32,K = np.array([2,3,4,7])):
    S = KSA(m=m,K=K)
    X = np.zeros(n)
    j = 0
    i = 0
    for k in range(n):
        i = (i+1)%m
        j = (j+S[i])%m

        S[int(j)], S[i] = S[i], S[int(j)]
        X[k] = S[int((S[int(i)]+S[int(j)])%m)]
    return X

@jit
def Ziff(Xs,n = 10,m = 2**10,p1 = 471,p2 = 1586,p3 = 6988,p4 = 9689):
    k = len(Xs)
    X = np.zeros(n+k)
    X[0:k] = Xs
    for i in range(k, k + n):
        s = int(X[i-p1])
        for p in [p2,p3,p4]:
            s = s^int(X[i-p])
        s = s%m
        X[i] = s
    return X[k:]

Xs = np.zeros(10000)
for i in range(len(Xs)):
    Xs[i] = i
@jit
def Marsagli(Xs,n = 10000,m = 2**32,p1 = 55,p2 = 119,p3 = 179,p4 = 256):
    k = len(Xs)
    X = np.zeros(n+k)
    X[0:k] = Xs
    for i in range(k, k + n):
        s = (X[i-p1]+X[i-p2]+X[i-p3]+X[i-p4])%m
        X[i] = s
    return X[k:]


## tests

















