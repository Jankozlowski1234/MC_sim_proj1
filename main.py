from numba import jit
import numpy as np
import pandas as pd
from scipy.special import kolmogorov,erfc
from scipy import stats
import urllib.request
import random as r
import math


##constants

M_GLCG = 2**10
M_ZIFF = 2**10
M_RC = 2**5
M_MAR = 2**10


#generators

@jit
def GLCG(n,M=M_GLCG,As=np.array([3,7,68]),Xs = np.array([1,2,5])):
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
def PRGA(n=10,m = M_RC,K = np.array([2,3,4,7])):
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
def Ziff(Xs,n = 10,m = M_ZIFF,p1 = 471,p2 = 1586,p3 = 6988,p4 = 9689):
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


@jit
def Marsagli(Xs,n = 10000,m = M_MAR,p1 = 55,p2 = 119,p3 = 179,p4 = 256):
    k = len(Xs)
    X = np.zeros(n+k)
    X[0:k] = Xs
    for i in range(k, k + n):
        s = (X[i-p1]+X[i-p2]+X[i-p3]+X[i-p4])%m
        X[i] = s
    return X[k:]


## tests
#KS test
@jit
def calculate_D(Y):
    n = len(Y)
    emp = lambda x: np.sum(Y <= x) / n
    D = 0

    y= 0
    x = abs(emp(y) - (y))
    if x > D:
        D = x

    y= 1
    x = abs(emp(y) - (y))
    if x > D:
        D = x

    for y in Y:
        x = abs(emp(y) - (y))
        if x > D:
            D = x
    return D


def KS_test(Y):
    n = len(Y)
    D = calculate_D(Y)
    D = D*(n)**0.5
    p = 1-kolmogorov(D+1/(6*(n)**0.5)+(D-1)/(4*n))
    return p

#print(KS_test(Xs))

#Chi_sq test


@jit
def count_statistic(Y,kat = np.array([0,0.2,0.4,0.6,0.7,1])):
    n = len(Y)
    k = len(kat)
    n_kat = k-1
    sizes = kat[1:]-kat[:(k-1)]
    sizes = sizes*n
    amount = np.zeros(n_kat)
    for i in range(n_kat):
        if(i==(n_kat-1)):
            amount[i] = np.sum((Y >= kat[i]) & (Y <= kat[i + 1]))
        else:
            amount[i] = np.sum((Y>=kat[i])&(Y<kat[i+1]))
    return np.sum((amount-sizes)**2/sizes)


def chi_sq_test(Y,kat = np.arange(start=0, stop=1.1, step=0.1)):
    statistic = count_statistic(Y,kat)
    k = len(kat) - 1
    return 1 - stats.chi2.cdf(statistic, k-1)



## poker test
M=100
@jit
def count(myList):
    return len(set(myList))

S_2 = np.array([1,15,25,10,1])
@jit
def calculate_Ps(M):
    Ps = np.zeros(5)
    for i in range(1,6):
        s = 1
        for j in range(i):
            s = s*(M-j)
        s = s*S_2[i-1]/(M**5)
        Ps[i-1] = s
    return Ps

@jit
def calculate_chi_st_poker(Y,M):
    Os = np.zeros(5)
    n = len(Y)
    k = int(np.floor(n/5))
    for i in range(k):
        j = count(Y[(5*i):(5*i+5)])-1
        Os[j] = Os[j]+1
    p = k*calculate_Ps(M)
    return np.sum(((Os-p)**2)/p)


def chi_sq_poker_test(Y,M):
    statistic = calculate_chi_st_poker(Y,M)
    return 1 - stats.chi2.cdf(statistic, 4)


##furier_test


def Furier_test(Y):
    Y = Y<=0.5
    Xs = 2*Y-1
    S = np.fft.fft(Xs)
    n = len(S)
    M = np.abs(S[:int(n/2)])
    T = np.sqrt(n*np.log(1/0.05))
    N_0 = 0.95*n/2
    N_1 = np.sum(M<T)
    d = (N_1-N_0)/np.sqrt(n*0.95*0.05/4)

    p = erfc(np.abs(d)/np.sqrt(2))
    return p

##part 2

#read pi, sqrt2, e
def read_digits(url):
    data = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            data.append(line.strip())
    datastring = []

    for line in data:
        datastring.append(line.decode("utf-8"))

    datastring = ''.join(datastring)
    datastring = list(map(int, list(datastring)))

    return (np.array(datastring))

@jit
def change_to_normal(B,n=5000):
    N = len(B)
    k = int(np.floor(N/n))
    C = 2*B-1
    S = np.zeros(k)
    for i in range(k):
        s = 0
        for j in range(n):
            s = s+C[i*n+j]
        s = s*1/(n)**0.5
        S[i] = s
    return S


def read_numbers_and_save_p_values(n = 5000):

    digits_pi = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.pi')
    digits_e = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.e')
    digits_sqrt2 = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.sqrt2')


    p_val_pi = 2*(1-stats.norm.cdf(np.abs(change_to_normal(digits_pi,n =n))))
    p_val_e = 2*(1-stats.norm.cdf(np.abs(change_to_normal(digits_e,n=n))))
    p_val_sqrt2 = 2*(1-stats.norm.cdf(np.abs(change_to_normal(digits_sqrt2,n=n))))

    N_pi = len(p_val_pi)
    N_e = len(p_val_e)
    N_sqrt2 = len(p_val_sqrt2)



    df_pi = pd.DataFrame({"P-value": p_val_pi, "Liczba": ["Pi" for _ in range(N_pi)],
                          "Test": ["monobit" for _ in range(N_pi)],"n":[n for _ in range(N_pi)]})
    df_e = pd.DataFrame({"P-value": p_val_e, "Liczba": ["e" for _ in range(N_e)],
                         "Test": ["monobit" for _ in range(N_e)],"n":[n for _ in range(N_e)]})
    df_sqrt2 = pd.DataFrame({"P-value": p_val_sqrt2, "Liczba": ["sqrt2" for _ in range(N_sqrt2)],
                             "Test": ["monobit" for _ in range(N_sqrt2)],"n":[n for _ in range(N_e)]})
    df = pd.concat([df_pi, df_e,df_sqrt2], ignore_index=True)
    df.to_csv('wyniki_zad2_n_5000.csv', index=False)

#read_numbers_and_save_p_values(n = 5000)

###generowanie wynikow zad 1


def policz_i_zapisz_wyniki_zad1(n = 100,k = 100):
    N = n*k


    GLCG_gen = GLCG(n = N)
    RC_gen = PRGA(n = N)
    Ziff_gen = Ziff(Xs = GLCG(n = 9689),n = N)
    Mar_gen = Marsagli(Xs = GLCG(n = 256),n = N)



    generators_2 = [[GLCG_gen,"GLCG",M_GLCG],[RC_gen,"RC",M_RC],[Ziff_gen,"Ziff",M_ZIFF],[Mar_gen,"Marsagli",M_MAR]]

    GLCG_stand = GLCG_gen/M_GLCG
    RC_stand = RC_gen/M_RC
    Ziff_stand = Ziff_gen/M_ZIFF
    Mar_stand = Mar_gen/M_MAR

    generators = [[GLCG_stand,"GLCG"],[RC_stand,"RC"],[Ziff_stand,"Ziff"],[Mar_stand,"Marsagli"]]

    tests =[[KS_test,"KS"],[chi_sq_test,"Chi"],[Furier_test,"Furier"]]


    df = pd.DataFrame()
    for i in range(k):
        for gen,gen_name in generators:
            for test,test_name in tests:
                p = test(gen[(i*n):((i+1)*n)])
                new_row = pd.DataFrame({"P-value": [p], "Test": [test_name], "Generator": [gen_name],"n":[n]})
                df = pd.concat([df, new_row], ignore_index=True)

        for gen,gen_name,M in generators_2:
            for test,test_name in [[chi_sq_poker_test,"poker"]]:
                p = test(gen[(i*n):((i+1)*n)],M)
                new_row = pd.DataFrame({"P-value": [p], "Test": [test_name], "Generator": [gen_name],"n":[n]})
                df = pd.concat([df, new_row], ignore_index=True)



    df.to_csv('wyniki_zad1.csv', index=False)


#   policz_i_zapisz_wyniki_zad1(n = 1000,k = 1000)

