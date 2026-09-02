import numpy as np

def norma(x, p):
    if p == 'inf':
        max = x[0]
        for element in x:
            if abs(element) > max:
                max = abs(element)
        return max
    else:
        sum = 0
        for element in x:
            sum += abs(element)**p
        return sum**(1/p)

def normaliza(X, p):
    Xp = X.copy()
    for i in range(len(X)):
        n = norma(X[i], p)
        if n != 0:
            Xp[i] = X[i] / n
    return Xp

def normaMatMC(A,q,p,Np):
  """
  Devuelve la norma ||A||\_{q,p} y el vector x en el cual se alcanza el maximo.
"""
  maxX = 0
  maxV = 0
  vectores = np.random.rand(Np, A.shape[1])
  for vector in vectores:
    v = vector / norma(vector, p)
    Av = A @ v
    nAv = norma(Av, q)
    if nAv > maxV:
      maxV = nAv
      maxX = v
  return maxV, maxX


def normaExacta (A, p = [ 1 , 'inf' ] ) :
    max = 0 
    if p == 1:
        for j in range(A.shape[1]):
            sum = 0
            for i in range(A.shape[0]):
                sum += abs(A[i][j])
            if sum > max:
                max = sum
        return max
    elif p == 'inf':
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum += abs(A[i][j])
            if sum > max:
                max = sum
        return max

    
def condMC(A,p, Np):
    normaA = normaMatMC(A,p,p,Np)[0]
    A_ = np.linalg.inv(A)
    normaA_ = normaMatMC(A_,p,p,Np)[0]
    return normaA * normaA_

def condExacta(A,p):
    normaA = normaExacta(A,p)
    A_ = np.linalg.inv(A)
    normaA_ = normaExacta(A_,p)
    return normaA * normaA_