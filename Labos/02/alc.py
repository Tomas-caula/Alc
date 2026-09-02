import numpy as np

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def escala(s):
    A = np.zeros((len(s), len(s)))
    for i in range(len(s)):
        A[i, i] = s[i]
    return A

def rota_y_escala(theta, s):
    # Implementar la función rota_y_escala que reciba un ángulo theta y una tira de números s, y retorne una matriz de 2 x 2 que rota el vector en un ángulo theta y luego lo escala en un factor s.
    return escala(s) @ rota(theta)

def afin(theta, s, b):
    A = np.zeros((3, 3))
    A[0:2, 0:2] = rota_y_escala(theta, s)
    A[0:2, 2] = b
    A[2, 2] = 1    
    return A

def trans_afin(v, theta,s, b):
    res = afin(theta, s, b) @ np.array([v[0], v[1], 1])
    return res[0:2]


