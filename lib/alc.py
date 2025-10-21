import numpy as np 

def esCuadrada (a):
        if len(a) == len(a[0]):
            return True
        else:
            return False
        
def triangSup (a):
    for indexF, fila in enumerate(a):
        for indexE, element in enumerate(fila):
            if (indexE <= indexF):
                a[indexF][indexE] = 0
    return a

def triangInf (a):
    for indexF, fila in enumerate(a):
        for indexE, element in enumerate(fila):
            if (indexE >= indexF):
                a[indexF][indexE] = 0
    return a

def diagonal (a):
    for indexF, fila in enumerate(a):
        for indexE, element in enumerate(fila):
            if (indexE != indexF):
                a[indexF][indexE] = 0
    return a

def traza (a):
    res = 0
    for indexF, fila in enumerate(a):
        for indexE, element in enumerate(fila):
            if (indexE == indexF):
                res += a[indexF][indexE]
    return res

def traspuesta (a):
    for indexF, fila in enumerate(a):
        for indexE, element in enumerate(fila):
            if (indexE != indexF):
                a[indexF][indexE] = 0
    return a

def esSimetrica (a):
    return a == traspuesta(a)

def calcularAx(a, x):
    return np.dot(a, x)

def intercambiarFIlas (a, i, j):
    a[i], a[j] = a[j], a[i]

def sumar_fila_multiplo(a,i,j,s):
    a[i] = np.sum(a[i] + s * a[j])


