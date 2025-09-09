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
