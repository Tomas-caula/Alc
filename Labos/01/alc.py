import numpy as np

def error(x, y): 
    j = np.float64(y)
    return abs(x - j)

def error_relativo(x,y):
    j = np.float64(y)
    return (abs(x - j))/abs(x)


def matricesIguales(A,B):
    if A.shape != B.shape:
        return False
    if A.shape == B.shape:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                e = error(A[i][j], B[i][j])
                if not e < 1e-7:
                    return False
    return True

assert(matricesIguales(np.diag([1, 1]), np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2)))
assert(not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]])))