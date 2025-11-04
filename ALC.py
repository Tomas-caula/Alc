# MÓDULO ALC
import numpy as np

def esCuadrada (A):
    return A.shape[0] == A.shape[1]

def triangSup (A):
    U = A
    n = U.shape[0]

    for i in range(n):
        for j in range(n):
            if i >= j:
                U[i][j] = 0

    return U

def triangInf (A):
    L = A
    n = L.shape[0]

    for i in range(n):
        for j in range(n):
            if i <= j:
                L[i][j] = 0

    return L

def diagonal (A):
    D = A
    n = D.shape[0]

    for i in range(n):
        for j in range(n):
            if i > j or i < j:
                D[i][j] = 0

    return D

def traza (A):
    n = A.shape[0]
    tr = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                tr += A[i][j]

    return tr

def traspuesta (A):
    f, c = A.shape
    At = np.zeros((c, f))

    for i in range(f):
        for j in range(c):
            At[j][i] = A[i][j]
                            
    return At

def esSimetrica (A):
    return np.allclose(A, traspuesta(A))

def calcularAx (A, x):
    n = A.shape[0]
    m = A.shape[1]
    b =np.zeros(n)

    for i in range(n):
        suma = 0
        for j in range(m):
            suma += A[i][j]*x[j]
        b[i] = suma

    return b

def intercambiarFilas (A, i, j):
    fi = A[i].copy()
    A[i] = A[j]
    A[j] = fi

    return A

def sumar_fila_multiplo (A, i, j, s):
    A[i] = A[i] + (A[j]*s)

    return A

def esDiagonalmenteDominante (A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i][i])    
        suma = 0
        for j in range(n):
            if j != i:
                suma += abs(A[i][j])  
        if diag <= suma:       
            return False

    return True

def matrizCirculante (v):
    n = v.shape[0]
    C = np.zeros((n, n))
    v_copia = v.copy()

    for i in range(n):
        C[i] = v_copia
        v_copia = np.array([v_copia[-1]] + list(v_copia[:-1]))

    return C

def matrizVandermonde (v):
    n = v.shape[0]
    V = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            V[i][j] = v[i] ** (n-1-j)
    
    return V

def numeroAureo (n):
    M = np.array([[1, 1],
                  [1, 0]])
    v = np.array([1, 0])
    
    phi_vals = []

    for _ in range(n):
        if v[1] != 0:
            phi_vals.append(v[0] / v[1])
    
        v_new = np.zeros(2)
        for i in range(2):
            suma = 0
            for j in range(2):
                suma += M[i][j] * v[j]
            v_new[i] = suma
        
        v = v_new

    return phi_vals

def matrizFiboncacci (n):
    F = np.zeros((n, n))

    F_lista = [0, 1]
    for _ in range(2, 2*n - 1):
        F_lista.append(F_lista[-1] + F_lista[-2])

    for i in range(n):
        for j in range(n):
            F[i][j] = F_lista[i+j]

    return F

def matrizHilbert (n):
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            H[i][j] = 1 / (i + j + 1)

    return H

def calcularValores():
    x = np.linspace(-1, 1, 200)

    coefs1 = np.array([1, -1, 1, -1, 1, -1])
    coefs2 = np.array([1, 0, 3])              
    coefs3 = np.array([1] + [0]*9 + [-2])     

    # Usamos la función matrizVandermonde para construir las potencias de x
    V1 = matrizVandermonde(x)
    V2 = matrizVandermonde(x)
    V3 = matrizVandermonde(x)

    V1 = V1[:, -len(coefs1):]  # usamos solo las potencias necesarias
    V2 = V2[:, -len(coefs2):]
    V3 = V3[:, -len(coefs3):]

    y1 = V1 @ coefs1
    y2 = V2 @ coefs2
    y3 = V3 @ coefs3

    return y1, y2, y3

def row_echelon_mod(A):
    """
    Devuelve la matriz A en forma escalonada por filas,
    aplicando pivoteo parcial (swap por máximo absoluto en cada columna).
    Usa las funciones 'intercambiarFilas' y 'sumar_fila_multiplo'.
    """
    A = np.copy(A).astype(float)
    f, c = A.shape
    if f == 0 or c == 0:
        return A

    col = 0
    max_fila = np.argmax(np.abs(A[:, col]))
    max_valor = np.abs(A[max_fila, col])

    if max_valor == 0:
        B = row_echelon_mod(A[:, 1:])
        return np.block([A[:, :1], B])

    if max_fila != 0:
        A = intercambiarFilas(A, 0, max_fila)

    pivote = A[0, col]
    for i in range(1, f):
        if A[i, col] != 0:
            factor = -A[i, col] / pivote
            A = sumar_fila_multiplo(A, i, 0, factor)

    submatriz = A[1:, 1:]
    B = row_echelon_mod(submatriz)

    return np.block([
        [A[:1, :]], 
        [np.column_stack((A[1:, :1], B))]
    ])

def error(x,y):
    x = np.float64(x)
    y = np.float64(y)
    return abs(x - y)

def errorRelativo(x,y):
    x = np.float64(x)
    y = np.float64(y)
    if x == 0:
        raise ValueError("El error relativo no está definido para x = 0")
    return abs(x - y) / abs(x)

def matricesIguales(A,B):
    return np.array_equal(A, B)

def sonIguales(x, y, atol=1e-8):
    return np.allclose(error(x, y), 0, atol=atol)

def rota (theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    return R

def escala (s):
    s = np.array(s)
    return np.diag(s)

def rota_y_escala (theta, s):
    M = escala(s) @ rota(theta)
    return M

def afin (theta, s, b):
    b = np.array(b)
    s = np.array(s)

    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    S = np.array([[s[0], 0, 0],
                  [0, s[1], 0],
                  [0, 0, 1]])
    
    T = np.array([[1, 0, b[0]],
                  [0, 1, b[1]],
                  [0, 0, 1]])
    
    A = T @ S @ R
    return A

def trans_afin (v, theta, s, b):
    b = np.array(b)
    Tr = rota_y_escala(theta, s) @ v
    vr = Tr + b
    return vr

# if __name__ == "__main__":
    # Tests para rota
    assert np.allclose(rota(0), np.eye(2))
    assert np.allclose(rota(np.pi/2), np.array([[0, -1], [1, 0]]))
    assert np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))

    # Tests para escala
    assert np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]]))
    assert np.allclose(escala([1, 1]), np.eye(2))
    assert np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]]))

    # Tests para rota_y_escala
    assert np.allclose(rota_y_escala(0, [2, 3]), np.array([[2, 0], [0, 3]]))
    assert np.allclose(rota_y_escala(np.pi/2, [1, 1]), np.array([[0, -1], [1, 0]]))
    assert np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]]))

    # Tests para afin
    assert np.allclose(
        afin(np.eye(2), 0, [1, 1], [1, 2]),
        np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
    )
    assert np.allclose(
        afin(rota(np.pi/2), 1, [1, 0], [0, 0]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    )
    assert np.allclose(
        afin(np.eye(2), 0, [2, 3], [1, 1]),
        np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]])
    )

    # Tests para trans_afin
    assert np.allclose(
        trans_afin(np.array([1, 0]), np.pi/2, [1, 1], [0, 0]),
        np.array([0, 1])
    )
    assert np.allclose(
        trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
        np.array([2, 3])
    )
    assert np.allclose(
        trans_afin(np.array([1, 0]), np.pi/2, [3, 2], [4, 5]),
        np.array([4, 7])
    )

    print("Todos los tests pasaron correctamente")

def norma (x, p):
    n = x.shape[0]
    suma = 0

    for i in range(n):
        suma += abs(x[i]) ** p
    
    norma = suma ** (1/p)

    return norma

def normaliza (X, p):
    l = len(X)
    Xn = []

    for i in range(l):
        Xn.append(X[i] / norma(X[i], p))

    return Xn

def normaMatMC (A, q, p, Np):
    n = A.shape[1]
    max_val = 0
    v = np.zeros(n)

    for i in range(Np):
        x = np.random.rand(n)
        xq = x / norma(x, q)
        Ax = norma(A@x, p)
        max_act = Ax

        if max_act > max_val:
            max_val = max_act
            v = x

    return v, max_val

def normaExacta(A, p=[1, 'inf']):
    normas = []
    
    if 1 in p:
        # Norma 1: máximo de la suma por columnas
        normas.append(np.max(np.sum(np.abs(A), axis=0)))
        
    if 'inf' in p:
        # Norma infinito: máximo de la suma por filas
        normas.append(np.max(np.sum(np.abs(A), axis=1)))
    
    return normas

def condMC(A, p):
    return normaliza(A, p) * normaliza(inversa(A), p)

def condExacto(A, p):
    norma_A = normaExacta(A, p)[0]
    norma_A_inv = normaExacta(inversa(A), p)[0]
    cond = norma_A * norma_A_inv
    
    return cond

def vectorT(v):
    return [[x] for x in v]

def prodVectorial (v, w):
    n = v.shape[0]
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i][j] = v[i] * w[j]

    return M

def prodMat (A, B):
    m = A.shape[0]
    p = B.shape[1]
    M = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            M[i][j] = producto_escalar(A[i,:], B[:,j])
    return M

def prodMatV(A, v):
    m, _ = A.shape
    w = np.zeros(m)
    for i in range(m):
        w[i] = producto_escalar(A[i,:], v) 
    return w

def producto_escalar(v, w):
    return sum(v[i] * w[i] for i in range(len(v)))

def identidad(n):
    I = np.zeros((n, n))
    for i in range(n):
        I[i][i] = 1
    return I

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def calculaLU(A): 
    n = A.shape[0] # Obtener el tamaño de la matriz cuadrada  
    U = np.array([[float(A[i][j]) for j in range(n)] for i in range(n)]) # Inicializar U como una copia de la matriz A 
    L = identidad(n).astype(float) # Inicializar L como una matriz identidad 
    ops = 0 
    
    for i in range(n): 
        # Verificar si el pivote es cero (no se puede dividir) 
        if U[i][i] == 0: 
            return None 
        
        for j in range(i+1, n): 
            L[j][i] = U[j][i] / U[i][i] # Calcular el factor de escalamiento: L[j][i] = U[j][i] / U[i][i] 
            ops += 1 

            for k in range(i, n): 
                U[j][k] = U[j][k] - L[j][i] * U[i][k] # U[j][k] = U[j][k] - L[j][i] * U[i][k], para todo k >= i 
                ops += 2 

    return L, U, ops

def res_tri(L, b, inferior=True):
    n = np.shape(L)[0]
    x = np.zeros(n)

    if inferior:
        for i in range(n):
            suma = 0
            for j in range(i):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
    else:
        for i in range(n-1, -1, -1):
            suma = 0
            for j in range(i+1, n):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]

    return x

def inversa(A):
    n = np.shape(A)[0]
    invA = np.zeros((n, n))
    I = identidad(n)
    L, U, _ = calculaLU(A)

    for i in range(n):
        e_i = I[:, i]
        yk = res_tri(L, e_i, inferior=True)
        xk = res_tri(U, yk, inferior=False)
        invA[:, i] = xk

    return invA

def calculaLDV(A):
    L, U, _ = calculaLU(A)
    Ut = traspuesta(U)
    V, D, _ = calculaLU(Ut)
    V = traspuesta(V)

    return L, D, V

def diagonalPositiva(A):
    n = np.shape(A)[0]

    for i in range(n):
        if A[i][i] > 0:
            return True
        else:
            return False

def esSDP (A):
    _, D, _ = calculaLDV(A)

    if esSimetrica(A) and diagonalPositiva(D):
        return True
    else:
        return False
    
def limpiar(R, tol=1e-12):
    f, c = R.shape
    for i in range(f):
        for j in range(c):
            if R[i][j] < tol and R[i][j] > -tol:
                R[i][j] = 0
    return R
    
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    if A.shape[0] != A.shape[1]:
        return None
    else:
        n = A.shape[0]
        Q = np.zeros((n, n))
        R = np.zeros((n, n))
        ops = 0

        Q[:,0] = A[:,0]/norma(A[:,0], 2)
        R[0][0] = norma(A[:,0], 2)

        for j in range(1, n):
            Q[:,j] = A[:,j]
            ops += 1

            for k in range(j):
                R[k][j] = producto_escalar(Q[:,k], Q[:,j])
                Q[:,j] = Q[:,j] - R[k][j] * Q[:,k]
                ops += 3
            
            R[j][j] = norma(Q[:,j], 2)
            Q[:,j] = Q[:,j]/R[j][j]
            ops += 1

    return Q, R, ops

def QR_con_HH(A,tol=1e-12):
    if A.shape[0] < A.shape[1]:
        return None
    else:
        m = A.shape[0]
        n = A.shape[1]
        R = A.copy()
        Q = identidad(m)

        for k in range(n):
            x = R[k:m, k]
            alpha = -sign(x[0]) * norma(x, 2)
            e = np.zeros(m-k)
            e[0] = 1
            u = x - alpha * e

            if norma(u, 2) > tol:
                u = u / norma(u, 2)
                Hk = identidad(m-k) - 2 * prodVectorial(u, u)
                H = identidad(m).astype(float)
                H[k:, k:] = Hk

                R = prodMat(H, R)
                R = limpiar(R)
                Q = prodMat(Q, traspuesta(H))

    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    else:
        return None
    
def f_A(A, v):
    Av = prodMatV(A, v)
    nrm = norma(Av, 2)
    if nrm > 0:
        return Av / nrm
    else:
        return Av

def metpot2k (A, tol=1e-15, K=1000):
    n = A.shape[0]
    v = np.random.rand(n)
    v_tilde = f_A(A, f_A(A, v))
    e = producto_escalar(v_tilde, v)
    k = 0

    while(abs(e - 1) > tol and k < K):
        v = v_tilde
        v_tilde = f_A(A, f_A(A, v))
        e = producto_escalar(v_tilde, v)
        k += 1
    
    lambda_A = producto_escalar(v_tilde, prodMatV(A, v_tilde))
    error = e - 1
    
    return v, lambda_A, k, error

def diagRH (A, tol = 1e-15, K = 1000):
    n = A.shape[0]
    v1, l1, _, _ = metpot2k (A, tol, K)

    e1 = np.zeros(n)
    e1[0] = 1
    u = e1 - v1
    u_norm = norma(u, 2)

    if u_norm == 0:
        H = identidad(n)
    else:
        H = identidad(n) - (2/(u_norm ** 2)) * prodVectorial(u, u)

    if n == 2:
        S = H
        D = prodMat(H, prodMat(A, traspuesta(H)))
    else:
        B = prodMat(H, prodMat(A, traspuesta(H)))
        A_tilde = B[1:, 1:]

        S_tilde, D_tilde = diagRH(A_tilde, tol, K)

        D = np.zeros((n, n))
        D[0, 0] = l1
        D[1:, 1:] = D_tilde

        S = identidad(n)
        for i in range(1, n):
            for j in range(1, n):
                S[i][j] = S_tilde[i-1][j-1]
                
        S = prodMat(H, S)

    return S, D

def transiciones_al_azar_continuas(n):
    T = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            T[i, j] = np.random.rand()

    for j in range(n):
        col = T[:, j]
        suma = np.sum(col)
        if suma != 0:
            T[:, j] = col / suma
        else:
            T[:, j] = np.ones(n) / n

    return T

def transiciones_al_azar_uniformes(n, thres):
    T = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if np.random.rand() < thres:
                T[i, j] = 1

    # Aseguro que cada columna tenga al menos un 1
    for j in range(n):
        if np.sum(T[:, j]) == 0:
            # Elegir una fila aleatoria y poner un 1
            fila = np.random.randint(0, n)
            T[fila, j] = 1

    for j in range(n):
        col = T[:, j]
        norm = np.sum(col)  # como son ceros y unos, suma == cantidad de unos
        T[:, j] = col / norm

    return T

def nucleo(A,tol=1e-15):
    A = prodMat(traspuesta(A), A)

    S, _ = diagRH(A, tol, K = 1000)

    return S

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    elemsNoNulos = {}
    dim = (m_filas,n_columnas)
    
    for i, j, aij in zip(listado[0], listado[1], listado[2]):
        if abs(aij) >= tol:
            elemsNoNulos[(i, j)] = aij
    
    return elemsNoNulos, dim

def multiplica_rala_vector(A,v):
    w = np.zeros(len(v))
    
    for (i, j), aij in A.items():
        w[i] += aij * v[j]
    return w

def es_markov(T,tol=1e-6):
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True

def esNucleo(A,S,tol=1e-5):
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

def svd_reducida(A,k="max",tol=1e-15):
    AtA = prodMat(traspuesta(A), A)
    V, D = diagRH(AtA, tol=tol, K=1000)

    sigma = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        if D[i][i] > tol:
            sigma[i] = D[i][i] ** 0.5
        else:
            sigma[i] = 0

    k = 0
    for i in range(len(sigma)):
        if sigma[i] > tol:
            k += 1

    SigmaM = np.zeros((k, k))
    for i in range(k):
        SigmaM[i][i] = sigma[i]

    V = V[:, :k]

    inv_SigmaM = np.zeros((k, k))
    for i in range(k):
        inv_SigmaM[i][i] = 1 / SigmaM[i][i]

    # Armo U
    U = prodMat(A, prodMat(V, inv_SigmaM))

    # Normalizo las columnas de U
    m, n = U.shape
    for j in range(n):
        norma = 0
        for i in range(m):
            norma += U[i][j] ** 2
        norma = norma ** 0.5
        for i in range(m):
            U[i][j] /= norma

    return U, sigma, traspuesta(V)