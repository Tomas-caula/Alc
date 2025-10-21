import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.alc import *

class TestTallerUno(unittest.TestCase):
    #def test_esCuadrada(self):
    #    self.assertTrue(esCuadrada([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    #    self.assertFalse(esCuadrada([[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]))

    #def test_triangInf(self):
    #    self.assertEqual(triangInf([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[1, 2, 3], [0, 5, 6], [0, 0, 9]])
    #    self.assertEqual(triangInf([[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]), [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]])

    def test_intercambiarFIlas(self):
        matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        intercambiarFIlas(matrix1, 0, 1)
        self.assertEqual(matrix1, [[4, 5, 6], [1, 2, 3], [7, 8, 9]])

        matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
        intercambiarFIlas(matrix2, 0, 1)
        self.assertEqual(matrix2, [[4, 5, 6], [1, 2, 3], [7, 8, 9, 10]])

        # Also test that the function returns None, as it modifies in-place
        matrix3 = [[1, 2], [3, 4]]
        self.assertIsNone(intercambiarFIlas(matrix3, 0, 1))

    def test_sumar_fila_multiplo(self):
        matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        sumar_fila_multiplo(matrix1, 0, 1, 2)
        self.assertEqual(matrix1, [9, 12, 15], [4, 5, 6], [7, 8, 9])

        matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        sumar_fila_multiplo(matrix2, 1, 2, 2)
        self.assertEqual(matrix2, [9, 12, 15], [18, 21, 24], [7, 8, 9])
if __name__ == '__main__':
    unittest.main()
