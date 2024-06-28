import unittest
import numpy as np
from analysis import Analysis


class TestAnalysisMethods(unittest.TestCase):

    def test_l1_norm(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = Analysis.l1_norm(a, b)
        print(f"result: {result}")
        self.assertEqual(result, 9)

    def test_l1_norm_ndim(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        result = Analysis.l1_norm_ndim(a, b)
        print(f"result: {result}")
        self.assertTrue(np.array_equal(result, np.array([18, 18])))

    def test_l2_norm(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = Analysis.l2_norm(a, b)
        print(f"result: {result}")
        self.assertAlmostEqual(result, 5.196152, places=6)

    def test_l2_norm_ndim(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        result = Analysis.l2_norm_ndim(a, b)
        print(f"result: {result}")
        self.assertTrue(np.allclose(result, np.array([10.392304, 10.392304])))

    def test_lp_norm(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = Analysis.lp_norm(a, b, 2)
        print(f"result: {result}")
        self.assertAlmostEqual(result, 5.196152, places=6)

    def test_lp_norm_ndim(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        result = Analysis.lp_norm_ndim(a, b, 2)
        print(f"result: {result}")
        self.assertTrue(np.allclose(result, np.array([10.392304, 10.392304])))

    def test_l_inf_norm(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = Analysis.l_inf_norm(a, b)
        print(f"result: {result}")
        self.assertEqual(result, 3)

    def test_l_inf_norm_ndim(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        result = Analysis.l_inf_norm_ndim(a, b)
        print(f"result: {result}")
        self.assertTrue(np.array_equal(result, np.array([6, 6])))


if __name__ == '__main__':
    unittest.main()

