import unittest

class TestMethods(unittest.TestCase):

    def test_import(self):
        import cosfire

    def test_gaussianfilter(self):
        import cosfire
        sigma = 2.6
        filter_gaussian = cosfire.GaussianFilter(sigma)
        mathlab_output = ([[0.09564854],[0.13844912],[0.1728452 ],[0.18611428],[0.1728452 ],[0.13844912],[0.09564854]])
        self.assertEqual(filter_gaussian, mathlab_output)

    def test_DoGfilter(self):
        import cosfire
        sigma, onoff = 1, 1
        filter_DoG = cosfire.DoGFilter(sigma, onoff)
        mathlab_output = ([-1.9652,  -2.3941,  -1.0730,  -1.7690,  -1.0730,  -2.3941,  -1.9652],
                        [-2.3941,  -2.9165,  -1.3043,  -2.1343,  -1.3043,  -2.9165,  -2.3941],
                        [-1.0730,  -1.3043,  -4.7250,  -1.2854,  -4.7250,  -1.3043,  -1.0730],
                        [-1.7690,  -2.1343,  -1.2854,   4.5945,  -1.2854,  -2.1343,  -1.7690],
                        [-1.0730,  -1.3043,  -4.7250,  -1.2854,  -4.7250,  -1.3043,  -1.0730],
                        [-2.3941,  -2.9165,  -1.3043,  -2.1343,  -1.3043,  -2.9165,  -2.3941],
                        [-1.9652,  -2.3941,  -1.0730,  -1.7690,  -1.0730,  -2.3941,  -1.9652])
        self.assertEqual(filter_DoG, mathlab_output)

    def test_gaborfilter(self):
        import cosfire
        import math
        sigma, theta, lambd, gamma, psi = 1, math.pi/2, 0, 1, 0
        filter_gaussian = cosfire.GaborFilter(sigma, theta, lambd, gamma, psi)
        mathlab_output = [0]

"""
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
"""

if __name__ == '__main__':
    unittest.main()