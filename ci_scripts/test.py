import unittest

class TestMethods(unittest.TestCase):

    def test_import(self):
        import cosfire

    def test_DOGfilter(self):
        import cosfire
        sigma = 2.6
        filter_DoG = cosfire.DoGFilter(sigma,1) #onoff = 1
        mathlab_output = [0]
        self.assertEqual(filter, mathlab_output)

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