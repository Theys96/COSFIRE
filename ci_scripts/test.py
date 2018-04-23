import unittest

class TestMethods(unittest.TestCase):

    def test_import(self):
        import cosfire

    # Simple test for initialization and running of some of the library's components
    def test_init(self):
        import cosfire as c
        import numpy as np
        from PIL import Image

        # Prototype image
        proto = np.asarray(Image.open('prototype.png').convert('L'), dtype=np.float64)
        cx, cy = (50,50)

        # Create COSFIRE operator and fit it with the prototype
        cosf = c.COSFIRE(
                c.CircleStrategy, c.DoGFilter, ([1,2,3], 1), [0,10,20,40]
               ).fit(proto, (cx, cy))


if __name__ == '__main__':
    unittest.main()