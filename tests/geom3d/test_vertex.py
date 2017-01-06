try:
    import unittest2 as unittest
except:
    import unittest

from random import *
from pygeop.geom3d import Vertex, Vector

class TestVertex(unittest.TestCase):
    def test_init(self):
        x = random()
        y = random()
        z = random()
        v = Vertex(x, y, z)
        self.assertEqual(v.x, x)
        self.assertEqual(v.y, y)
        self.assertEqual(v.z, z)

    def test_position(self):
        v = Vertex()
        x = random()
        y = random()
        z = random()
        v.position = Vector(x, y, z)
        self.assertEqual(v.x, x)
        self.assertEqual(v.y, y)
        self.assertEqual(v.z, z)

        p = v.position
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        self.assertEqual(p.z, z)

if __name__ == '__main__':
    unittest.main()
