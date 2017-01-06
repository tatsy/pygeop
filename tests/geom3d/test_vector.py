try:
    import unittest2 as unittest
except:
    import unittest

from random import *
from pygeop import PygpException
from pygeop.geom3d import Vector

class TestVector(unittest.TestCase):
    def test_init(self):
        x = random()
        y = random()
        z = random()
        v = Vector(x, y, z)
        self.assertEqual(v.x, x)
        self.assertEqual(v.y, y)
        self.assertEqual(v.z, z)

    def test_dot(self):
        v1 = Vector(random(), random(), random())
        v2 = Vector(random(), random(), random())
        ans = v1.dot(v2)
        self.assertEqual(ans, v1.x * v2.x + v1.y * v2.y + v1.z * v2.z)

    def test_cross(self):
        v1 = Vector(random(), random(), random())
        v2 = Vector(random(), random(), random())
        v3 = v1.cross(v2)
        x = v1.y * v2.z - v1.z * v2.y
        y = v1.x * v2.x - v1.x * v2.z
        z = v1.z * v2.y - v1.y * v2.x
        self.assertEqual(v3.x, x)
        self.assertEqual(v3.y, y)
        self.assertEqual(v3.z, z)

    def test_normalize(self):
        v0 = Vector(0.0, 0.0, 0.0)
        with self.assertRaises(PygpException):
            v0.normalize()

        v0 = Vector(1.0, 2.0, 3.0)
        v0 = v0.normalize()
        self.assertAlmostEqual(v0.norm(), 1.0)

    def test_add(self):
        v1 = Vector(random(), random(), random())
        v2 = Vector(random(), random(), random())
        v3 = v1 + v2
        self.assertEqual(v3.x, v1.x + v2.x)
        self.assertEqual(v3.y, v1.y + v2.y)
        self.assertEqual(v3.z, v1.z + v2.z)

    def test_neg(self):
        v = Vector(random(), random(), random())
        u = -v
        self.assertEqual(u.x, -v.x)
        self.assertEqual(u.y, -v.y)
        self.assertEqual(u.z, -v.z)

    def test_sub(self):
        v1 = Vector(random(), random(), random())
        v2 = Vector(random(), random(), random())
        v3 = v1 - v2
        self.assertEqual(v3.x, v1.x - v2.x)
        self.assertEqual(v3.y, v1.y - v2.y)
        self.assertEqual(v3.z, v1.z - v2.z)

    def test_mul(self):
        v = Vector(random(), random(), random())
        s = random() + 1.0e-8

        u = v * s
        self.assertEqual(u.x, v.x * s)
        self.assertEqual(u.y, v.y * s)
        self.assertEqual(u.z, v.z * s)

        u = s * v
        self.assertEqual(u.x, v.x * s)
        self.assertEqual(u.y, v.y * s)
        self.assertEqual(u.z, v.z * s)

    def test_truediv(self):
        v = Vector(random(), random(), random())
        s = random() + 1.0e-8
        u = v / s
        self.assertEqual(u.x, v.x / s)
        self.assertEqual(u.y, v.y / s)
        self.assertEqual(u.z, v.z / s)

        with self.assertRaises(ZeroDivisionError):
            v / 0.0

    def test_repr(self):
        try:
            print(self)
        except ExceptionType:
            self.fail("Vector.__repr__ raised Exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()
