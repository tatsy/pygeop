try:
    import unittest2 as unittest
except:
    import unittest

from pygeop.geom3d import TriMesh, Vertex

class TestTriMesh(unittest.TestCase):
    def test_load(self):
        try:
            mesh = TriMesh('data/box.obj')
        except:
            self.fail("TriMesh.load unexpectedly raised an exception!")

    def test_size_parameters(self):
        mesh = TriMesh('data/box.obj')
        self.assertEqual(mesh.n_vertices(), 8)
        self.assertEqual(mesh.n_faces(), 12)

if __name__ == '__main__':
    unittest.main()
