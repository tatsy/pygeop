import re
import numpy as np

from .vector import Vector

def parse_face(items):
    pat = re.compile('([0-9]+)\/([0-9]+)\/([0-9]+)')
    mat = pat.search(items[0])
    if mat is None:
        pat = re.compile('([0-9]+)\/\/([0-9]+)')
        mat = pat.search(items[0])

    if mat is None:
        pat = re.compile('([0-9]+)\/([0-9]+)')
        mat = pat.search(items[0])

    if mat is None:
        pat = re.compile('([0-9]+)')
        mat = pat.search(items[0])

    indices = [ int(pat.search(it).group(1)) - 1 for it in items ]
    return indices

class ObjMesh(object):
    def __init__(self, filename=None):
        self._vertices = np.array([], dtype=np.float32)
        self._normals = np.array([], dtype=np.float32)
        self._texcoords = np.array([], dtype=np.float32)
        self._indices = np.array([], dtype=np.uint32)

        if filename is not None:
            self.load(filename)

    # Property for vertices
    def _get_vertices(self):
        return self._vertices

    def _set_vertices(self, vertices):
        self._vertices = np.array(vertices, dtype=np.float32).flatten()
        self.compute_normals()

    vertices = property(_get_vertices, _set_vertices)

    # Property for normals
    def _get_normals(self):
        return self._normals

    def _set_normals(self, normals):
        self._normals = np.array(normals, dtype=np.float32).flatten()

    normals = property(_get_normals, _set_normals)

    # Property for texcoords
    def _get_texcoords(self):
        return self._texcoords

    def _set_texcoords(self, texcoords):
        self._texcoords = np.array(texcoords, dtype=np.float32).flatten()

    texcoords = property(_get_texcoords, _set_texcoords)

    # Property for indices
    def _get_indices(self):
        return self._indices

    def _set_indices(self, indices):
        self._indices = np.array(indices, dtype=np.uint32).flatten()

    indices = property(_get_indices, _set_indices)

    # Load mesh
    def load(self, filename):
        with open(filename, 'r') as f:
            lines = [ l.strip() for l in f ]
            lines = filter(lambda l : l != '' and not l.startswith('#'), lines)

            vertices = []
            normals = []
            texcoords = []
            indices = []
            for l in lines:
                it = [ x for x in re.split('\s+', l.strip()) ]
                if it[0] == 'v':
                    it = [ float(i) for i in it[1:] ]
                    vertices.append(it)

                elif it[0] == 'vt':
                    texcoords.append((float(it[1]), float(it[2])))

                elif it[0] == 'vn':
                    it = [ float(i) for i in it[1:] ]
                    normals.append(it)

                elif it[0] == 'f':
                    it = it[1:]
                    indices.append(parse_face(it))
                else:
                    print('Unknown identifier: {}'.format(it[0]))

            if len(indices) > 0:
                self.indices = np.array(indices).flatten()

            if len(vertices) > 0:
                self.vertices = np.array(vertices).flatten()

            if len(normals) > 0:
                self.normals = np.array(normals).flatten()

            if len(texcoords) > 0:
                self.texcoords = np.array(texcoords).flatten()

    def save(self, filename):
        assert len(self.vertices) > 0
        assert self.vertices.ndim == 1
        assert self.vertices.dtype == np.float32
        assert self.indices.ndim == 1
        assert self.indices.dtype == np.uint32

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, self.vertices.size, 3):
                x = self.vertices[i + 0]
                y = self.vertices[i + 1]
                z = self.vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write normals
            has_normal = False
            if self.normals.size > 0:
                has_normal = True
                for i in range(0, self.normals.size, 3):
                    x = self.normals[i + 0]
                    y = self.normals[i + 1]
                    z = self.normals[i + 2]
                    fp.write('vn {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write texcoords
            has_texcoord = False
            if self.texcoords.size >0:
                has_texcoord = True
                for i in range(0, self.texcoords.size, 2):
                    x = self.texcoords[i + 0]
                    y = self.texcoords[i + 1]
                    fp.write('vt {0:.8f} {1:.8f}\n'.format(x, y))

            # Write indices
            for i in range(0, len(self.indices), 3):
                i0 = self.indices[i + 0] + 1
                i1 = self.indices[i + 1] + 1
                i2 = self.indices[i + 2] + 1

                if has_normal and has_texcoord:
                    fp.write('f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n'.format(i0, i1, i2))

                elif has_texcoord:
                    fp.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(i0, i1, i2))

                elif has_normal:
                    fp.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(i0, i1, i2))

                else:
                    fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))

    def compute_normals(self):
        vectors = [ Vector(self.vertices[i + 0], self.vertices[i + 1], self.vertices[i + 2])
                    for i in range(0, self.vertices.size, 3) ]
        normals = [ Vector(0.0, 0.0, 0.0) for i in range(self.n_vertices()) ]

        for i in range(0, self.indices.size, 3):
            i0 = self.indices[i + 0]
            i1 = self.indices[i + 1]
            i2 = self.indices[i + 2]
            v0 = vectors[i0]
            v1 = vectors[i1]
            v2 = vectors[i2]
            normal = (v1 - v0).cross(v2 - v0)
            normals[i0] += normal
            normals[i1] += normal
            normals[i2] += normal

        for n in normals:
            n.normalize()

        self.normals = [ (n.x, n.y, n.z) for n in normals ]


    def n_vertices(self):
        return self.vertices.size // 3

    def n_normals(self):
        return self.normals.size// 3
