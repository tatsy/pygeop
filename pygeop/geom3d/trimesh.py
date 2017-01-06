import re
import math

from .vector import Vector
from .vertex import Vertex
from .halfedge import Halfedge
from .face import Face

class TriMesh(object):
    def __init__(self, filename=''):
        self.vertices = []
        self.halfedges = []
        self.faces = []
        self.indices = []

        if filename != '':
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as fp:
            self.clear()
            temp_vertices = []
            temp_indices = []
            for l in fp:
                l = l.strip()
                if l.startswith('v '):
                    v = [ float(it) for it in re.split('\s+', l)[1:] ]
                    temp_vertices.append((v[0], v[1], v[2]))

                if l.startswith('f '):
                    f = [ int(it) - 1 for it in re.split('\s+', l)[1:] ]
                    temp_indices.extend([ f[0], f[1], f[2] ])

            unique_vertices = {}
            for i in temp_indices:
                v = temp_vertices[i]

                if v not in unique_vertices:
                    unique_vertices[v] = len(self.vertices)
                    self.vertices.append(Vertex(v[0], v[1], v[2]))
                    self.vertices[-1].index = unique_vertices[v]

                self.indices.append(unique_vertices[v])

        self._make_halfedge()

    def save(self, filename):
        with open(filename, 'w') as fp:
            for v in self.vertices:
                fp.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(v.x, v.y, v.z))

            for i in range(0, len(self.indices), 3):
                i0 = self.indices[i + 0] + 1
                i1 = self.indices[i + 1] + 1
                i2 = self.indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))

    def n_vertices(self):
        return len(self.vertices)

    def n_faces(self):
        return len(self.faces)

    def _make_halfedge(self):
        table = [ [] for i in range(len(self.vertices)) ]

        for i in range(0, len(self.indices), 3):
            he0 = Halfedge()
            he1 = Halfedge()
            he2 = Halfedge()

            he0.vertex_from = self.vertices[self.indices[i + 0]]
            he1.vertex_from = self.vertices[self.indices[i + 1]]
            he2.vertex_from = self.vertices[self.indices[i + 2]]

            he0.vertex_to = self.vertices[self.indices[i + 1]]
            he1.vertex_to = self.vertices[self.indices[i + 2]]
            he2.vertex_to = self.vertices[self.indices[i + 0]]

            assert he0.vertex_from.index != he0.vertex_to.index
            assert he1.vertex_from.index != he1.vertex_to.index
            assert he2.vertex_from.index != he2.vertex_to.index

            he0.next = he1
            he1.next = he2
            he2.next = he0

            self.vertices[self.indices[i + 0]].halfedge = he0
            self.vertices[self.indices[i + 1]].halfedge = he1
            self.vertices[self.indices[i + 2]].halfedge = he2

            face = Face()
            face.halfedge = he0

            self.halfedges.extend([ he0, he1, he2 ])
            self.faces.append(face)

            table[self.vertices[self.indices[i + 0]].index].append(he0)
            table[self.vertices[self.indices[i + 1]].index].append(he1)
            table[self.vertices[self.indices[i + 2]].index].append(he2)

        for he0 in self.halfedges:
            for he1 in table[he0.vertex_to.index]:
                if he0.vertex_from == he1.vertex_to and \
                   he1.vertex_from == he0.vertex_to:

                   he0.opposite = he1
                   he1.opposite = he0
                   break

            assert he0.opposite is not None

    def clear(self):
        self.vertices.clear()
        self.halfedges.clear()
        self.faces.clear()
        self.indices.clear()
