import re
import math

from ..exception import PygpException
from .vertex import Vertex
from .halfedge import Halfedge
from .face import Face
from .objmesh import ObjMesh

class TriMesh(object):
    def __init__(self, filename=''):
        self.vertices = []
        self.halfedges = []
        self.faces = []
        self.indices = []

        if filename != '':
            self.load(filename)

    def load(self, filename):
        obj = ObjMesh(filename)

        unique_vertices = {}
        for i in obj.indices:
            vx = obj.vertices[i * 3 + 0]
            vy = obj.vertices[i * 3 + 1]
            vz = obj.vertices[i * 3 + 2]
            v = (vx, vy, vz)

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

    def collapse_halfedge(self, v_from, v_to, update_position=None):
        if v_from.degree() <= 3 or v_to.degree() <= 3:
            raise PygpException('Invalid collapse operation!')

        # Find target halfedge
        target_halfedge = None
        for he in v_from.halfedges():
            if he.vertex_from is v_from and he.vertex_to is v_to:
                target_halfedge = he
                break

        if target_halfedge is None:
            raise PygpException('Specified halfedge does not exist!')

        reverse_halfedge = target_halfedge.opposite

        # Boundary halfedge
        is_boundary = v_from.is_boundary and v_to.is_boundary
        if target_halfedge.face is None:
            target_halfedge, reverse_halfedge = reverse_halfedge, target_halfedge

        # Update v_to's halfedge
        target_halfedge.vertex_to.halfedge = target_halfedge.next.opposite.next

        # Update halfedges of surrounding vertices
        target_halfedge.next.vertex_to.halfedge = target_halfedge.next.opposite
        if not is_boundary:
            reverse_halfedge.next.vertex_to.halfedge = reverse_halfedge.next.opposite

        # Update topology
        he0 = target_halfedge.next.opposite
        he1 = target_halfedge.next.next.opposite
        he0.opposite, he1.opposite = he1, he0

        if not is_boundary:
            he2 = reverse_halfedge.next.opposite
            he3 = reverse_halfedge.next.next.opposite
            he2.opposite, he3.opposite = he3, he2

        for he in v_to.halfedges():
            he.vertex_from = v_to
            he.opposite.vertex_to = v_to

        # Remove faces
        self.faces[target_halfedge.face.index] = None
        if not is_boundary:
            self.faces[reverse_halfedge.face.index] = None

        # Delete halfedge
        self.halfedges[target_halfedge.index] = None
        self.halfedges[reverse_halfedge.index] = None

        # Delete/update vertex
        self.vertices[v_from.index] = None
        if update_position is not None:
            self.vertices[v_to.index].position = update_position

    def collapse_halfedge_boundary(self, target_halfedge):
        reverse_halfedge = target_halfedge.opposite
        if target_halfedge.face is None:
            target_halfedge, reverse_halfedge = reverse_halfedge, target_halfedge



    def flip_halfedge(self, he):
        rev = he.opposite

        # Get surronding vertices, halfedges and faces
        v0 = he.vertex_to
        v1 = he.next.vertex_to
        v2 = rev.next.vertex_to
        v3 = rev.vertex_to

        he0 = he.next
        he1 = he.next.next
        he2 = rev.next.next
        he3 = rev.next

        f0 = he.face
        f1 = rev.face

        # Update halfedges of to/from vertices
        v0.halfedge = he0
        v3.halfedge = he3

        # Update halfedge's source and destination
        he.vertex_from = v1
        he.vertex_to = v2
        rev.vertex_from = v2
        rev.vertex_to = v1

        # Update face circulation
        he.next = he2
        he2.next = he0
        he0.next = he

        rev.next = he1
        he1.next = he3
        he3.next = rev

        # Update faces
        f0.halfedge = he
        he.face = f0
        he2.face = f0
        he0.face = f0

        f1.halfedge = rev
        rev.face = f1
        he1.face = f1
        he3.face = f1

    def clean(self):
        # Compute new vertex indices
        count = 0
        new_index_table = [ 0 ] * self.n_vertices()
        for i, v in enumerate(self.vertices):
            new_index_table[i] = count
            if v is not None:
                count += 1

        # Update vertex array
        self.vertices = [ v for v in self.vertices if v is not None ]
        for i, v in enumerate(self.vertices):
            v.index = i

        # Update halfedge array
        self.halfedges = [ he for he in self.halfedges if he is not None ]
        for i, he in enumerate(self.halfedges):
            he.index = i

        self.faces = [ f for f in self.faces if f is not None ]
        for i, f in enumerate(self.faces):
            f.index = i

        self.indices = [ -1 ] * (len(self.faces) * 3)
        for i, f in enumerate(self.faces):
            vs = list(f.vertices())
            assert len(vs) == 3

            self.indices[i * 3 + 0] = vs[0].index
            self.indices[i * 3 + 1] = vs[1].index
            self.indices[i * 3 + 2] = vs[2].index

            assert vs[0].index < len(self.vertices)
            assert vs[1].index < len(self.vertices)
            assert vs[2].index < len(self.vertices)

    def _make_halfedge(self):
        table = [ [] for i in range(len(self.vertices)) ]

        self.halfedges.clear()
        self.faces.clear()

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

            he0.face = face
            he1.face = face
            he2.face = face

            self.halfedges.extend([ he0, he1, he2 ])
            self.faces.append(face)

            table[self.vertices[self.indices[i + 0]].index].append(he0)
            table[self.vertices[self.indices[i + 1]].index].append(he1)
            table[self.vertices[self.indices[i + 2]].index].append(he2)

        # Set opposite halfedges
        for he0 in self.halfedges:
            for he1 in table[he0.vertex_to.index]:
                if he0.vertex_from == he1.vertex_to and \
                   he1.vertex_from == he0.vertex_to:

                   he0.opposite = he1
                   he1.opposite = he0
                   break

            # Opposite halfedge not found
            # Mark vertices as border vertices
            if he0.opposite is None:
                he0.vertex_from.is_boundary = True
                he0.vertex_to.is_boundary = True

                he1 = Halfedge()
                he1.vertex_from = he0.vertex_to
                he1.vertex_to = he0.vertex_from
                he1.opposite = he0
                he0.opposite = he1

                he1.vertex_from.halfedge = he1

                self.halfedges.append(he1)

        # Process border vertices
        for v in self.vertices:
            if v.is_boundary:
                he = v.halfedge
                while True:
                    if he.opposite.next is None:
                        he.opposite.next = v.halfedge
                        break

                    he = he.opposite.next

        for i, he in enumerate(self.halfedges):
            he.index = i

        for i, f in enumerate(self.faces):
            f.index = i

    def verify(self):
        for v in self.vertices:
            if v is None:
                continue

            if v.index < 0:
                return False

            if v.halfedge is None:
                return False

        for he in self.halfedges:
            if he is None:
                continue

            if he.index < 0:
                return False

            if he.vertex_from is None or he.vertex_to is None:
                return False

            if he.next is None:
                return False

            if he.opposite is None:
                return False

            if he.face is None:
                return False

        for f in self.faces:
            if f is None:
                continue

            if f.index < 0:
                return False

            if f.halfedge is None:
                return False

        return True

    def clear(self):
        self.vertices.clear()
        self.halfedges.clear()
        self.faces.clear()
        self.indices.clear()
