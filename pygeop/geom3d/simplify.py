from itertools import chain
from heapq import *

import numpy as np

from ..exception import PygpException
from .trimesh import TriMesh
from .vector import Vector

class PriorityQueue(object):
    def __init__(self):
        self.h = []

    def push(self, v):
        heappush(self.h, v)

    def top(self):
        if len(self.h) == 0:
            raise Exception('Queue is empty!')
        return self.h[0]

    def pop(self):
        return heappop(self.h)

    def clear(self):
        self.h = []


class UnionFindTree(object):
    def __init__(self, n):
        self.n = n
        self.val = [ -1 ] * n

    def root(self, x):
        if self.val[x] < 0:
            return x

        self.val[x] = self.root(self.val[x])
        return self.val[x]

    def merge(self, x, y):
        x = self.root(x)
        y = self.root(y)
        if x == y:
            return

        self.val[x] += self.val[y]
        self.val[y] = x

    def same(self, x, y):
        return self.root(x) == self.root(y)

def simplify(mesh, ratio=0.5, remains=-1, show_progress=True):
    N_v = mesh.n_vertices()

    n_remove = int(N_v * (1.0 - ratio))
    if remains > 0:
        if remains <= 3:
            raise PygpException('remainig vertices must be more than 3!')
        n_remove = N_v - remains

    Qs = [ np.zeros((4, 4)) for i in range(N_v) ]
    for t in mesh.faces:
        vs = list(t.vertices())

        ps = [ Vector(v.x, v.y, v.z) for v in vs ]
        norm = (ps[2] - ps[0]).cross(ps[1] - ps[0])
        norm = Vector.normalize(norm)

        d = -norm.dot(ps[0])
        pp = np.array([ norm.x, norm.y, norm.z, d ])
        Q = pp.reshape((pp.size, -1)) * pp

        Qs[vs[0].index] += Q
        Qs[vs[1].index] += Q
        Qs[vs[2].index] += Q

    pque = PriorityQueue()
    for he in mesh.halfedges:
        i1 = he.vertex_from.index
        i2 = he.vertex_to.index
        v1 = he.vertex_from.position()
        v2 = he.vertex_to.position()
        v_bar = 0.5 * (v1 + v2)
        Q1 = Qs[i1]
        Q2 = Qs[i2]
        v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
        qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))
        pque.push((qem, i1, i2))


    trial = 0
    uftree = UnionFindTree(N_v)
    while trial < n_remove:
        # Find edge with minimum QEM
        minindex = (-1, -1)
        minvalue = 1.0e20
        v, ii, jj = pque.pop()
        if mesh.vertices[ii] is None or mesh.vertices[jj] is None:
            continue

        v_i = mesh.vertices[ii]
        v_j = mesh.vertices[jj]
        assert v_i is not None and v_j is not None
        assert v_i.index != v_j.index

        # Rotate v_i's halfedge
        if v_i.halfedge.next.vertex_to == v_j:
            v_i.halfedge = v_i.halfedge.opposite.next
            if v_i.halfedge.vertex_to == v_j:
                v_i.halfedge = v_i.halfedge.opposite.next

        if v_i.halfedge.vertex_to == v_j:
            v_i.halfedge = v_i.halfedge.opposite.next
            if v_i.halfedge.next.vertex_to == v_j:
                v_i.halfedge = v_i.halfedge.opposite.next

        #assert v_i.halfedge.vertex_to != v_j and v_i.halfedge.next.vertex_to != v_j

        for he_j in list(v_j.halfedges()):
            if he_j.vertex_to == v_i:
                continue

            he_j.vertex_from = v_i
            he_j.opposite.vertex_to = v_i

            for he_i in v_i.halfedges():
                if he_i.vertex_to != he_j.vertex_to:
                    continue

                if he_i.next.vertex_to != v_j and he_j.next.vertex_to == v_i or \
                   he_i.next.vertex_to == v_j and he_j.next.vertex_to != v_i:

                       he_j.opposite = he_i.opposite
                       he_i.opposite = he_j.opposite

        v_new = 0.5 * (v_i.position() + v_j.position())
        v_i.x = v_new.x
        v_i.y = v_new.y
        v_i.z = v_new.z

        mesh.vertices[v_j.index] = None
        uftree.merge(v_i.index, v_j.index)
        assert v_i.index == uftree.root(v_j.index)

        Qs[v_i.index] = np.zeros((4, 4))
        for he in v_i.halfedges():
            vs = []
            he_it = he
            while True:
                vs.append(he_it.vertex_to)
                he_it = he_it.next
                if he_it == he:
                    break

            assert len(vs) == 3

            ps = [ Vector(v.x, v.y, v.z) for v in vs ]
            norm = (ps[2] - ps[0]).cross(ps[1] - ps[0])
            if norm.norm() == 0.0:
                continue

            norm = Vector.normalize(norm)

            d = -norm.dot(ps[0])
            pp = np.array([ norm.x, norm.y, norm.z, d ])
            Q = pp.reshape((pp.size, -1)) * pp
            Qs[v_i.index] += Q

        for u in v_i.vertices():
            i1 = v_i.index
            i2 = u.index
            assert i1 != i2

            v1 = v_i.position()
            v2 = u.position()
            v_bar = 0.5 * (v1 + v2)
            Q1 = Qs[i1]
            Q2 = Qs[i2]
            v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
            qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))
            pque.push((qem, i1, i2))

        trial += 1
        if show_progress:
            print('.', end='', flush=True)
            if trial % 100 == 0:
                print(' {}'.format(trial))

    print('{} vertices removed!'.format(n_remove))

    # Save
    count = 0
    new_index = [ 0 ] * N_v
    for i, v in enumerate(mesh.vertices):
        new_index[i] = count
        if v is not None:
            count += 1

    mesh.vertices = [ v for v in mesh.vertices if v is not None ]
    for i in range(0, len(mesh.indices), 3):
        i0 = mesh.indices[i + 0]
        i1 = mesh.indices[i + 1]
        i2 = mesh.indices[i + 2]

        i0 = uftree.root(i0)
        i1 = uftree.root(i1)
        i2 = uftree.root(i2)

        i0 = new_index[i0]
        i1 = new_index[i1]
        i2 = new_index[i2]

        mesh.indices[i + 0] = i0
        mesh.indices[i + 1] = i1
        mesh.indices[i + 2] = i2
