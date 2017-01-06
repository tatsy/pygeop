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
    nv = mesh.n_vertices()

    # How many vertices are removed?
    n_remove = 5000 #int(nv * (1.0 - ratio))
    if remains > 0:
        if remains <= 3:
            raise PygpException('remainig vertices must be more than 3!')
        n_remove = nv - remains

    # Compute matrix Q
    Qs = [ np.zeros((4, 4)) for i in range(nv) ]
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

    # Push QEMs
    pque = PriorityQueue()
    for he in mesh.halfedges:
        i1 = he.vertex_from.index
        i2 = he.vertex_to.index
        v1 = he.vertex_from.position
        v2 = he.vertex_to.position
        v_bar = 0.5 * (v1 + v2)
        Q1 = Qs[i1]
        Q2 = Qs[i2]
        v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
        qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))
        pque.push((qem, i1, i2))

    removed = 0
    uftree = UnionFindTree(nv)
    while removed < n_remove:
        # Find edge with minimum QEM
        v, ii, jj = pque.pop()
        assert ii != jj

        # Take vertex pair
        v_i = mesh.vertices[ii]
        v_j = mesh.vertices[jj]
        if v_i is None or v_j is None:
            # None vertex is already removed
            continue

        # Rotate v_i's halfedge so that it does not in the removed face.
        if v_i.halfedge.next.vertex_to == v_j:
            v_i.halfedge = v_i.halfedge.opposite.next
            if v_i.halfedge.vertex_to == v_j:
                v_i.halfedge = v_i.halfedge.opposite.next

        if v_i.halfedge.vertex_to == v_j:
            v_i.halfedge = v_i.halfedge.opposite.next
            if v_i.halfedge.next.vertex_to == v_j:
                v_i.halfedge = v_i.halfedge.opposite.next

        assert v_i.halfedge.vertex_to.index != v_j.index and \
               v_i.halfedge.next.vertex_to.index != v_j.index

        # Update oppsite halfedges
        he_i_j = None
        for he_i in v_i.halfedges():
            if he_i.vertex_to == v_j:
                he_i_j = he_i
                break

        assert he_i_j is not None
        he_j_i = he_i_j.opposite

        if he_i_j.next.vertex_to.halfedge == he_i_j.next.next:
            he_i_j.next.vertex_to.halfedge = he_i_j.next.next.opposite.next

        if he_j_i.next.vertex_to.halfedge == he_j_i.next.next:
            he_j_i.next.vertex_to.halfedge = he_j_i.next.next.opposite.next

        he_i_j.next.opposite.opposite, he_i_j.next.next.opposite.opposite = \
            he_i_j.next.next.opposite, he_i_j.next.opposite

        he_j_i.next.opposite.opposite, he_j_i.next.next.opposite.opposite = \
            he_j_i.next.next.opposite, he_j_i.next.opposite

        # Update position
        v_new = 0.5 * (v_i.position + v_j.position)
        v_i.position = v_new

        # update halfedge destinations
        for he in v_i.halfedges():
            he.vertex_from = v_i
            he.opposite.vertex_to = v_i

        assert v_i not in v_i.vertices()

        # Manage merged vertex indices
        mesh.vertices[v_j.index] = None
        uftree.merge(v_i.index, v_j.index)
        assert v_i.index == uftree.root(v_j.index)

        # Update matrix Q
        update_vertices = [ v_i ] #[ v for v in v_i.vertices() ] + [ v_i ]
        for v in update_vertices:
            Qs[v.index] = np.zeros((4, 4))
            for he in v.halfedges():
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

                d = -norm.normalize().dot(ps[0])
                pp = np.array([ norm.x, norm.y, norm.z, d ])
                Q = pp.reshape((pp.size, -1)) * pp
                Qs[v_i.index] += Q

        # Update QEMs
        for v in update_vertices:
            for u in v.vertices():
                i1 = v.index
                i2 = u.index
                assert i1 != i2

                v1 = v.position
                v2 = u.position
                v_bar = 0.5 * (v1 + v2)
                Q1 = Qs[i1]
                Q2 = Qs[i2]
                v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
                qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))
                pque.push((qem, i1, i2))

        # Progress
        removed += 1
        if show_progress:
            print('.', end='', flush=True)
            if removed % 100 == 0:
                print(' {}'.format(removed))

    print('{} vertices removed!'.format(n_remove))

    # Compact vertices and update indices for faces
    count = 0
    new_index = [ 0 ] * nv
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
