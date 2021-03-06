import math
import time
from itertools import chain
from heapq import *

import numpy as np

from ..exception import PygpException
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
        if len(self.h) == 0:
            raise Exception('Queue is empty!')
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

class QEMNode(object):
    def __init__(self, value, ii, jj, vec):
        self.value = value
        self.ii = ii
        self.jj = jj
        self.vec = vec

    def __lt__(self, n):
        return (self.value, self.ii, self.jj).__lt__((n.value, n.ii, n.jj))

def progress_bar(x, total, width=50):
    ratio = (x + 0.5) / total
    tick = min(int(width * ratio), width - 1)
    bar = [ '=' ] * tick + [ ' ' ] * (width - tick)
    if tick != width:
        bar[tick] = '>'

    bar = ''.join(bar)
    percent = min(100.0, 100.0 * ratio)
    print('[ {0:6.2f} % ] [ {1} ]'.format(percent, bar),
          end='\n' if x >= total else '\r', flush=True)

def compute_QEM(Q1, Q2, v1, v2):
    if v1.is_boundary and v2.is_boundary:
        v_bar = 0.5 * (v1.position + v2.position)
        v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])

    elif v1.is_boundary:
        v = v1.position
        v_bar = np.array([ v.x, v.y, v.z, 1.0 ])

    elif v2.is_boundary:
        v = v2.position
        v_bar = np.array([ v.x, v.y, v.z, 1.0 ])

    else:
        Q = np.identity(4)
        Q[:3, :4] = (Q1 + Q2)[:3, :4]
        if np.linalg.det(Q) < 1.0e-6:
            v_bar = 0.5 * (v1.position + v2.position)
            v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
        else:
            v_bar = np.linalg.solve(Q, np.array([0.0, 0.0, 0.0, 1.0]))

    qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))

    return qem, Vector(v_bar[0], v_bar[1], v_bar[2])

def simplify(mesh, ratio=0.5, remains=-1, show_progress=True):
    EPS = 1.0e-12
    start_time = time.clock()
    nv = mesh.n_vertices()

    # How many vertices are removed?
    n_remove = int(nv * (1.0 - ratio))
    if remains > 0:
        if remains <= 3:
            raise PygpException('remainig vertices must be more than 3!')
        n_remove = nv - remains

    # Compute matrix Q
    if show_progress:
        print('Computing matrix Q')

    Qs = [ np.zeros((4, 4)) for i in range(nv) ]
    for i, t in enumerate(mesh.faces):
        vs = list(t.vertices())
        assert len(vs) == 3

        ps = [ v.position for v in vs ]
        norm = (ps[1] - ps[0]).cross(ps[2] - ps[0])
        w = norm.norm()
        norm /= (w + EPS)

        d = -norm.dot(ps[0])
        pp = np.array([ norm.x, norm.y, norm.z, d ])
        Q = pp.reshape((pp.size, 1)) * pp

        Qs[vs[0].index] += w * Q
        Qs[vs[1].index] += w * Q
        Qs[vs[2].index] += w * Q

        if show_progress:
            progress_bar(i, len(mesh.faces))

    if show_progress:
        progress_bar(len(mesh.faces), len(mesh.faces))

    # Push QEMs
    if show_progress:
        print('Computing QEMs')

    pque = PriorityQueue()
    for i, he in enumerate(mesh.halfedges):
        v1 = he.vertex_from
        v2 = he.vertex_to
        i1 = v1.index
        i2 = v2.index
        Q1 = Qs[i1]
        Q2 = Qs[i2]
        qem, v_bar = compute_QEM(Q1, Q2, v1, v2)
        pque.push(QEMNode(qem, i1, i2, v_bar))

        if show_progress:
            progress_bar(i, len(mesh.halfedges))

    if show_progress:
        progress_bar(len(mesh.halfedges), len(mesh.halfedges))

    removed = 0
    uftree = UnionFindTree(nv)
    while removed < n_remove:
        # Find edge with minimum QEM
        try:
            qn = pque.pop()
        except Exception as e:
            break

        ii, jj, v_bar = qn.ii, qn.jj, qn.vec
        assert ii != jj

        v_i = mesh.vertices[ii]
        v_j = mesh.vertices[jj]
        if v_i is None or v_j is None:
            # None vertex is already removed
            continue

        if not v_i.is_boundary and v_j.is_boundary:
            ii, jj = jj, ii
            v_i = mesh.vertices[ii]
            v_j = mesh.vertices[jj]

        # Vertex with degree less than 4 should not be contracted
        if v_i.degree() <= 3 or v_j.degree() <= 3:
            continue

        # Check face flip
        is_flip = False
        for f in chain(v_i.faces(), v_j.faces()):
            vs = list(f.vertices())
            assert len(vs) == 3

            if any([ v is v_i for v in vs ]) and any([ v is v_j for v in vs ]):
                # This is collpased face
                continue

            ps = [ v.position for v in vs ]
            norm_before = (ps[1] - ps[0]).cross(ps[2] - ps[0])\

            is_found = False
            for i, v in enumerate(vs):
                if v is v_i or v is v_j:
                    ps[i] = v_bar
                    is_found = True
                    break

            assert is_found

            norm_after = (ps[1] - ps[0]).cross(ps[2] - ps[0])

            cos = norm_before.dot(norm_after) / (norm_before.norm() * norm_after.norm() + EPS)
            if cos <= 1.0e-20:
                is_flip = True
                break

        if is_flip:
            continue

        # Check face degeneration
        neighbor_v_i = set([ v.index for v in v_i.vertices() ])
        neighbor_v_j = set([ v.index for v in v_j.vertices() ])
        neighbor_both = neighbor_v_i.intersection(neighbor_v_j)

        is_degenerate = False
        for i in neighbor_both:
            if mesh.vertices[i].degree() < 4:
                is_degenerate = True
                break

        if is_degenerate:
            continue

        # Collapse halfedge
        try:
            mesh.collapse_halfedge(v_j, v_i, v_bar)
            uftree.merge(v_i.index, v_j.index)
            assert v_i.index == uftree.root(v_j.index)
        except Exception as e:
            print(e.message)
            continue
            # raise e

        assert mesh.vertices[v_i.index] is not None
        assert mesh.vertices[v_j.index] is None

        # Check triangle shapes
        is_update = True
        update_vertices = [ v_i ] #list(chain([ v_i ], v_i.vertices()))
        while is_update:
            is_update = False
            for he in v_i.halfedges():
                if he.face is None or he.opposite.face is None:
                    # Boundary halfedge
                    continue

                v0 = he.next.vertex_to.position
                v1 = he.vertex_to.position
                v2 = he.vertex_from.position
                v3 = he.opposite.next.vertex_to.position

                e0 = v1 - v0
                e1 = v2 - v0
                c1 = e0.dot(e1) / (e0.norm() * e1.norm() + EPS)
                a1 = math.acos(max(-1.0, min(c1, 1.0)))

                e2 = v1 - v3
                e3 = v2 - v3
                c2 = e2.dot(e3) / (e2.norm() * e3.norm() + EPS)
                a2 = math.acos(max(-1.0, min(c2, 1.0)))

                if a1 + a2 > math.pi:
                    mesh.flip_halfedge(he)
                    is_update = True
                    break

        # Progress
        removed += 1
        if show_progress:
            if removed == n_remove or removed % max(1, n_remove // 1000) == 0:
                progress_bar(removed, n_remove)

        if removed == n_remove:
            break

        # Update matrix Q
        for v in update_vertices:
            Qs[v.index] = np.zeros((4, 4))
            for f in v.faces():
                vs = list(f.vertices())
                assert len(vs) == 3

                ps = [ v.position for v in vs ]
                norm = (ps[1] - ps[0]).cross(ps[2] - ps[0])
                w = norm.norm()
                norm /= (w + EPS)

                d = -norm.dot(ps[0])
                pp = np.array([ norm.x, norm.y, norm.z, d ])
                Q = pp.reshape((pp.size, 1)) * pp
                Qs[v.index] += w * Q

        # Update QEMs
        for v1 in update_vertices:
            for v2 in v1.vertices():
                assert v1.index != v2.index
                assert mesh.vertices[v1.index] is not None
                assert mesh.vertices[v2.index] is not None

                if v1.degree() <= 3 or v2.degree() <= 3: continue
                if v1.degree() >= 7 or v2.degree() >= 7: continue

                Q1 = Qs[v1.index]
                Q2 = Qs[v2.index]
                qem, v_bar = compute_QEM(Q1, Q2, v1, v2)

                pque.push(QEMNode(qem, v1.index, v2.index, v_bar))

    print('')
    if removed < n_remove:
        print('Target number is not reached!')

    print('{} vertices removed!'.format(removed))
    print('{:.2f} sec elapsed!'.format(time.clock() - start_time))

    # Compact vertices and update indices for faces
    mesh.clean()
