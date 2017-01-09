import time
from itertools import chain
from heapq import *

import numpy as np

from ..exception import PygpException
from .trimesh import TriMesh
from .vector import Vector
from .halfedge import Halfedge

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
    tick = int(width * ratio)
    bar = [ '=' ] * tick + [ ' ' ] * (width - tick)
    if tick != width:
        bar[tick] = '>'

    bar = ''.join(bar)
    print('[ {0:6.2f} % ] [ {1} ]'.format(100.0 * ratio, bar), end='\r', flush=True)

def compute_QEM(Q1, Q2, v1, v2):
    Q = np.identity(4)
    Q[:3, :4] = (Q1 + Q2)[:3, :4]
    if np.linalg.det(Q) < 1.0e-6:
        v_bar = 0.5 * (v1 + v2)
        v_bar = np.array([ v_bar.x, v_bar.y, v_bar.z, 1.0 ])
    else:
        v_bar = np.linalg.solve(Q, np.array([0.0, 0.0, 0.0, 1.0]))

    qem = float(np.dot(v_bar, np.dot(Q1 + Q2, v_bar)))

    return qem, Vector(v_bar[0], v_bar[1], v_bar[2])

def simplify(mesh, ratio=0.5, remains=-1, show_progress=True):
    start_time = time.clock()
    nv = mesh.n_vertices()

    # How many vertices are removed?
    n_remove = int(nv * (1.0 - ratio))
    if remains > 0:
        if remains <= 3:
            raise PygpException('remainig vertices must be more than 3!')
        n_remove = nv - remains

    # Extract neighboring vertices
    neighbor_verts = [ [] for i in range(nv) ]
    for v_i in mesh.vertices:
        neighbor_verts[v_i.index] = set([ v.index for v in v_i.vertices() ])

    neighbor_faces = [ [] for i in range(nv) ]
    for i, f in enumerate(mesh.faces):
        for v in f.vertices():
            neighbor_faces[v.index].append(i)

    for i in range(nv):
        neighbor_faces[i] = set(neighbor_faces[i])

    # Compute matrix Q
    Qs = [ np.zeros((4, 4)) for i in range(nv) ]
    for t in mesh.faces:
        vs = list(t.vertices())
        assert len(vs) == 3

        ps = [ v.position for v in vs ]
        norm = (ps[1] - ps[0]).cross(ps[2] - ps[0])
        w = norm.norm()
        norm = norm / w

        d = -norm.dot(ps[0])
        pp = np.array([ norm.x, norm.y, norm.z, d ])
        Q = pp.reshape((pp.size, 1)) * pp

        Qs[vs[0].index] += w * Q
        Qs[vs[1].index] += w * Q
        Qs[vs[2].index] += w * Q

    # Push QEMs
    pque = PriorityQueue()
    for he in mesh.halfedges:
        i1 = he.vertex_from.index
        i2 = he.vertex_to.index
        v1 = he.vertex_from.position
        v2 = he.vertex_to.position
        Q1 = Qs[i1]
        Q2 = Qs[i2]
        qem, v_bar = compute_QEM(Q1, Q2, v1, v2)
        pque.push(QEMNode(qem, i1, i2, v_bar))

    removed = 0
    uftree = UnionFindTree(nv)
    while removed < n_remove:
        # Find edge with minimum QEM
        try:
            qn = pque.pop()
        except IndexError:
            break

        qem, v_i, v_j, v_bar = qn.value, qn.ii, qn.jj, qn.vec
        assert v_i != v_j
        if mesh.vertices[v_i] is None or mesh.vertices[v_j] is None:
            # None vertex is already removed
            continue

        # Vertex with degree less than 4 should not be contracted
        if len(neighbor_verts[v_i]) <= 3 or \
           len(neighbor_verts[v_j]) <= 3:
            continue

        # Get the list of vertices around v_i and v_j
        new_neighbor_verts = neighbor_verts[v_i].union(neighbor_verts[v_j])
        try:
            new_neighbor_verts.remove(v_i)
            new_neighbor_verts.remove(v_j)
        except KeyError as e:
            print(e.message)

        new_neighbor_faces = neighbor_faces[v_i].symmetric_difference(neighbor_faces[v_j])
        remove_faces = neighbor_faces[v_i].intersection(neighbor_faces[v_j])

        # Check face flip
        is_flip = False
        for i in new_neighbor_faces:
            vs = list(mesh.faces[i].vertices())
            vs = [ mesh.vertices[uftree.root(v.index)] for v in vs ]
            assert len(vs) == 3

            ps = [ v.position for v in vs ]
            norm_before = (ps[1] - ps[0]).cross(ps[2] - ps[0])

            is_find = False
            for i, v in enumerate(vs):
                if v.index == v_i or v.index == v_j:
                    ps[i] = v_bar
                    is_find = True
                    break

            assert is_find

            norm_after = (ps[1] - ps[0]).cross(ps[2] - ps[0])

            if norm_before.dot(norm_after) <= 1.0e-6:
                is_flip = True
                break

        if is_flip:
            continue

        # Check face degeneration
        is_degenerate = False
        for i in new_neighbor_verts:
            if v_j in neighbor_verts[i]:
                if len(neighbor_verts[i]) < 4:
                    is_degenerate = True
                    break
            else:
                if len(neighbor_verts[i]) < 3:
                    is_degenerate = True
                    break

        if is_degenerate:
            continue

        for i in neighbor_verts[v_j]:
            try:
                neighbor_verts[i].remove(v_j)
            except KeyError as e:
                print(e.message)

            neighbor_verts[i].add(v_i)

        for i in new_neighbor_verts:
            neighbor_faces[i] = neighbor_faces[i].difference(remove_faces)

        for i in remove_faces:
            mesh.indices[i * 3 + 0] = -1
            mesh.indices[i * 3 + 1] = -1
            mesh.indices[i * 3 + 2] = -1

        neighbor_verts[v_i] = new_neighbor_verts
        neighbor_verts[v_j] = None
        neighbor_faces[v_i] = new_neighbor_faces
        neighbor_faces[v_j] = None

        # Manage merged vertex indices
        mesh.vertices[v_i].position = v_bar
        mesh.vertices[v_j] = None
        uftree.merge(v_i, v_j)
        assert v_i == uftree.root(v_j)

        # Update matrix Q
        update_vertices = neighbor_verts[v_i].union([ v_i ])
        for i in update_vertices:
            Qs[i] = np.zeros((4, 4))
            for j in neighbor_faces[i]:
                vs = list(mesh.faces[j].vertices())
                vs = [ mesh.vertices[uftree.root(v.index)] for v in vs ]
                assert len(vs) == 3

                ps = [ v.position for v in vs ]
                norm = (ps[1] - ps[0]).cross(ps[2] - ps[0])
                w = norm.norm()
                norm = norm / w

                d = -norm.dot(ps[0])
                pp = np.array([ norm.x, norm.y, norm.z, d ])
                Q = pp.reshape((pp.size, 1)) * pp
                Qs[i] += w * Q

        # Update QEMs
        for i in update_vertices:
            for j in neighbor_verts[i]:
                assert i != j
                assert mesh.vertices[i] is not None
                assert mesh.vertices[j] is not None

                if len(neighbor_verts[i]) <= 3: continue
                if len(neighbor_verts[j]) <= 3: continue

                v1 = mesh.vertices[i].position
                v2 = mesh.vertices[j].position
                Q1 = Qs[i]
                Q2 = Qs[j]
                qem, v_bar = compute_QEM(Q1, Q2, v1, v2)
                pque.push(QEMNode(qem, i, j, v_bar))

        # Progress
        removed += 1
        if show_progress:
            if removed == n_remove or removed % (n_remove // 1000) == 0:
                progress_bar(removed, n_remove)

    print('')
    print('{} vertices removed!'.format(n_remove))
    print('{:.2f} sec elapsed!'.format(time.clock() - start_time))

    # Compact vertices and update indices for faces
    count = 0
    new_index_table = [ 0 ] * mesh.n_vertices()
    for i, v in enumerate(mesh.vertices):
        new_index_table[i] = count
        if v is not None:
            count += 1

    mesh.vertices = [ v for v in mesh.vertices if v is not None ]
    for i, v in enumerate(mesh.vertices):
        v.index = i

    new_indices = [ -1 ] * len(mesh.indices)
    for i in range(0, len(mesh.indices), 3):
        i0 = mesh.indices[i + 0]
        i1 = mesh.indices[i + 1]
        i2 = mesh.indices[i + 2]

        i0 = uftree.root(i0)
        i1 = uftree.root(i1)
        i2 = uftree.root(i2)

        i0 = new_index_table[i0]
        i1 = new_index_table[i1]
        i2 = new_index_table[i2]
        if i0 == i1 or i1 == i2 or i2 == i0:
            continue

        new_indices[i + 0] = i0
        new_indices[i + 1] = i1
        new_indices[i + 2] = i2

    mesh.indices = list(filter(lambda i : i >= 0, new_indices))
    mesh._make_halfedge()
