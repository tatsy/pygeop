from .vector import Vector

class Vertex(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.index= -1
        self.halfedge = None

    def copy(self):
        v = Vertex(self.x, self.y, self.z)
        v.index = self.index
        v.halfedge = self.halfedge
        return v

    def _position(self):
        return Vector(self.x, self.y, self.z)

    def _set_position(self, p):
        self.x = p.x
        self.y = p.y
        self.z = p.z

    position = property(_position, _set_position)

    @staticmethod
    def distance(v1, v2):
        p1 = v1.position
        p2 = v2.position
        return (p1 - p2).norm()

    def degree(self):
        return len(list(self.halfedges()))

    def halfedges(self):
        he = self.halfedge
        while True:
            yield he

            he = he.opposite.next
            if he == self.halfedge:
                break

    def vertices(self):
        he = self.halfedge
        while True:
            yield he.vertex_to

            he = he.opposite.next
            if he == self.halfedge:
                break

    def __eq__(self, v):
        return self.x == v.x and self.y == v.y and self.z == v.z

    def __repr__(self):
        return '({0:.3f}, {1:.3f}, {2:.3f})'.format(self.x, self.y, self.z)
