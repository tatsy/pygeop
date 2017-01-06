from .vector import Vector

class Vertex(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index= -1
        self.halfedge = None

    def copy(self):
        return Vertex(self.x, self.y, self.z)

    def position(self):
        return Vector(self.x, self.y, self.z)

    def setPosition(self, p):
        self.x = p.x
        self.y = p.y
        self.z = p.z

    @staticmethod
    def distance(v1, v2):
        p1 = v1.position()
        p2 = v2.position()
        return (p1 - p2).norm()

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
