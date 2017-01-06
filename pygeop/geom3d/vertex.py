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

    def to_v(self):
        return Vector(self.x, self.y, self.z)

    @staticmethod
    def distance(v1, v2):
        dx = v1.x - v2.x
        dy = v1.y - v2.y
        dz = v1.z - v2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

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
