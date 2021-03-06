import math
from ..exception import PygpException

class Vector(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        x = self.y * v.z - self.z * v.y
        y = self.z * v.x - self.x * v.z
        z = self.x * v.y - self.y * v.x
        return Vector(x, y, z)

    def normalize(self):
        try:
            self = self / self.norm()
        except ZeroDivisionError:
            raise PygpException('Vector length is zero!')

        return self

    def norm(self):
        return math.sqrt(self.norm2())

    def norm2(self):
        return self.dot(self)

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, v):
        if isinstance(v, Vector):
            return Vector(self.x * v.x, self.y * v.y, self.z * v.z)
        else:
            return Vector(self.x * v, self.y * v, self.z * v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __truediv__(self, v):
        if isinstance(v, Vector):
            if v.x == 0.0 or v.y == 0.0 or v.z == 0.0:
                raise ZeroDivisionError()
            return Vector(self.x / v.x, self.y / v.y, self.z / v.z)
        else:
            if v == 0.0:
                raise ZeroDivisionError()
            return Vector(self.x / v, self.y / v, self.z / v)

    def __repr__(self):
        return '( %.4f, %.4f, %.4f )' % (self.x, self.y, self.z)
