import math

class Point(object):
    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __mul__(self, s):
        return Point(self.x * s, self.y * s)

    def __rmul__(self, s):
        return Point(self.x * s, self.y * s)

    def __truediv__(self, s):
        if s == 0.0:
            raise ZeroDivisionError()
        return Point(self.x / s, self.y / s)

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def det(self, p):
        return self.x * p.y - self.y * p.x

    def norm(self):
        return math.sqrt(self.dot(self))

    def norm2(self):
        return self.dot(self)

    def perp(self):
        return Point(-self.y, self.x)

    def proj(self, p):
        k = self.det(p) / self.norm2()
        return self * k

    def __lt__(self, p):
        if self.x != p.x:
            return self.x < p.x
        return self.y < p.y

    def __gt__(self, p):
        if self.x != p.x:
            return self.x > p.x
        return self.y > p.y

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    def __repr__(self):
        return '(%f, %f)' % (self.x, self.y)


def tri(a, b, c):
    return (b - a).det(c - a)
