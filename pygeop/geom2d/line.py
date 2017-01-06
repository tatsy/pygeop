from .utils import *
from .point import *

class Line(object):
    def __init__(self, s, t):
        self.s = s
        self.t = t

    def length(self):
        return (self.s - self.t).norm()

    def vec(self):
        return self.t - self.s

    def proj(self, p):
        return self.s + self.vec().proj(p - self.s)

    def isectLL(self, l):
        # Intersecting.
        if sgn(self.vec().det(l.vec())) != 0: return 1
        # Parallel but different lines.
        if sgn(self.vec().det(l.s - self.s)) != 0: return 0
        # On the same line.
        return -1

    def isectLS(self, l):
        return sgn(tri(self.s, self.t, l.s)) * sgn(tri(self.s, self.t, l.t)) <= 0

    def isectSS(self, l):
        return self.isectLS(l) and l.isectLS(self)

    def pointLL(self, l):
        return self.s + self.vec() * (l.s - self.s).det(l.vec()) / self.vec().det(l.vec())

    def distLP(self, p):
        return abs(tri(self.s, self.t, p)) / self.vec().norm()

    def distSP(self, p):
        if sgn(self.vec().dot(p - self.s)) <= 0:
            return (p - self.s).norm()
        if sgn(self.vec().dot(p - self.t)) <= 0:
            return (p - self.t).norm()
        return self.distLP(p)

    def __eq__(self, l):
        return self.s == l.s and self.t == l.t

    def __repr__(self):
        return '[%s -> %s]' % (self.s.__str__(), self.t.__str__())
