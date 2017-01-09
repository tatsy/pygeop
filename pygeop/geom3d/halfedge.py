class Halfedge(object):
    def __init__(self):
        self.vertex_from = None
        self.vertex_to = None
        self.face = None
        self.next = None
        self.opposite = None
        self.index = -1

    def __eq__(self, he):
        if self.vertex_from is None or self.vertex_to is None: return False
        if he.vertex_from is None or he.vertex_to is None: return False
        if self.vertex_from.index < 0 or self.vertex_to.index < 0: return False
        if he.vertex_from.index < 0 or he.vertex_to.index < 0: return False
        return self.vertex_from.index == he.vertex_from.index and \
               self.vertex_to.index == he.vertex_to.index

    def __repr__(self):
        return '{0} -> {1}'.format(self.vertex_from.index, self.vertex_to.index)
