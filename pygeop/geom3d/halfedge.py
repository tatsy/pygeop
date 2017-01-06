class Halfedge(object):
    def __init__(self):
        self.vertex_from = None
        self.vertex_to = None
        self.face = None
        self.next = None
        self.opposite = None

    def __repr__(self):
        return '{0} -> {1}'.format(self.vertex_from.index, self.vertex_to.index)
