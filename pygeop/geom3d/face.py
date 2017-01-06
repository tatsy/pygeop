class Face(object):
    def __init__(self):
        self.halfedge = None

    def vertices(self):
        he = self.halfedge
        while True:
            yield he.vertex_from

            he = he.next
            if he == self.halfedge:
                break
