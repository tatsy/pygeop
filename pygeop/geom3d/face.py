class Face(object):
    def __init__(self):
        self.halfedge = None
        self.index = -1

    def halfedges(self):
        he = self.halfedge
        while True:
            yield he

            he = he.next
            if he == self.halfedge:
                break

    def vertices(self):
        return ( he.vertex_to for he in self.halfedges() )
