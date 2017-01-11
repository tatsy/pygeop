import os
import sys
import argparse

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

from pygeop.geom3d import TriMesh, simplify


def main():
    parser = argparse.ArgumentParser(description='Demo for mesh simplification.')
    parser.add_argument('--input', '-i', required=True,
                        help='Input OBJ file')
    parser.add_argument('--remain', '-r', type=int, default=-1,
                        help='# of vertices to be remained')

    args = parser.parse_args()

    mesh = TriMesh()
    mesh.load(args.input)

    simplify(mesh, remains=args.remain)

    mesh.save('output.obj')

if __name__ == '__main__':
    main()
