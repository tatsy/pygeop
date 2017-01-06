import argparse

from pygeop.geom3d import *

def main():
    parser = argparse.ArgumentParser(description='Demo for mesh simplification.')
    parser.add_argument('--input', '-i', required=True)

    args = parser.parse_args()

    mesh = TriMesh()
    mesh.load(args.input)

    simplify(mesh)

    mesh.save('output.obj')

if __name__ == '__main__':
    main()
