import argparse
import numpy as np
import ngsolve
from ngsolve import x, y, z, dx, grad
from netgen.csg import Pnt, OrthoBrick, CSGeometry 
from pathlib import Path

def main(args):
    rng = np.random.default_rng(seed=args.seed)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    outdir = Path(f'{args.outdir}/box3d/')
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Saving meshes to {outdir}')

    for i in range(args.count):
        print(f'Meshing [{i}/{args.count}]')
        dims = rng.uniform(low=args.min_size, high=args.max_size, size=3)

        cube = OrthoBrick(Pnt(0,0,0), Pnt(dims[0],dims[1],dims[2]))
        geo = CSGeometry()
        geo.Add(cube)
        mesh = geo.GenerateMesh(maxh=args.maxh)

        mesh_id = str(i).zfill(len(str(args.count)))
        mesh.Save(outdir / f'{mesh_id}.vol')

        del cube, geo, mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_size', default=1, type=int, help='set max cube size')
    parser.add_argument('--min_size', default=0.25, type=int, help='set min cube size')
    parser.add_argument('--seed', default=0, type=int, help='random seed for numpy')
    parser.add_argument('-n', '--count', default=100, type=int, help='number of meshes to generate')
    parser.add_argument('--maxh', default=0.1, type=float, help='maxh for mesh generation')
    parser.add_argument('--outdir', required=True, type=str, help='path to output directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())
