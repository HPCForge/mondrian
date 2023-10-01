import argparse
import numpy as np
import ngsolve
from ngsolve import x, y, z, dx, grad
from netgen.csg import Pnt, OrthoBrick, CSGeometry 
from pathlib import Path

def main(args):
    min_size = args.max_size / 4
    rng = np.random.default_rng(seed=args.seed)

    print(f'Saving meshes to {args.outdir}')
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        dims = rng.uniform(low=min_size, high=args.max_size, size=3)

        cube = OrthoBrick(Pnt(0,0,0), Pnt(dims[0],dims[1],dims[2]))
        geo = CSGeometry()
        geo.Add(cube)
        mesh = geo.GenerateMesh(maxh=0.25)
        # mesh.GenerateVolumeMesh()

        mesh_id = str(i).zfill(len(str(args.count)))
        out_file = f'{args.outdir}/box3d_{mesh_id}.vol' 
        mesh.Export(out_file, 'Neutral Format')
        with open(out_file, 'r+') as f:
            lines = f.readlines()
            lines.insert(0, 'NETGEN\n')
            f.seek(0)
            f.writelines(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_size', default=1, type=int, help='set max cube size')
    parser.add_argument('--seed', default=0, type=int, help='random seed for numpy')
    parser.add_argument('-n', '--count', default=10, type=int, help='number of meshes to generate')
    parser.add_argument('--maxh', default=1, type=float, help='maxh for mesh generation')
    parser.add_argument('--outdir', required=True, type=str, help='path to output directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
