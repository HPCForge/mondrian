import argparse
import numpy as np
import h5py
import glob
from dataclasses import dataclass
from typing import List, Tuple
import mfem.ser as mfem
from ctypes import c_int, c_double

def main(args):
    mesh_files = glob.glob(f'{args.srcdir}/*.mesh')
    sol_files = glob.glob(f'{args.srcdir}/*.gf')

    out_file = args.outdir
    if not out_file.endswith('.hdf5'):
        out_file += '.hdf5'

    print(f'converting data in {args.srcdir}')
    print(f'writing to {out_file}')

    with h5py.File(out_file, 'w') as f:
        for idx, (mesh_file, sol_file) in enumerate(zip(mesh_files, sol_files)):
            edges, vertices, sol = load_mesh(mesh_file, sol_file)
            assert vertices.shape[0] == sol.shape[0]
            dataset_id = str(idx).zfill(len(str(len(mesh_files))))
            g = f.create_group(dataset_id)
            g.create_dataset('edges', data=edges)
            g.create_dataset('vertices', data=vertices)
            g.create_dataset('features', data=sol)

def load_mesh(mesh_file, sol_file):
    mesh = mfem.Mesh(mesh_file)

    vertices = mesh.GetVertexArray()
    ev = mesh.GetEdgeVertexTable()

    edges = []
    for edge_id in range(mesh.GetNEdges()):
        row_size = ev.RowSize(edge_id)
        # GetRow returns a SWIG int*, this has to be read from the address...
        edge = np.array((c_int * row_size).from_address(int(ev.GetRow(edge_id))))
        edges.append(edge)

    # edges will be [2, num_edges]
    edges = np.stack(edges, axis=1)

    # vertices will be [num_vertices, dim]
    vertices = np.stack(vertices, axis=0)

    gf = mfem.GridFunction(mesh, sol_file)
    # GetTrueDofs writes to its argument in-place.
    dof = mfem.Vector()
    gf.GetTrueDofs(dof)
    # GetData return a swig double*
    dofs = list((c_double * dof.Size()).from_address(int(dof.GetData())))
    dofs = np.array(dofs)
    dofs[dofs < 1e-300] = 0

    return edges, vertices, dofs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', required=True, help='path to solved meshes')
    parser.add_argument('--outdir', required=True, help='path to write hdf5 file')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
