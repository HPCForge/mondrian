import argparse
import numpy as np
import h5py
import glob
from dataclasses import dataclass
from typing import List, Tuple
import mfem.ser as mfem
from ctypes import c_int, c_double

def main(args):
    mesh_files = sorted(glob.glob(f'{args.srcdir}/*.mesh'))
    sol_files = sorted(glob.glob(f'{args.srcdir}/*.gf'))

    print(mesh_files)

    out_file = args.outdir
    if not out_file.endswith('.hdf5'):
        out_file += '.hdf5'

    print(f'converting data in {args.srcdir}')
    print(f'writing to {out_file}')

    with h5py.File(out_file, 'w') as f:
        for idx, (mesh_file, sol_file) in enumerate(zip(mesh_files, sol_files)):
            edges, edges_len, vertices, sol, bdr_vertex_ids, init_vertex_features = load_mesh(mesh_file, sol_file)
            assert vertices.shape[0] == sol.shape[0]
            dataset_id = str(idx).zfill(len(str(len(mesh_files))))
            g = f.create_group(dataset_id)
            g.create_dataset('edges', data=edges)
            g.create_dataset('edge-len', data=edges_len)
            g.create_dataset('vertices', data=vertices)
            g.create_dataset('boundary-vertex-ids', data=bdr_vertex_ids)
            g.create_dataset('boundary-values', data=init_vertex_features)
            g.create_dataset('solution', data=sol)

            print(edges.shape, edges_len.shape, bdr_vertex_ids.shape, init_vertex_features.shape)

def load_mesh(mesh_file, sol_file):
    r"""
    Mesh conversion has several parts:
        1. Load the edges
        2. Get the length of each edge (used as edge feature in pyg)
        3. get vertex coordinates (optionally used in pyg)
        4. get boundary nodes (and the solution at boundary nodes)
        5. get the interior nodes (to use as ground truth) 
    """
    mesh = mfem.Mesh(mesh_file)

    vertices = mesh.GetVertexArray()
    ev = mesh.GetEdgeVertexTable()
    elem_to_edge = mesh.ElementToEdgeTable()

    # get edges
    edges = []
    edges_len = []
    for edge_id in range(mesh.GetNEdges()):
        row_size = ev.RowSize(edge_id)
        # GetRow returns a SWIG int*, this has to be read from the address...
        edge = np.array((c_int * row_size).from_address(int(ev.GetRow(edge_id))), dtype=int)
        edges.append(edge)

        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        edge_len = np.linalg.norm(v1 - v2)
        edges_len.append(edge_len)
    # edges will be [2, num_edges]
    edges = np.stack(edges, axis=1)
    # edges_len will be num_edges, 1
    edges_len = np.stack(edges_len, axis=0)

    # vertices will be [num_vertices, dim]
    vertices = np.stack(vertices, axis=0)

    # get boundary nodes
    bdr_vertices = []
    for bdr_id in range(mesh.GetNBE()):
        bdr_elem = mesh.GetBdrElement(bdr_id)
        nvertices = bdr_elem.GetNVertices()
        elem_vertices = list((c_int * nvertices).from_address(int(bdr_elem.GetVertices())))
        elem_vertices = np.array(elem_vertices)
        bdr_vertices.extend(elem_vertices)
    bdr_vertex_id = np.unique(np.array(bdr_vertices))

    # TODO: Working with GridFunction like this may not
    # generalize to other problems with multiple variables.
    gf = mfem.GridFunction(mesh, sol_file)
    
    #nodal_vals = mfem.Vector()
    #gf.GetNodalValues(nodal_vals, 1)
    #nv = np.array((c_double * nodal_vals.Size()).from_address(int(nodal_vals.GetData())))
    #nv = np.expand_dims(nv, 1)
    #print(nv)

    
    # GetTrueDofs writes to its argument in-place.
    dof = mfem.Vector()
    gf.GetTrueDofs(dof)
    # GetData return a swig double*
    dofs = np.empty(dof.Size())
    for i in range(dof.Size()):
        dofs[i] = dof[i]
    dofs = np.expand_dims(dofs, 1)
    #dofs = np.array((c_double * dof.Size()).from_address(int(dof.GetData())))
    #dofs = np.expand_dims(dofs, 1)
    #dofs[dofs < 1e-300] = 0
    
    # initially, we know solution only on boundary
    init_vertex_features = np.zeros_like(dofs)
    init_vertex_features[bdr_vertex_id] = dofs[bdr_vertex_id]

    return edges, edges_len, vertices, dofs, bdr_vertex_id, init_vertex_features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', required=True, help='path to solved meshes')
    parser.add_argument('--outdir', required=True, help='path to write hdf5 file')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
