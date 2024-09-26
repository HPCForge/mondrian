import torch
from torch_geometric.nn.pool.voxel_grid import voxel_grid
from torch_cluster import grid_cluster

import math
import numba
from numba import cuda

def compute_clusters(
        coords,
        batch_idx,
        n_subdomains_x,
        n_subdomains_y):
    r"""
    Args:
        coords: [N, 2]
        batch_idx [N], indicates which batch a coord belongs to
    Returns:
        cluster: [N], indicate which cluster a coord belongs to
    """
    assert torch.is_floating_point(coords)

    size_x = 2 / n_subdomains_x
    size_y = 2 / n_subdomains_y
    size = torch.tensor([size_x, size_y])
    start = [-1, -1]
    end = [1 - size_x - 0.1, 1 - size_y - 0.1]
    return voxel_grid(coords, size, batch=batch_idx, start=start, end=end), size_x, size_y

def subdomains_to_edge_index(node_ids,
                             subdomains):
    # sort subdomains and permute node_ids
    subdomains, indices = torch.sort(subdomains)
    node_ids = node_ids[indices].clone()

    # convert subdomains to a segment list,
    # node_ids[segment[i]:segment[i+1]] contains the nodes in subdomain i 
    _, counts = torch.unique(subdomains, return_counts=True)
    segment = torch.cumsum(counts, dim=0)
    segment = torch.cat((torch.tensor([0], device=segment.device), segment))

    # Each node is connected to everything else in subdomain.
    # This assumes self-connections and undirected edges.
    degrees = counts ** 2
    edge_index = torch.zeros((2, degrees.sum()),
                             dtype=torch.long,
                             device=node_ids.device)

    edge_index_segment = torch.cat((torch.tensor([0], device=edge_index.device),
                                    torch.cumsum(degrees, dim=0)))

    # segment[i + 1] - segment[i] is the size of subdomain i
    max_subdomain_size = int(torch.max(segment[1:] - segment[:-1])) 

    block_dim = (32, 2)
    grid_dim_x = math.ceil(max_subdomain_size / block_dim[0])
    grid_dim_y = math.ceil(max_subdomain_size / block_dim[1])
    grid_dim = grid_dim_x, grid_dim_y, segment.size(0) - 1

    build_edge_index[grid_dim, block_dim](
            segment,
            int(segment.size(0)),
            node_ids,
            int(node_ids.size(0)),
            edge_index_segment,
            int(edge_index_segment.size(0)),
            edge_index,
            int(edge_index.size(1)),
        ) 

    return edge_index

@cuda.jit(
    'void('
    'int64[:], int64,'
    'int64[:], int64,'
    'int64[:], int64,'
    'int64[:, :], int64)'
)
def build_edge_index(segment,
                     segment_len,
                     node_id,
                     node_id_len,
                     edge_index_segment,
                     edge_index_segment_len,
                     edge_index,
                     edge_index_len):
    subdomain_idx = cuda.blockIdx.z

    if subdomain_idx < segment_len - 1:
        read_start = segment[subdomain_idx]
        read_end = segment[subdomain_idx + 1]
        write_start = edge_index_segment[subdomain_idx]
        write_end = edge_index_segment[subdomain_idx + 1]

        start_idx_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        start_idx_j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        subdomain_size = read_end - read_start

        for i in range(start_idx_i, subdomain_size, cuda.gridDim.x):
            for j in range(start_idx_j, subdomain_size, cuda.gridDim.y):
                write_idx = write_start + i * subdomain_size + j
                if write_idx < write_end:
                    edge_index[0, write_idx] = node_id[read_start + i] 
                    edge_index[1, write_idx] = node_id[read_start + j]

if __name__ == '__main__':
    coords = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).float()
    batch_idx = torch.tensor([0, 0, 0, 0])
    c = compute_clusters(coords, batch_idx, 8, 8)
    #print(c)

    coords = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).float()
    batch_idx = torch.tensor([0, 0, 1, 1])
    c = compute_clusters(coords, batch_idx, 8, 8)
    #print(c)

    node_ids = torch.arange(4).cuda()
    subdomains = torch.tensor([0, 0, 1, 1]).cuda()
    edge_index = subdomains_to_edge_index(node_ids,
                                          subdomains)
    print(edge_index)

    node_ids = torch.arange(100).cuda()
    subdomains = torch.zeros(100).cuda()
    subdomains[:25] = 0
    subdomains[25:50] = 1
    subdomains[50:] = 2
    edge_index = subdomains_to_edge_index(node_ids,
                                          subdomains)
    print(edge_index.size())
    print(edge_index)
