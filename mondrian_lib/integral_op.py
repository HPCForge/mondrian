import einops
import torch
from torch_geometric.utils import scatter

def subdomain_integral_op(
        v,
        src_subdomains,
        src_coords,
        tgt_coords=None,
        tgt_subdomains=None,
        op_v=None,
        op_src_kernel=None,
        op_tgt_kernel=None,
        op_tgt_out=None):
    r"""
    Apply an integral operator to each subdomain.
    Computes u|_{D_i} = op_tgt_out(  tgt_coords, \int_{D_i} op_v(v)  )
    Args:
        v: [N, c]
        src_subdomains: [N]
        src_coords: [N, 2]
        tgt_coords: [N', 2],
        tgt_subdomains: [N']
        op_v: A function applied directly to v before integrating
        op_src: A matrix-valued function applied to src_coords before integrating 
        op_tgt: A matrix-calued function applied to tgt_coords after integrating
    """
    if tgt_coords is None:
        tgt_coords = src_coords
    if tgt_subdomains is None:
        tgt_subdomains = src_subdomains

    if op_src_kernel:
        v_enc = einops.einsum(v, op_src(src_coords), 'N i, N h i -> N h')
    elif op_v:
        v_enc = op_v(torch.cat((v, src_coords), dim=1))

    # (approximately) compute integral of v in each subdomain.
    sub_int_v = scatter(v_enc, src_subdomains, dim=0, reduce='mean')

    # repeat for each tgt_coord
    subdomain_ids, tgt_sub_counts = torch.unique(tgt_subdomains, return_counts=True)
    # subdomains with no entries are not included in the counts, so we add zeros in to get
    # the correct input size for repeat_interleave.
    fill_counts = torch.zeros(sub_int_v.size(0), dtype=torch.long, device=v.device)
    fill_counts[subdomain_ids] = tgt_sub_counts

    # integral of each subdomain, repeated for every point in tgt_coords
    sub_int_v = torch.repeat_interleave(sub_int_v, fill_counts, dim=0)

    assert (op_tgt_kernel is None) or (op_tgt_out is None)
    if op_tgt_kernel:
        sub_int_v = einops.einsum(sub_int_v, op_tgt(tgt_coords), 'N i, N h i -> N h')   
    elif op_tgt_out:
        u = op_tgt_out(torch.cat((sub_int_v, tgt_coords), dim=1))
    else:
        u = sub_int_v

    return u, tgt_coords


def subdomain_integral_to_reference(
        n_subdomains,
        v,
        src_coords,
        src_subdomains=None,
        src_batch=None,
        tgt_coords=None,
        op_v=None,
        kernel=None,
        op_src_kernel=None,
        op_tgt_kernel=None,
        op_tgt_out=None):
    r"""
    Apply an integral operator to each subdomain.
    Computes u|_{D_i} = op_tgt_out(  tgt_coords, \int_{D_i} op_v(v)  )
    Args:
        v: [N, c]
        src_subdomains: [N]
        src_coords: [N, 2]
        tgt_coords: [N', 2],
        tgt_subdomains: [N']
        op_v: A function applied directly to v before integrating
        op_src: A matrix-valued function applied to src_coords before integrating 
        op_tgt: A matrix-calued function applied to tgt_coords after integrating
    """
    if op_v:
        v = op_v(v)

    if op_src_kernel:
        src_coords = op_src_kernel(src_coords)
        v = einops.einsum(src_coords, v, 'J h n, J n -> J h')

    # (approximately) compute integral of v in each subdomain.
    # [S, h]
    sub_int_v = scatter(v, src_subdomains, dim=0, reduce='mean')

    # integral of each subdomain, repeated for every point in reference domain tgt_coords
    # [S * disc, h]
    #print(src_subdomains.max())
    #print(sub_int_v.size(), n_subdomains)
    #sub_int_v = torch.repeat_interleave(sub_int_v, tgt_coords.size(0), dim=0)

    # for every subdomain, project onto reference grid
    if op_tgt_kernel:
        u = einops.einsum(sub_int_v, op_tgt_kernel(tgt_coords), 'S h, J m h -> S J m')

    non_empty_subdomain_ids = torch.unique(src_subdomains)

    # allocate enough space for theoretical max number of subdomains
    batch_size = int(src_batch.max() + 1)
    print(n_subdomains, batch_size)
    out = torch.zeros(((n_subdomains + 1) * batch_size, u.size(1), u.size(2)))
    out[non_empty_subdomain_ids] = u
    
    # u is [S, J, m], where S is the number subdomains across the entire batch.
    # Each subdomain across batches is given a unique, increasing ID. So S % batch_size == 0.
    # We can just do S // batch_size to assign each reference domain to its batch
    print(out.size(), batch_size)
    assert out.size(0) % batch_size == 0

    # [B, S', J, m] S' is the number of subdomains per item
    out = torch.unflatten(out, dim=0, sizes=(batch_size, u.size(0) // batch_size))
    
    return out
