import torch 

def get_subdomain_indices(overlap,
                          x_res,
                          y_res,
                          domain_size_x,
                          domain_size_y,
                          subdomain_size_x,
                          subdomain_size_y):
    # get the resolution per 1x1 box
    res_per_global_x = x_res / domain_size_x
    res_per_global_y = y_res / domain_size_y

    # get the resolution of each subdomain
    res_per_x = int(subdomain_size_x * res_per_global_x)
    res_per_y = int(subdomain_size_y * res_per_global_y)
    
    # get the stride of subdomains
    subdomain_step_x = int((1 - overlap) * res_per_x)
    subdomain_step_y = int((1 - overlap) * res_per_y)
    
    x_idx = torch.arange(0, x_res, subdomain_step_x)
    y_idx = torch.arange(0, y_res, subdomain_step_y)

    # shift the subdomains on the edge into the interior
    x_idx[-1] = x_res - res_per_x
    y_idx[-1] = y_res - res_per_y
    x_idx[-2] -= res_per_x // 4
    y_idx[-2] -= res_per_y // 4
    return x_idx, y_idx, res_per_x, res_per_y
