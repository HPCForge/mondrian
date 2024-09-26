#include <torch/extension.h>

#include <iostream>
#include <vector>

/**
 * Constructs an edge_index corresponding
 * to the different subdomains.
 */
torch::Tensor
subdomain_edge_index(
	torch::Tensor x_start,
	torch::Tensor x_end,
	torch::Tensor y_start,
	torch::Tensor y_end
	torch::Tensor coords_x,
	torch::Tensor coords_y) {
  auto n_subdomains = x_start.size(0) * y_start.size(0);
  std::vector<std::vector<int>> subdomains(n_subdomains, std::vector<int>(0)); 

  int subdomain = 0;
  for (int x_idx = 0; x_idx < x_start.size(0); ++x_idx) {
    for (int y_idx = 0; y_idx < y_start.size(0); ++y_idx) {
      for (int p_idx = 0; p_idx < coords_x.size(0); ++p_idx) {
        if (x_start[x_idx] < coords_x[p_idx] && x_end[x_idx] > coords_x[p_idx] &&
	    y_start[y_idx] < coords_y[p_idx] && y_end[y_idx] > coords_y[p_idx]) {
	  subdomains[subdomain].push_back(p_idx);
	}
      } 
      subdomain++;
    }
  }

  torch::Tensor edge_index;


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("subdomain_edge_index", &subdomain_edge_index, "");
}
