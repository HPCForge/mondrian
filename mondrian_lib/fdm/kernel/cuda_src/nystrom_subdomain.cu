/**
 * Setup problem to pass into 
 */
__global__
void construct_input(
   
  // src and target coordinates to
  // sample in each subdomain
  float* src_coord,
  float* tgt_coord,
  pytorch_type v
  pytorch_type u
  int* subdomain_lookup,
  int n_subdomains) {
  int sub = blockIdx.x;	
  int row_start = block_idx.y * blockDim.y + threadIdx.x;
  for (int sub = 0; sub < n_subdomains; sub += blockDim.x) {
    
  } 
}
