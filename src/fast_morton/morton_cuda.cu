#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t part1by2(uint32_t v){
    v &= 0x3ffu;                              // 10 bits (R <= 1024)
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v << 8))  & 0x0300f00fu;
    v = (v | (v << 4))  & 0x030c30c3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

__global__ void morton_kernel(const float* __restrict__ pos,
                              uint32_t* __restrict__ out,
                              int N, int Rm1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float fx = fminf(fmaxf(pos[3*i+0], 0.f), 1.f) * Rm1;
    float fy = fminf(fmaxf(pos[3*i+1], 0.f), 1.f) * Rm1;
    float fz = fminf(fmaxf(pos[3*i+2], 0.f), 1.f) * Rm1;

    uint32_t x = (uint32_t)fx;
    uint32_t y = (uint32_t)fy;
    uint32_t z = (uint32_t)fz;

    uint32_t key = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
    out[i] = key;
}

torch::Tensor morton3d_keys(torch::Tensor pos, int R){
    TORCH_CHECK(pos.is_cuda(), "pos must be CUDA");
    TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");
    TORCH_CHECK(pos.scalar_type() == at::kFloat, "pos must be float32");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos shape [N,3]");

    const int N = (int)pos.size(0);
    auto keys = torch::empty({N}, pos.options().dtype(torch::kInt32));

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    morton_kernel<<<grid, block>>>(
        pos.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(keys.data_ptr<int32_t>()),
        N, R - 1
    );
    return keys;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("morton3d_keys", &morton3d_keys, "Morton keys (CUDA)");
}
