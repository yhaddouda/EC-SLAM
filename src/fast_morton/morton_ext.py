import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

# Make sure we only build for Orin (Ampere, SM 8.7). Must be set before `load`.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")

_src = str(Path(__file__).with_name("morton_cuda.cu"))

# First import triggers a build to ~/.cache/torch_extensions; later imports reuse it.
_morton = load(
    name="morton_ext",
    sources=[_src],
    extra_cuda_cflags=["-O3"],
    verbose=False,   # set True the first time if you want to see the build log
)

def morton3d_keys_cuda(positions01, R: int):
    # positions01: [N,3] tensor (might be float64 in EC-SLAM)
    
    # FIX: Ensure input is float32 (CUDA kernel expects float)
    if positions01.dtype != torch.float32:
        positions01 = positions01.float()
        
    return _morton.morton3d_keys(positions01.contiguous(), int(R))
