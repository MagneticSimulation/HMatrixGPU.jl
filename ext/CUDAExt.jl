module CUDAExt

using HMatrixGPU
using CUDA

CUDA.allowscalar(false)

function set_cuda_backend()
    HMatrixGPU.all_backends[1] = CUDA.CUDABackend()
    HMatrixGPU.set_backend("cuda")
    return nothing
end

function __init__()
    return set_cuda_backend()
end

end
