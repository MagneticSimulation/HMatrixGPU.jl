module CUDAExt

using HMatrixGPU
using CUDA

CUDA.allowscalar(false)

function set_cuda_backend()
    HMatrixGPU.all_backends[1] = CUDA.CUDABackend()
    HMatrixGPU.set_backend("cuda")
    return nothing
end

# backend follows the data: assembled host arrays follow the device arrays
HMatrixGPU.to_backend(like::CuArray, a::AbstractArray) = a isa CuArray ? a : CuArray(a)

function __init__()
    return set_cuda_backend()
end

end
