module AMDGPUExt

using HMatrixGPU
using AMDGPU

function set_amd_backend()
    HMatrixGPU.all_backends[2] = AMDGPU.ROCBackend()
    HMatrixGPU.set_backend("amd")
    return nothing
end

# backend follows the data: assembled host arrays follow the device arrays
HMatrixGPU.to_backend(like::ROCArray, a::AbstractArray) = a isa ROCArray ? a : ROCArray(a)

function __init__()
    AMDGPU.functional() && set_amd_backend()
    return nothing
end

end
