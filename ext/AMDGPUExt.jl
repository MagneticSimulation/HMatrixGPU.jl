module AMDGPUExt

using HMatrixGPU
using AMDGPU

function set_amd_backend()
    HMatrixGPU.all_backends[2] = AMDGPU.ROCBackend()
    HMatrixGPU.set_backend("amd")
    return nothing
end

function __init__()
    return set_amd_backend()
end

end
