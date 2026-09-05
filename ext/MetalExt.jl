module MetalExt

using HMatrixGPU
using Metal

function set_metal_backend()
    HMatrixGPU.all_backends[4] = Metal.MetalBackend()
    HMatrixGPU.set_backend("apple")
    return nothing
end

# backend follows the data: assembled host arrays follow the device arrays
HMatrixGPU.to_backend(like::MtlArray, a::AbstractArray) = a isa MtlArray ? a : MtlArray(a)

function __init__()
    Metal.functional() && set_metal_backend()
    return nothing
end

end
