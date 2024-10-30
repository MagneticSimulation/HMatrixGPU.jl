module MetalExt

using HMatrixGPU
using Metal

function set_metal_backend()
    HMatrixGPU.all_backends[4] = Metal.MetalBackend()
    HMatrixGPU.set_backend("apple")
    return nothing
end

function __init__()
    return set_metal_backend()
end

end
