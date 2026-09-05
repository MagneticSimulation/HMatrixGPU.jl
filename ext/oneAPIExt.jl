module oneAPIExt

using HMatrixGPU
using oneAPI

function set_oneApi_backend()
    HMatrixGPU.all_backends[3] = oneAPI.oneAPIBackend()
    HMatrixGPU.set_backend("intel")
    return nothing
end

# backend follows the data: assembled host arrays follow the device arrays
HMatrixGPU.to_backend(like::oneArray, a::AbstractArray) = a isa oneArray ? a : oneArray(a)

function __init__()
    oneAPI.functional() && set_oneApi_backend()
    return nothing
end

end
