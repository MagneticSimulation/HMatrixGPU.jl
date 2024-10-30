module oneAPIExt

using HMatrixGPU
using oneAPI

function set_oneApi_backend()
    HMatrixGPU.all_backends[3] = oneAPI.oneAPIBackend()
    HMatrixGPU.set_backend("intel")
    return nothing
end

function __init__()
    return set_oneApi_backend()
end

end
