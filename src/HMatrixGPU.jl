module HMatrixGPU

using Printf
using KernelAbstractions

include("tree.jl")
include("block.jl")
include("aca.jl")
include("hmatrix.jl")
include("mult.jl")

export ClusterTree, BlockTree, ACA_plus, HMatrix, info


const default_backend = Backend[CPU()]
const all_backends = Backend[CPU(), CPU(), CPU(), CPU()]


export set_backend
"""
    set_backend(backend="cuda")

Set the backend of HMatrixGPU. 

The available options and their corresponding hardware and backends are shown below:

| Option                 | Hardware            | Backend                  |
| :--------------------- | :------------------ | :------------------------ |
| "cpu"                  | CPU                 | `KernelAbstractions.CPU()` |
| "cuda" or "nvidia"     | NVIDIA GPU          | `CUDA.CUDABackend()`     |
| "amd" or "roc"         | AMD GPU             | `AMDGPU.ROCBackend()`    |
| "oneAPI" or "intel"    | Intel GPU           | `oneAPI.oneAPIBackend()` |
| "metal" or "apple"     | Apple GPU           | `Metal.MetalBackend()`   |

# Examples

To set the backend to use CUDA (NVIDIA GPU):

```julia
using HMatrixGPU
using CUDA
```

To set the backend to use the CPU/CUDA:
```
set_backend("cpu")
set_backend("cuda")
```

"""
function set_backend(backend="cuda")
    backend_names = ["CUDA", "AMDGPU", "oneAPI", "Metal"]
    card_id = 0
    x = lowercase(backend)
    if x == "cuda" || x == "nvidia"
        card_id = 1
    elseif x == "amd" || x == "roc" || x == "amdgpu"
        card_id = 2
    elseif x == "oneapi" || x == "intel"
        card_id = 3
    elseif x == "metal" || x == "apple"
        card_id = 4
    end

    if card_id > 0
        default_backend[] = all_backends[card_id]
        backend_name = backend_names[card_id]
        if Base.find_package(backend_name) === nothing
            @info(@sprintf("Please install %s.jl!", backend_name))
            return false
        end

        if default_backend[] == CPU()
            @info(@sprintf("Please import %s!", backend_name))
            return false
        end
    else
        default_backend[] = CPU()
    end

    @info(@sprintf("Switch the backend to %s", default_backend[]))
    return true
end


end
