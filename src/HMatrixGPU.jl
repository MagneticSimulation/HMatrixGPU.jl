module HMatrixGPU

using Printf
using KernelAbstractions
using RecipesBase

include("tree.jl")
include("block.jl")
include("aca.jl")
include("hmatrix.jl")
include("plotting.jl")
include("kernel_matrix.jl")
include("assembly_gpu.jl")
include("mult.jl")

export ClusterTree, BlockTree, ACA_plus, HMatrix, info, KernelMatrix
export hmatrix_blocks

const groupsize = Ref(512)
function set_groupsize(x)
    return groupsize[] = x
end
export set_groupsize

export @using_gpu
macro using_gpu()
    quote
        try
            using CUDA
        catch
        end
        try
            using AMDGPU
        catch
        end
        try
            using oneAPI
        catch
        end
        try
            using Metal
        catch
        end
    end
end

# find an already-loaded vendor module by package name without depending
# on it: Base.loaded_modules is a Vector{PkgId} on Julia <= 1.11 and a
# Dict{PkgId,Module} on >= 1.12
function _loaded_vendor_module(pkg::String)
    for entry in Base.loaded_modules
        pid = entry isa Pair ? entry.first : entry
        pid.name == pkg || continue
        mod = entry isa Pair ? entry.second : Base.root_module(pid)
        mod === nothing && continue
        return mod
    end
    return nothing
end

const _backend_table = Dict("cpu" => nothing,
    "cuda" => (:CUDA, :CUDABackend), "nvidia" => (:CUDA, :CUDABackend),
    "amd" => (:AMDGPU, :ROCBackend), "roc" => (:AMDGPU, :ROCBackend),
    "amdgpu" => (:AMDGPU, :ROCBackend),
    "oneapi" => (:oneAPI, :oneAPIBackend), "intel" => (:oneAPI, :oneAPIBackend),
    "metal" => (:Metal, :MetalBackend), "apple" => (:Metal, :MetalBackend))

"""
    backend_from_name(name) -> KernelAbstractions.Backend

Strict resolver for backend names ("cpu", "cuda"/"nvidia", "amd"/"roc",
"oneapi"/"intel", "metal"/"apple"); the `backend=` keyword uses it. Errors
with an actionable message when the vendor package is not loaded or has no
functional device. There is deliberately no auto-detection: the user (or a
higher-level package) chooses the backend explicitly.
"""
function backend_from_name(name)::KernelAbstractions.Backend
    key = lowercase(string(name))
    haskey(_backend_table, key) ||
        error("unknown backend `$(name)` (expected \"cpu\", \"cuda\", \"amd\", \"oneapi\" or \"metal\")")
    spec = _backend_table[key]
    spec === nothing && return KernelAbstractions.CPU()
    pkg, sym = spec
    mod = _loaded_vendor_module(String(pkg))
    mod === nothing &&
        error("backend \"$(name)\" requires $(pkg).jl — run `using $(pkg)` first")
    isdefined(mod, sym) ||
        error("$(pkg).jl is loaded but does not define $(sym) (version mismatch?)")
    if isdefined(mod, :functional) && !mod.functional()
        error("backend \"$(name)\" was requested but no functional device was detected")
    end
    return getproperty(mod, sym)()
end

"""
    move_to_backend(B, a)

Move the array `a` to the backend `B` (a no-op when `a` is already there).
"""
function move_to_backend(B::KernelAbstractions.Backend, a::AbstractArray)
    KernelAbstractions.get_backend(a) == B && return a
    d = KernelAbstractions.zeros(B, eltype(a), size(a))
    copyto!(d, a)
    return d
end

"""
    to_backend(like, a)

Move the array `a` to the backend where `like` lives ("backend follows the
data"); a no-op when both live on the same backend.
"""
to_backend(like::AbstractArray, a::AbstractArray) =
    move_to_backend(KernelAbstractions.get_backend(like), a)

function kernel_array(B::KernelAbstractions.Backend, a::AbstractArray)
    return move_to_backend(B, a)
end

function create_zeros(B::KernelAbstractions.Backend, ::Type{T}, dims...) where {T}
    return KernelAbstractions.zeros(B, T, dims)
end

function create_ones(B::KernelAbstractions.Backend, ::Type{T}, dims...) where {T}
    return KernelAbstractions.ones(B, T, dims)
end

end
