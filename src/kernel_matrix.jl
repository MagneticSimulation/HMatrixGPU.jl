"""
    KernelMatrix(g, targets, sources; dims::Int = 1)

Lazy, matrix-free kernel matrix: the entry contract is `K[i,j] = g(x_i, y_j)`
— `g` receives the *coordinates* (column views) of the i-th target and j-th
source point, not indices. For `dims = d > 1` (vector kernels, `g` returning
a d x d matrix) the DOFs are flattened as `K[d(p-1)+c, d(q-1)+e] =
g(x_p, y_q)[c, e]`. Point sets accept both forms (d x N matrix / vector of
points); coordinates are used exactly as given (no scaling). `g` must be
total: regularize coincident points inside `g` (see the examples).
"""
struct KernelMatrix{F,TA<:AbstractMatrix{Float64},SB<:AbstractMatrix{Float64}} <: AbstractMatrix{Float64}
    g::F
    targets::TA      # d x N_t (columns = points), host
    sources::SB      # d x N_s, host
    dims::Int        # DOF per point (1 = scalar kernel)
end

function KernelMatrix(g, targets, sources; dims::Int = 1)
    Xt = _point_matrix(targets)
    Ys = _point_matrix(sources)
    dims >= 1 || error("`dims` must be a positive integer")
    size(Xt, 1) == size(Ys, 1) ||
        error("target and source points must have the same spatial dimension")
    # validation probe on an off-diagonal pair (first target x last source),
    # so a singular self-pair is not hit: errors surface at construction,
    # not mid-assembly
    v = g(view(Xt, :, 1), view(Ys, :, size(Ys, 2)))
    if dims == 1
        v isa Number ||
            error("g returned $(typeof(v)) for dims=1; expected a Number " *
                  "(for a matrix-valued kernel pass dims=d)")
    else
        (v isa AbstractMatrix && size(v) == (dims, dims)) ||
            error("g returned $(typeof(v)) for dims=$dims; expected a " *
                  "$(dims)x$(dims) matrix")
    end
    return KernelMatrix(g, Xt, Ys, dims)
end

Base.size(K::KernelMatrix) = (K.dims * size(K.targets, 2),
                              K.dims * size(K.sources, 2))

function Base.getindex(K::KernelMatrix, i::Int, j::Int)
    d = K.dims
    p, c = div(i - 1, d) + 1, rem(i - 1, d) + 1
    q, e = div(j - 1, d) + 1, rem(j - 1, d) + 1
    v = K.g(view(K.targets, :, p), view(K.sources, :, q))
    return d == 1 ? Float64(v) : Float64(v[c, e])
end

# batch entry — the only form the ACA assembly queries. For vector kernels
# one g call per point pair fills the whole d x d cell (single-entry memo;
# ACA queries arrive in whole cells, so the memo always hits)
function Base.getindex(K::KernelMatrix, I::AbstractVector{Int},
                       J::AbstractVector{Int})
    d = K.dims
    out = Matrix{Float64}(undef, length(I), length(J))
    if d == 1
        for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
            out[ii, jj] = K.g(view(K.targets, :, i), view(K.sources, :, j))
        end
        return out
    end
    last = (0, 0)
    v = nothing
    for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        p, c = div(i - 1, d) + 1, rem(i - 1, d) + 1
        q, e = div(j - 1, d) + 1, rem(j - 1, d) + 1
        if (p, q) != last
            v = K.g(view(K.targets, :, p), view(K.sources, :, q))
            last = (p, q)
        end
        out[ii, jj] = v[c, e]
    end
    return out
end
