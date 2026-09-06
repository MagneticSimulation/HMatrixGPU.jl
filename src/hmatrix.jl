using SparseArrays

"""
    mutable struct HMatrix

Hierarchical matrix stored as three CSR-compressed operators on the active
backend (CPU, CUDA, AMDGPU, oneAPI or Metal via KernelAbstractions). The
matrix-vector product runs as three thread-group kernels without shared
memory, barriers or atomics:

1. `xord = x[source_index_map]` — the input vector permuted into cluster order,
2. `vx = V * xord`                — one workgroup per rank row of the far field,
3. `y[tmap[i]] = (D + U) * [xord; vx]` — one workgroup per matrix row.

Since the near-field and `U` row sets partition the matrix rows and the target
index map is a permutation, every entry of `y` is written exactly once.

# Fields
- `m::Int`, `n::Int`: dimensions of the full matrix.
- `target_index_map::AbstractVector{Int}`: permutation from cluster-ordered
  rows to original row indices.
- `source_index_map::AbstractVector{Int}`: permutation from cluster-ordered
  columns to original column indices.
- `near_rowptr/near_colval/near_data`: CSR of the near-field (dense) blocks.
  Column indices are cluster positions (`Int32`).
- `v_rowptr/v_colval/v_data`: CSR of the far-field `V` factors, one row per
  rank row of every admissible block.
- `u_rowptr/u_colval/u_data`: CSR of the far-field `U` factors, rows in
  cluster order; column indices address the `vx` buffer.
- `vx_buffer::AbstractVector{T}`: intermediate buffer (length = total rank rows).
- `x_buffer::AbstractVector{T}`: input vector permuted into cluster order.
- `ranks::Vector{Int}`: rank of each admissible block (host side, for `info`).
- `ndense::Int`, `napprox::Int`: number of near-field and far-field blocks.
"""
mutable struct HMatrix{T}
    m::Int
    n::Int
    target_index_map::AbstractVector{<:Integer}
    source_index_map::AbstractVector{<:Integer}
    near_rowptr::AbstractVector{<:Integer}
    near_colval::AbstractVector{<:Integer}
    near_data::AbstractVector{T}
    v_rowptr::AbstractVector{<:Integer}
    v_colval::AbstractVector{<:Integer}
    v_data::AbstractVector{T}
    u_rowptr::AbstractVector{<:Integer}
    u_colval::AbstractVector{<:Integer}
    u_data::AbstractVector{T}
    vx_buffer::AbstractVector{T}
    x_buffer::AbstractVector{T}
    ranks::Vector{Int}
    ndense::Int
    napprox::Int
end

Base.size(h::HMatrix) = (h.m, h.n)

# GPU batched dense-K assembly: generic over KernelAbstractions backends with a
# working device svd (src/assembly_gpu.jl). When the probe fails (or for
# matrix-free kernels) the CPU ACA path is used.
gpu_dense_assembly_available(B::KernelAbstractions.Backend) = device_svd_available(B)

# landing backend: like= > backend= > device of the primary data > CPU()
# (no global default — the package keeps no backend state)
function _resolve_landing(like, backend, datas...)
    (like !== nothing && backend !== nothing) &&
        error("specify either `like` or `backend`, not both")
    like !== nothing && return KernelAbstractions.get_backend(like)
    backend !== nothing && return (backend isa KernelAbstractions.Backend ?
                                   backend : backend_from_name(backend))
    seen = nothing
    for a in datas
        # a lazy/custom kernel struct carries no backend information and
        # KernelAbstractions.get_backend errors for array types it does not
        # know: treat such data as host data
        Ba = try
            KernelAbstractions.get_backend(a)
        catch
            KernelAbstractions.CPU()
        end
        Ba isa KernelAbstractions.CPU && continue
        seen === nothing && (seen = Ba; continue)
        Ba == seen || error("input data lives on different backends ($seen vs $Ba)")
    end
    return something(seen, KernelAbstractions.CPU())
end

"""
    HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5,
            index_map_using_cpu=true, svd_recompress=true)

Creates a hierarchical matrix (`HMatrix`) from a given matrix `K` and two cluster
trees `X` and `Y`.

# Arguments
- `K::AbstractMatrix`: The matrix to decompose hierarchically (may be matrix-free;
  only batched `getindex` queries are used during assembly).
- `X::ClusterTree`: Cluster tree representing the target partitioning.
- `Y::ClusterTree`: Cluster tree representing the source partitioning.
- `eta::Float64`: Admissibility parameter controlling the low-rank approximation.
- `eps::Float64`: Tolerance level for approximation error.
- The compressed structure is backend-resident: on the CPU all factor arrays
  are ordinary CPU arrays (the same kernels run on the CPU), on a GPU backend
  they are device arrays.
- `index_map_using_cpu`: Keep the cluster index maps on the CPU during tree
  construction (default; almost always the right choice).
- `svd_recompress`: Recompress the ACA factors with a truncated SVD (default).
- `backend`: where the factor arrays land — a name (`"cpu"`, `"cuda"`, `"amd"`,
  `"oneapi"`, `"metal"`) or a KernelAbstractions backend object. Requesting a
  GPU backend whose vendor package is not loaded errors with
  "run `using CUDA` first"; a loaded package without a functional device
  errors as well (no auto-detection, no silent fallback).
- `like::Union{Nothing,AbstractArray}`: alternative to `backend` — move all
  factor arrays to the backend where `like` lives (backend follows the data).
- With neither keyword given, the factors follow the device of the primary
  data `K` (a device-resident `K` keeps the matrix on its device); host data
  defaults to the CPU.

# Contracts
- The device is fixed at construction and the package keeps no backend state:
  the landing backend is resolved as `like=` > `backend=` > the device of the
  primary data `K` > `CPU()`. Loading a vendor package (`using CUDA`) has zero
  side effects — it neither switches a global backend nor affects later
  constructions in any way.
- Instances are independent, so CPU and GPU `HMatrix` instances (even from
  different vendors) can coexist in one process and their matvecs can be
  interleaved freely.
- `row_block_size`/`col_block_size` default to `nothing`, which follows the
  trees' `dims` (`X.dims`/`Y.dims`): vector problems with `dims=3` pivot one
  whole cell (3 components) per group, which prevents the anisotropy of vector
  kernels from starving the weak components. Explicit values override the
  default.
- `mul!`/`*` require `x` (and `result`) to live on the same backend as the
  matrix; cross-device inputs error out instead of being moved.
- A dense `K` on a GPU backend assembles on the GPU fast path, which
  ignores `svd_recompress`/`row_block_size`/`col_block_size` (the randomized
  SVD truncates optimally without grouped pivoting).
- Complex-valued kernels are not supported (the constructor errors).
"""
function HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5,
                 index_map_using_cpu=true, svd_recompress=true,
                 row_block_size=nothing, col_block_size=nothing,
                 backend=nothing, like=nothing)
    eltype(K) <: Complex &&
        error("HMatrix does not support complex-valued kernels (eltype(K) = $(eltype(K)))")

    # resolve the landing backend (like= > backend= > device of K > CPU())
    B = _resolve_landing(like, backend, K)

    block_tree = BlockTree(X, Y; eta=eta, index_map_using_cpu=index_map_using_cpu,
                           backend=B)
    merge_dense_matrices!(block_tree.root)

    # Traverse the block tree to gather dense and approximated blocks
    dense_blocks, approx_blocks = traverse(block_tree)

    target_map = collect(Int, block_tree.target_index_map)
    source_map = collect(Int, block_tree.source_index_map)

    # GPU batched fast path: dense kernels on a GPU backend whose device svd
    # probe passes (src/assembly_gpu.jl). Matrix-free kernels and CPU backends
    # use the CPU ACA path; row_block_size/col_block_size only apply to the CPU
    # path (the GPU randomized SVD needs no grouped pivoting).
    if !(B isa KernelAbstractions.CPU) && gpu_dense_assembly_available(B) &&
       K isa DenseMatrix
        K_gpu, dense_all, U_factors, V_factors, ranks, approx_block_indices =
            build_matrices_gpu_dense(
                Matrix(K), B, block_tree.target_index_map, block_tree.source_index_map,
                dense_blocks, approx_blocks; eps=eps)
        return build_csr_hmatrix_gpu(K_gpu, B, size(K, 1), size(K, 2),
                                     target_map, source_map, dense_all,
                                     approx_block_indices, U_factors, V_factors,
                                     ranks; T=eltype(K))
    end

    # CPU path: grouped ACA+ per block (threaded), then CSR packing. The block
    # sizes default to the trees' dims: cluster ranges of a dims-expanded tree
    # are multiples of dims, so the grouped pivoting stays divisibility-safe.
    rb = something(row_block_size, X.dims)
    cb = something(col_block_size, Y.dims)
    dense_matrices, U_matrices, V_matrices, dense_block_indices, approx_block_indices = build_matrices(K,
                                                                                                       block_tree.target_index_map,
                                                                                                       block_tree.source_index_map,
                                                                                                       dense_blocks,
                                                                                                       approx_blocks;
                                                                                                       eps=eps,
                                                                                                       svd_recompress=svd_recompress,
                                                                                                       row_block=rb,
                                                                                                       col_block=cb)

    return build_csr_hmatrix(eltype(K), size(K, 1), size(K, 2),
                             target_map, source_map,
                             dense_matrices, dense_block_indices,
                             U_matrices, V_matrices, approx_block_indices;
                             backend=B)
end

# ---------------------------------------------------------------------------
# High-level constructor family: a kernel *function* g (plus point sets or
# custom trees) instead of a hand-written lazy AbstractMatrix. All forms are
# thin wrappers: they build the cluster trees, wrap g in a KernelMatrix and
# forward to the low-level HMatrix(K, X, Y; ...) above — no parallel
# implementation. The low-level constructor stays the full-control entry.
# ---------------------------------------------------------------------------

# square case: targets = sources = pts (do-block friendly: the function is
# the first argument)
function HMatrix(g::Function, pts; dims=1, max_points_per_leaf=32, kwargs...)
    return HMatrix(g, pts, pts; dims=dims,
                   max_points_per_leaf=max_points_per_leaf, kwargs...)
end

function HMatrix(g::Function, pts_t, pts_s; dims=1, max_points_per_leaf=32,
                 backend=nothing, like=nothing, kwargs...)
    Pt = _point_matrix(pts_t)
    Ps = _point_matrix(pts_s)
    # mixed target/source point sets are rejected when the landing backend
    # would be decided from the data (API_DESIGN §4.4 (ii)); an explicit
    # like=/backend= always wins and is checked by _resolve_landing below
    if like === nothing && backend === nothing
        Bt = KernelAbstractions.get_backend(Pt)
        Bs = KernelAbstractions.get_backend(Ps)
        Bt == Bs || error("target and source points live on different " *
                          "backends ($Bt vs $Bs)")
    end
    B = _resolve_landing(like, backend, Pt, Ps)   # device follows the points
    X = ClusterTree(Pt; max_points_per_leaf, dims)  # device input: downloaded once inside
    Y = ClusterTree(Ps; max_points_per_leaf, dims)
    K = KernelMatrix(g, X.coordinates, Y.coordinates; dims)
    return HMatrix(K, X, Y; backend=B, kwargs...)
end

# custom trees: dims defaults to the trees' own dims
function HMatrix(g::Function, X::ClusterTree, Y::ClusterTree; dims=nothing,
                 kwargs...)
    d = something(dims, X.dims)
    (d == X.dims == Y.dims) ||
        error("dims=$d conflicts with the trees' dims ($(X.dims), $(Y.dims))")
    K = KernelMatrix(g, X.coordinates, Y.coordinates; dims=d)
    return HMatrix(K, X, Y; kwargs...)
end

# user-provided lazy K, library-built trees (the data tier follows K inside
# the low-level constructor)
function HMatrix(K::AbstractMatrix, pts_t, pts_s; dims=1,
                 max_points_per_leaf=32, kwargs...)
    X = ClusterTree(_point_matrix(pts_t); max_points_per_leaf, dims)
    Y = ClusterTree(_point_matrix(pts_s); max_points_per_leaf, dims)
    return HMatrix(K, X, Y; kwargs...)
end

# GPU packing: build the three CSR operators with the factors already on the
# device; the near-field data is gathered from the uploaded matrix by a kernel
# near-field CSR index structure for the GPU packing path (host, indices only)
function near_csr_structure(dense_block_indices::Vector{Tuple{Int,Int,Int,Int}},
                            m::Int)
    rowptr = zeros(Int, m + 1)
    for (rs, re, cs, ce) in dense_block_indices
        len = ce - cs + 1
        for i in rs:re
            rowptr[i + 1] += len
        end
    end
    cumsum!(rowptr, rowptr)
    colval = Vector{Int}(undef, rowptr[end])
    cursor = rowptr[1:m]
    for (rs, re, cs, ce) in dense_block_indices
        ncols = ce - cs + 1
        for ii in 1:(re - rs + 1)
            pos = cursor[rs + ii - 1]
            for jj in 1:ncols
                colval[pos + jj] = cs + jj - 1
            end
            cursor[rs + ii - 1] += ncols
        end
    end
    return rowptr, colval
end

# gather kernel: near_data[k] = K[tmap[row], smap[colval[k]]] with the row
# found by binary search over the CSR row offsets
@kernel function fill_near_data!(out, @Const(rowptr), @Const(colval), @Const(Kmat),
                                 @Const(tmap), @Const(smap))
    k = @index(Global, Linear)
    @inbounds if k <= length(out)
        lo, hi = 1, length(rowptr) - 1
        while lo < hi
            mid = (lo + hi + 1) >>> 1
            if rowptr[mid] < k
                lo = mid
            else
                hi = mid - 1
            end
        end
        out[k] = Kmat[tmap[lo], smap[colval[k]]]
    end
end

function build_csr_hmatrix_gpu(K_gpu, backend::KernelAbstractions.Backend,
                               m::Int, n::Int,
                               target_map::Vector{Int}, source_map::Vector{Int},
                               dense_block_indices::Vector{Tuple{Int,Int,Int,Int}},
                               approx_block_indices::Vector{Tuple{Int,Int,Int,Int}},
                               U_factors::Vector{<:AbstractMatrix},
                               V_factors::Vector{<:AbstractMatrix},
                               ranks::Vector{Int}; T::Type=eltype(K_gpu))
    # near-field CSR index structure (host, indices only)
    near_rowptr, near_colval = near_csr_structure(dense_block_indices, m)
    nnz_near = near_rowptr[end]

    # far-field CSR index structure (host); the per-block ranks come from the
    # randomized SVD loop
    L = sum(ranks; init=0)
    napprox = length(approx_block_indices)
    ndense = length(dense_block_indices)

    vcnt = zeros(Int, L)
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r = ranks[bi]                    # the block's truncated rank (columns)
        nc = ce - cs + 1                 # the block's own column count
        for j in 1:r
            vcnt[uoff + j] = nc
        end
        uoff += r
    end
    v_rowptr = [0; cumsum(vcnt)]
    v_colval = Vector{Int}(undef, v_rowptr[end])
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r, ncols = ranks[bi], ce - cs + 1
        for j in 1:r
            pos = v_rowptr[uoff + j]
            for jj in 1:ncols
                v_colval[pos + jj] = cs + jj - 1
            end
        end
        uoff += r
    end
    u_rowptr = zeros(Int, m + 1)
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r = ranks[bi]
        for i in rs:re
            u_rowptr[i + 1] += r
        end
    end
    cumsum!(u_rowptr, u_rowptr)
    u_colval = Vector{Int}(undef, u_rowptr[end])
    # per (block, row) destination starts for the U scatter: a matrix row can
    # receive U contributions from several admissible blocks, so cluster-row
    # offsets alone are not enough — every block-row pair gets its own segment
    nrows_u = sum(re - rs + 1 for (rs, re, cs, ce) in approx_block_indices; init=0)
    useg = Vector{Int}(undef, nrows_u)             # 0-based u_data destinations
    useg_off = zeros(Int, napprox + 1)
    cursor_u = u_rowptr[1:m]
    uoff = 0
    t = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r = ranks[bi]
        useg_off[bi] = t
        for ii in 1:(re - rs + 1)
            i = rs + ii - 1
            pos = cursor_u[i]
            useg[t + ii] = pos
            for jj in 1:r
                u_colval[pos + jj] = uoff + jj
            end
            cursor_u[i] += r
        end
        t += re - rs + 1
        uoff += r
    end
    useg_off[napprox + 1] = t

    # Index arrays are stored as Int32 to halve the index traffic of the
    # memory-bound matvec kernels; the near-field and low-rank nnz grow only
    # linearly with the problem size, so Int32 is safe for any realistic size.
    maxindex = max(near_rowptr[end], v_rowptr[end], u_rowptr[end], m, n, L)
    if maxindex > typemax(Int32)
        error("HMatrix: matrix too large for Int32 CSR indices (nnz = $maxindex).")
    end

    # move the index structures and allocate the data on the device
    move = a -> begin
        KernelAbstractions.get_backend(a) == backend && return a
        dest = KernelAbstractions.zeros(backend, eltype(a), size(a))
        copyto!(dest, a)
        dest
    end
    near_rowptr_d = move(Int32.(near_rowptr))
    near_colval_d = move(Int32.(near_colval))
    tmap_d = move(Int32.(target_map))
    smap_d = move(Int32.(source_map))
    useg_d = move(Int32.(useg))
    near_data = KernelAbstractions.zeros(backend, T, nnz_near)
    v_data = KernelAbstractions.zeros(backend, T, v_rowptr[end])
    u_data = KernelAbstractions.zeros(backend, T, u_rowptr[end])

    # near-field data: device gather from the uploaded matrix
    if nnz_near > 0
        kernel! = fill_near_data!(backend, groupsize[])
        kernel!(near_data, near_rowptr_d, near_colval_d, K_gpu, tmap_d, smap_d;
                ndrange=nnz_near)
    end

    # far-field data: per block, the V factor rows are consecutive in the v CSR
    # (one device-to-device broadcast each), and the U factor rows are scattered
    # to the interleaved u segments (one kernel per block, destinations taken
    # from the block's slice of `useg`)
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r, ncols = ranks[bi], ce - cs + 1
        vr = (v_rowptr[uoff + 1] + 1):(v_rowptr[uoff + r] + ncols)
        reshape(view(v_data, vr), ncols, r) .= V_factors[bi]
        h = re - rs + 1
        seg = view(useg_d, useg_off[bi] + 1:useg_off[bi] + h)
        fill_u_block!(backend, 256)(u_data, seg, U_factors[bi], r; ndrange=h)
        uoff += r
    end

    return HMatrix(m, n,
                   move(Int32.(target_map)), move(Int32.(source_map)),
                   near_rowptr_d, near_colval_d, near_data,
                   move(Int32.(v_rowptr)), move(Int32.(v_colval)), move(v_data),
                   move(Int32.(u_rowptr)), move(Int32.(u_colval)), move(u_data),
                   move(zeros(T, L)), move(zeros(T, n)),
                   ranks, ndense, napprox)
end


"""
    build_csr_hmatrix(T, m, n, target_map, source_map, dense_matrices, dense_block_indices,
                      U_matrices, V_matrices, approx_block_indices) -> HMatrix

Flatten the assembled blocks into the three CSR operators (near field, far-field
`V`, far-field `U`) on the CPU and move them to the requested backend. Column
indices of the near field and of `V` are pre-multiplied with `source_map`, so
the kernels read the input vector `x` in its original ordering directly.
"""
function build_csr_hmatrix(::Type{T}, m::Int, n::Int,
                           target_map::Vector{Int}, source_map::Vector{Int},
                           dense_matrices::Vector{<:Matrix},
                           dense_block_indices::Vector{Tuple{Int,Int,Int,Int}},
                           U_matrices::Vector{<:Matrix}, V_matrices::Vector{<:Matrix},
                           approx_block_indices::Vector{Tuple{Int,Int,Int,Int}};
                           backend=nothing, like=nothing) where {T}
    # landing backend: like= > backend= > CPU() (no data arguments to consult)
    B = _resolve_landing(like, backend)
    move = function (a)
        KernelAbstractions.get_backend(a) == B && return a
        dest = KernelAbstractions.zeros(B, eltype(a), size(a))
        copyto!(dest, a)
        return dest
    end
    # ---- near field: one CSR row per matrix row (cluster order) ----
    # Column indices are stored as *cluster positions*; the input vector is
    # permuted into cluster order once per product (x_buffer), which keeps all
    # x accesses of the kernels sequential.
    near_rowptr = zeros(Int, m + 1)
    for (rs, re, cs, ce) in dense_block_indices
        len = ce - cs + 1
        for i in rs:re
            near_rowptr[i + 1] += len
        end
    end
    cumsum!(near_rowptr, near_rowptr)
    nnz_near = near_rowptr[end]
    near_colval = Vector{Int}(undef, nnz_near)
    near_data = Vector{T}(undef, nnz_near)
    cursor = near_rowptr[1:m]
    for (bi, (rs, re, cs, ce)) in enumerate(dense_block_indices)
        D = dense_matrices[bi]
        ncols = ce - cs + 1
        for ii in 1:(re - rs + 1)
            i = rs + ii - 1
            pos = cursor[i]
            for jj in 1:ncols
                near_colval[pos + jj] = cs + jj - 1
                near_data[pos + jj] = D[ii, jj]
            end
            cursor[i] += ncols
        end
    end

    # ---- far field: L rank rows for V, U segments per matrix row ----
    ranks = [size(U, 2) for U in U_matrices]
    L = sum(ranks; init=0)
    napprox = length(approx_block_indices)
    ndense = length(dense_block_indices)

    vcnt = zeros(Int, L)
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r = size(V_matrices[bi], 1)
        for j in 1:r
            vcnt[uoff + j] = size(V_matrices[bi], 2)
        end
        uoff += r
    end
    v_rowptr = [0; cumsum(vcnt)]
    v_colval = Vector{Int}(undef, v_rowptr[end])
    v_data = Vector{T}(undef, v_rowptr[end])
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        V = V_matrices[bi]
        r, ncols = size(V)
        for j in 1:r
            base = v_rowptr[uoff + j]
            for jj in 1:ncols
                v_colval[base + jj] = cs + jj - 1
                v_data[base + jj] = V[j, jj]
            end
        end
        uoff += r
    end

    u_rowptr = zeros(Int, m + 1)
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        r = size(U_matrices[bi], 2)
        for i in rs:re
            u_rowptr[i + 1] += r
        end
    end
    cumsum!(u_rowptr, u_rowptr)
    u_colval = Vector{Int}(undef, u_rowptr[end])
    u_data = Vector{T}(undef, u_rowptr[end])
    cursor_u = u_rowptr[1:m]
    uoff = 0
    for (bi, (rs, re, cs, ce)) in enumerate(approx_block_indices)
        U = U_matrices[bi]
        nrows, r = size(U)
        for ii in 1:nrows
            i = rs + ii - 1
            pos = cursor_u[i]
            for jj in 1:r
                u_colval[pos + jj] = uoff + jj
                u_data[pos + jj] = U[ii, jj]
            end
            cursor_u[i] += r
        end
        uoff += r
    end

    # Index arrays are stored as Int32 to halve the index traffic of the
    # memory-bound matvec kernels; the near-field and low-rank nnz grow only
    # linearly with the problem size, so Int32 is safe for any realistic size.
    maxindex = max(near_rowptr[end], v_rowptr[end], u_rowptr[end], m, n, L)
    if maxindex > typemax(Int32)
        error("HMatrix: matrix too large for Int32 CSR indices (nnz = $maxindex).")
    end

    return HMatrix(m, n,
                   move(Int32.(target_map)), move(Int32.(source_map)),
                   move(Int32.(near_rowptr)), move(Int32.(near_colval)),
                   move(near_data),
                   move(Int32.(v_rowptr)), move(Int32.(v_colval)),
                   move(v_data),
                   move(Int32.(u_rowptr)), move(Int32.(u_colval)),
                   move(u_data),
                   move(zeros(T, L)), move(zeros(T, n)),
                   ranks, ndense, napprox)
end

"""
    build_matrices(K, target_index_map, source_index_map, dense_blocks,
                   approx_blocks; eps=1e-5, svd_recompress=true,
                   row_block=1, col_block=1)

Extracts near-field blocks and factorizes far-field blocks of `K`, following
the block structure given by the block tree traversal.

The cross approximation always computes in `Float64` (the queries are
converted); the returned factors are converted to `eltype(K)`. Blocks whose
approximation did not converge within the storage-crossover rank are returned
with `converged = false` and must be stored densely by the caller.

# Arguments
- `K::AbstractMatrix`: Original matrix (may be matrix-free; only batched
  `getindex` queries are used).
- `target_index_map`, `source_index_map`: cluster-order to original-DOF maps.
- `dense_blocks`, `approx_blocks`: near/far block pairs from the block tree.
- `eps`: Relative tolerance for the approximation.
- `svd_recompress`: Truncate the ACA factors with a relative SVD (default).
- `row_block`, `col_block`: Group sizes for block pivoting (DOFs per point).

# Returns
- `dense_matrices::Vector{Matrix}`: Near-field blocks extracted from `K`.
- `U_matrices::Vector{Matrix}`, `V_matrices::Vector{Matrix}`: Low-rank factors.
- `dense_block_indices`, `approx_block_indices`: `(rows, cols)` ranges of the
  kept blocks, in cluster-order indices (end-inclusive).
"""
function build_matrices(K::AbstractMatrix, target_index_map::AbstractArray{Int},
                        source_index_map::AbstractArray{Int}, dense_blocks::Vector,
                        approx_blocks::Vector; eps=1e-5, svd_recompress=true,
                        row_block=1, col_block=1)
    T = eltype(K)               # factors are stored in the kernel's precision
    dense_matrices = Matrix[]  # Dense blocks
    U_matrices = Matrix[]  # Low-rank U matrices
    V_matrices = Matrix[]  # Low-rank V matrices
    dense_block_indices = Vector{Tuple{Int,Int,Int,Int}}()
    approx_block_indices = Vector{Tuple{Int,Int,Int,Int}}()

    # Construct dense blocks from dense block indices
    for (a, b) in dense_blocks
        target_ids = view(target_index_map, (a.start_idx):(a.end_idx - 1))
        source_ids = view(source_index_map, (b.start_idx):(b.end_idx - 1))
        dense_block = K[target_ids, source_ids]
        push!(dense_matrices, dense_block)
        push!(dense_block_indices, (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
    end

    # Construct approximated blocks from approx block indices
    for (a, b) in approx_blocks
        target_ids = view(target_index_map, (a.start_idx):(a.end_idx - 1))
        source_ids = view(source_index_map, (b.start_idx):(b.end_idx - 1))
        target_ids_cpu = collect(target_ids)
        source_ids_cpu = collect(source_ids)
        # the ACA runs one order tighter than the SVD recompression; with the
        # relative stopping criteria this keeps the accumulated matvec error at
        # the level of the user-facing tolerance eps
        # the ACA and SVD always compute in Float64 (the queries are
        # converted); factors are stored in the kernel's precision afterwards
        Uc, Vc, converged = ACA_plus(length(target_ids), length(source_ids),
                                     I -> Float64.(K[target_ids_cpu[I], source_ids]),
                                     J -> Float64.(K[target_ids, source_ids_cpu[J]]),
                                     eps / 10.0; row_block=row_block,
                                     col_block=col_block)
        if converged && isa(Uc, Matrix) && isa(Vc, Matrix) && svd_recompress
            Uc, Vc = SVD_recompress(Uc, Vc, eps / 10.0)
        end
        Uc = convert(Matrix{T}, Uc)
        Vc = convert(Matrix{T}, Vc)

        # Check if approximation is beneficial and the ACA converged, otherwise
        # store as dense block
        if converged && size(Uc, 1) * size(Uc, 2) + size(Vc, 1) * size(Vc, 2) <
           length(target_ids) * length(source_ids)
            push!(U_matrices, Uc)
            push!(V_matrices, Vc)
            push!(approx_block_indices,
                  (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
        else
            dense_block = K[target_ids, source_ids]
            push!(dense_matrices, dense_block)
            push!(dense_block_indices,
                  (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
        end
    end

    return dense_matrices, U_matrices, V_matrices, dense_block_indices, approx_block_indices
end


"""
    sparsify_hmatrix(K, X, Y; eta=1.5)

Convert dense blocks of an H-matrix to a sparse matrix representation for efficient storage and computation.

# Arguments
- `K::AbstractMatrix`: Original dense matrix to be sparsified.
- `X::ClusterTree`: Cluster tree defining row partitioning.
- `Y::ClusterTree`: Cluster tree defining column partitioning.
- `eta::Float64=1.5`: Admissibility parameter for block clustering (controls block tree structure).

# Returns
- `SparseMatrixCSC`: Sparse matrix containing all dense blocks from the H-matrix structure.
"""
function sparsify_hmatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, index_map_using_cpu=true)

    block_tree = BlockTree(X, Y; eta=eta, index_map_using_cpu=index_map_using_cpu)
    merge_dense_matrices!(block_tree.root)

    dense_blocks, _ = traverse(block_tree)

    target_index_map = block_tree.target_index_map
    source_index_map = block_tree.source_index_map

    # Initialize row indices, column indices, and non-zero values for the sparse matrix
    I = Int[]  # Row indices
    J = Int[]  # Column indices
    V = eltype(K)[]  # Non-zero values

    # Iterate over dense blocks and directly populate the sparse matrix
    for (a, b) in dense_blocks
        # Get the index ranges for the target and source blocks
        target_ids = view(target_index_map, (a.start_idx):(a.end_idx - 1))
        source_ids = view(source_index_map, (b.start_idx):(b.end_idx - 1))

        # Extract the dense block
        dense_block = K[target_ids, source_ids]

        # Add non-zero elements of the dense block to the sparse matrix indices and values
        for (i, row) in enumerate(Array(target_ids))
            for (j, col) in enumerate(Array(source_ids))
                push!(I, row)
                push!(J, col)
                push!(V, dense_block[i, j])
            end
        end
    end

    # Construct the sparse matrix
    n_rows = size(K, 1)
    n_cols = size(K, 2)
    sparse_matrix = sparse(I, J, V, n_rows, n_cols)

    return sparse_matrix
end


"""
    info(hmatrix::HMatrix) -> Dict

Returns information about the CSR-compressed `HMatrix`: size, block counts,
rank range and the compression ratio.
"""
function info(hmatrix::HMatrix)
    nnz_near = length(hmatrix.near_data)
    nnz_lowrank = length(hmatrix.v_data) + length(hmatrix.u_data)
    original_size = hmatrix.m * hmatrix.n
    compressed_size = nnz_near + nnz_lowrank
    if isempty(hmatrix.ranks)
        min_rank = max_rank = 0
    else
        min_rank = minimum(hmatrix.ranks)
        max_rank = maximum(hmatrix.ranks)
    end
    return Dict("data_type" => eltype(hmatrix.near_data), "size" => (hmatrix.m, hmatrix.n),
                "leaves" => hmatrix.ndense + hmatrix.napprox,
                "admissible_leaves" => hmatrix.napprox,
                "full_leaves" => hmatrix.ndense, "min_rank" => min_rank,
                "max_rank" => max_rank,
                "rank_rows" => length(hmatrix.vx_buffer),
                "near_field_nnz" => nnz_near, "low_rank_nnz" => nnz_lowrank,
                "compression_ratio" => original_size / compressed_size)
end

"""
    hmatrix_blocks(H::HMatrix) -> (dense, lowrank)

Reconstruct the leaf blocks of the assembled `HMatrix` from its CSR index
arrays (works for CPU and GPU-resident matrices; only the index arrays are
downloaded). Returns `(dense, lowrank)` where `dense` is a vector of
`(rows, cols)` index ranges and `lowrank` a vector of `(rows, cols, rank)`.
"""
function hmatrix_blocks(H::HMatrix)
    m = H.m
    near_rowptr = Int.(collect(H.near_rowptr))
    near_colval = Int.(collect(H.near_colval))
    v_rowptr = Int.(collect(H.v_rowptr))
    v_colval = Int.(collect(H.v_colval))
    u_rowptr = Int.(collect(H.u_rowptr))
    u_colval = Int.(collect(H.u_colval))
    ranks = H.ranks

    # ---- near field: a dense block is a maximal contiguous column run of a
    #      row; rectangles are recovered by merging identical runs across
    #      consecutive rows (dense blocks in one row are never column-adjacent
    #      — the block tree merges such siblings) ----
    dense = Tuple{UnitRange{Int},UnitRange{Int}}[]
    open = Dict{Tuple{Int,Int},Int}()      # (c1, c2) => first row of the run
    for i in 1:m
        runs = Tuple{Int,Int}[]
        p = near_rowptr[i] + 1             # near_rowptr holds 0-based offsets
        while p <= near_rowptr[i + 1]
            c1 = c2 = near_colval[p]
            p += 1
            while p <= near_rowptr[i + 1] && near_colval[p] == c2 + 1
                c2 += 1
                p += 1
            end
            push!(runs, (c1, c2))
        end
        closed = Tuple{Int,Int}[]
        for run in keys(open)
            run in runs || push!(closed, run)
        end
        for run in closed
            push!(dense, (open[run]:i - 1, run[1]:run[2]))
            delete!(open, run)
        end
        for run in runs
            get!(open, run, i)
        end
    end
    for (run, r1) in open
        push!(dense, (r1:m, run[1]:run[2]))
    end

    # ---- far field: block bi owns vx buffer positions (cum[bi-1], cum[bi]].
    #      A matrix row's U segment holds the (full) rank range of every block
    #      containing that row — blocks it does not contain leave gaps, so the
    #      segment is walked value by value and each block transition recorded ----
    lowrank = Tuple{UnitRange{Int},UnitRange{Int},Int}[]
    cum = cumsum(ranks)
    rfirst = zeros(Int, length(ranks))
    rlast = zeros(Int, length(ranks))
    for i in 1:m
        p1, p2 = u_rowptr[i], u_rowptr[i + 1]
        p1 == p2 && continue
        bi = 0
        for p in p1 + 1:p2
            nb = searchsortedfirst(cum, u_colval[p])
            if nb != bi
                bi = nb
                rfirst[bi] == 0 && (rfirst[bi] = i)
            end
            rlast[bi] = i
        end
    end
    a = 0
    for bi in eachindex(ranks)
        r = ranks[bi]
        a2 = a + r
        if r > 0 && rfirst[bi] != 0
            cs = v_colval[v_rowptr[a + 1] + 1]     # columns: the run of the
            ce = v_colval[v_rowptr[a2 + 1]]        # block's rank rows in V
            push!(lowrank, (rfirst[bi]:rlast[bi], cs:ce, r))
        end
        a = a2
    end

    # exact invariants: the near CSR holds every dense-block entry once, and
    # the leaves must tile the matrix exactly (the V/U CSRs hold the rank-r
    # factors, not the block entries, so no per-side invariant exists there)
    nnz_dense = sum(length(r) * length(c) for (r, c) in dense; init=0)
    nnz_dense == length(H.near_data) ||
        @warn "dense area mismatch" nnz_dense length(H.near_data)
    area = nnz_dense + sum(length(r) * length(c) for (r, c, _) in lowrank; init=0)
    area == H.m * H.n ||
        @warn "reconstructed blocks do not tile the matrix exactly" area (H.m * H.n)

    return dense, lowrank
end

# extension point: `HMatrixGPUPlotsExt` (ext/) adds a method when Plots.jl is
# loaded
function plot_hmatrix end

# per-block scatter: block-row ii of the U factor goes to its segment in the
# interleaved u CSR data (destinations recorded per block-row pair, so matrix
# rows shared by several blocks never collide)
@kernel function fill_u_block!(u_data, @Const(useg), Uf, @Const(r))
    ii = @index(Global, Linear)
    @inbounds if ii <= length(useg)
        for jj in 1:r
            u_data[useg[ii] + jj] = Uf[ii, jj]
        end
    end
end
