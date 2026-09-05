using SparseArrays

"""
    mutable struct HMatrixCPU

Hierarchical matrix used for testing and validation.

# Fields
- `K::AbstractMatrix`: The matrix for which the hierarchical block structure is constructed.
- `target_index_map::Vector{Int}`: Mapping of target indices for reordering.
- `source_index_map::Vector{Int}`: Mapping of source indices for reordering.
- `dense_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of dense blocks.
- `approx_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of low-rank approximated blocks.
- `dense_blocks::Vector{Matrix}`: Dense matrix blocks from direct interactions.
- `U_matrices::Vector{Matrix}`: Low-rank approximated U matrices.
- `V_matrices::Vector{Matrix}`: Low-rank approximated V matrices.
"""
mutable struct HMatrixCPU
    K::AbstractMatrix                       # Original matrix for hierarchical decomposition
    target_index_map::Vector{Int}           # Index map for target reordering
    source_index_map::Vector{Int}           # Index map for source reordering
    dense_block_indices::Vector{Tuple{Int,Int,Int,Int}}  # Dense block indices
    approx_block_indices::Vector{Tuple{Int,Int,Int,Int}} # Low-rank block indices
    dense_blocks::Vector{Matrix}            # Dense interaction blocks
    U_matrices::Vector{Matrix}              # Low-rank U matrices
    V_matrices::Vector{Matrix}              # Low-rank V matrices
end

"""
    mutable struct HMatrix

Hierarchical matrix stored as three CSR-compressed operators on the active
backend (CPU, CUDA, AMDGPU, oneAPI or Metal via KernelAbstractions). The
matrix-vector product runs as two thread-per-row kernels without shared
memory, barriers or atomics:

1. `vx = V * x`                — one thread per rank row of the far field,
2. `y = (D + U) * [x; vx]`     — one thread per matrix row (cluster order),
   scattering through `target_index_map` (each output entry is written
   exactly once, so no atomics are needed).

# Fields
- `m::Int`, `n::Int`: dimensions of the full matrix.
- `target_index_map::AbstractVector{Int}`: permutation from cluster-ordered
  rows to original row indices.
- `source_index_map::AbstractVector{Int}`: permutation from cluster-ordered
  columns to original column indices.
- `near_rowptr/near_colval/near_data`: CSR of the near-field (dense) blocks.
  Column indices are already mapped through `source_index_map`.
- `v_rowptr/v_colval/v_data`: CSR of the far-field `V` factors. One row per
  rank row of every admissible block; column indices already mapped.
- `u_rowptr/u_colval/u_data`: CSR of the far-field `U` factors. Rows in
  cluster order; column indices address the `vx` buffer of rank rows.
- `vx_buffer::AbstractVector{T}`: intermediate buffer of length `L` (total
  number of rank rows).
- `x_buffer::AbstractVector{T}`: the input vector permuted into cluster order
  (length `n`), so that all kernel accesses to `x` are sequential.
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

"""
    HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5,
            flatten=true, index_map_using_cpu=true, svd_recompress=true)

Creates a hierarchical matrix (`HMatrix`) from a given matrix `K` and two cluster
trees `X` and `Y`.

# Arguments
- `K::AbstractMatrix`: The matrix to decompose hierarchically (may be matrix-free;
  only batched `getindex` queries are used during assembly).
- `X::ClusterTree`: Cluster tree representing the target partitioning.
- `Y::ClusterTree`: Cluster tree representing the source partitioning.
- `eta::Float64`: Admissibility parameter controlling the low-rank approximation.
- `eps::Float64`: Tolerance level for approximation error.
- `flatten::Bool`: If `true` (default), build the backend-resident CSR structure
  used by the GPU kernels; if `false`, build a CPU reference `HMatrixCPU`.
- `index_map_using_cpu`: Keep the cluster index maps on the CPU during tree
  construction (default; almost always the right choice).
- `svd_recompress`: Recompress the ACA factors with a truncated SVD (default).
- `like::Union{Nothing,AbstractArray}`: when given, all factor arrays are moved
  to the backend where `like` lives (backend follows the data) instead of the
  global default backend.
"""
function HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5,
                 flatten=true, index_map_using_cpu=true, svd_recompress=true,
                 row_block_size=1, col_block_size=1, like=nothing)
    block_tree = BlockTree(X, Y; eta=eta, index_map_using_cpu=index_map_using_cpu)
    merge_dense_matrices!(block_tree.root)

    # Traverse the block tree to gather dense and approximated blocks
    dense_blocks, approx_blocks = traverse(block_tree)

    # Build the block matrices and indices
    dense_matrices, U_matrices, V_matrices, dense_block_indices, approx_block_indices = build_matrices(K,
                                                                                                       block_tree.target_index_map,
                                                                                                       block_tree.source_index_map,
                                                                                                       dense_blocks,
                                                                                                       approx_blocks;
                                                                                                       eps=eps,
                                                                                                       svd_recompress=svd_recompress,
                                                                                                       row_block=row_block_size,
                                                                                                       col_block=col_block_size)

    # Return CPU-based structure if flatten is false
    if !flatten
        return HMatrixCPU(K, X.index_map, Y.index_map, dense_block_indices,
                          approx_block_indices, dense_matrices, U_matrices, V_matrices)
    end

    target_map = collect(Int, block_tree.target_index_map)
    source_map = collect(Int, block_tree.source_index_map)
    return build_csr_hmatrix(eltype(K), size(K, 1), size(K, 2),
                             target_map, source_map,
                             dense_matrices, dense_block_indices,
                             U_matrices, V_matrices, approx_block_indices;
                             like=like)
end

"""
    build_csr_hmatrix(T, m, n, target_map, source_map, dense_matrices, dense_block_indices,
                      U_matrices, V_matrices, approx_block_indices) -> HMatrix

Flatten the assembled blocks into the three CSR operators (near field, far-field
`V`, far-field `U`) on the CPU and move them to the active backend. Column
indices of the near field and of `V` are pre-multiplied with `source_map`, so
the kernels read the input vector `x` in its original ordering directly.
"""
function build_csr_hmatrix(::Type{T}, m::Int, n::Int,
                           target_map::Vector{Int}, source_map::Vector{Int},
                           dense_matrices::Vector{Matrix},
                           dense_block_indices::Vector{Tuple{Int,Int,Int,Int}},
                           U_matrices::Vector{Matrix}, V_matrices::Vector{Matrix},
                           approx_block_indices::Vector{Tuple{Int,Int,Int,Int}};
                           like=nothing) where {T}
    # move an assembled host array to the landing backend: the backend of
    # `like` when given (backend follows the data), the global default_backend
    # otherwise
    move = function (a)
        like === nothing ? kernel_array(a) : to_backend(like, a)
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
    build_matrices(K::AbstractMatrix, target_index_map::Vector{Int}, source_index_map::Vector{Int},
                   dense_blocks::Vector, approx_blocks::Vector; eps=1e-5)

Builds dense and low-rank approximated matrices from `K` based on block structures.

# Arguments
- `K::AbstractMatrix`: Original matrix from which the blocks are extracted.
- `target_index_map::Vector{Int}`: Index map for target reordering.
- `source_index_map::Vector{Int}`: Index map for source reordering.
- `dense_blocks::Vector`: List of dense block pairs from direct interactions.
- `approx_blocks::Vector`: List of low-rank approximable block pairs.
- `eps::Float64`: Tolerance level for approximation error.

# Returns
- `dense_matrices::Vector{Matrix}`: Dense blocks extracted from `K`.
- `approx_matrices::Vector{Tuple{Matrix, Matrix}}`: Low-rank (U, V) approximation matrices for blocks.
- `dense_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of dense blocks.
- `approx_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of approximated blocks.
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
    info(hmatrix::HMatrixCPU) -> Dict

Returns information about the `HMatrixCPU` object, including matrix size, ranks,
and the compression ratio.

# Arguments
- `hmatrix::HMatrixCPU`: The hierarchical matrix object to analyze.

# Output
Returns a dictionary with information about the hierarchical matrix.
"""
function info(hmatrix::HMatrixCPU)
    # Basic matrix information
    n_rows, n_cols = size(hmatrix.K)
    data_type = eltype(hmatrix.K)

    # Tree statistics
    num_dense_leaves = length(hmatrix.dense_blocks)
    num_approx_leaves = length(hmatrix.U_matrices)
    num_leaves = num_dense_leaves + num_approx_leaves

    # Sparse block rank statistics
    ranks = [size(U, 2) for U in hmatrix.U_matrices]
    if length(ranks) == 0
        min_rank = 0
        max_rank = 0
    else
        min_rank = minimum(ranks)
        max_rank = maximum(ranks)
    end

    # Dense block size statistics
    dense_sizes = [length(block) for block in hmatrix.dense_blocks]
    min_dense_size = minimum(dense_sizes)
    max_dense_size = maximum(dense_sizes)

    # Leaf size statistics (number of elements per leaf)
    U_sizes = [size(U, 1) * size(U, 2) for U in hmatrix.U_matrices]
    V_sizes = [size(V, 1) * size(V, 2) for V in hmatrix.V_matrices]

    # Compression ratio calculation
    original_size = n_rows * n_cols
    compressed_size = sum(U_sizes) + sum(V_sizes) + sum(dense_sizes)
    compression_ratio = original_size / compressed_size

    # Return dictionary of information
    return Dict("data_type" => data_type, "size" => (n_rows, n_cols),
                "leaves" => num_leaves, "admissible_leaves" => num_approx_leaves,
                "full_leaves" => num_dense_leaves, "min_rank" => min_rank,
                "max_rank" => max_rank, "min_dense_size" => min_dense_size,
                "max_dense_size" => max_dense_size,
                "compression_ratio" => compression_ratio)
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
