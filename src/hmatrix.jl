"""
    mutable struct HMatrix

Hierarchical matrix structure used to store dense and low-rank approximated blocks of a matrix.

# Fields
- `K::AbstractMatrix`: The matrix for which the hierarchical block structure is constructed.
- `target_index_map::Vector{Int}`: Mapping of target indices for reordering.
- `source_index_map::Vector{Int}`: Mapping of source indices for reordering.
- `dense_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of dense blocks.
- `approx_block_indices::Vector{Tuple{Int, Int, Int, Int}}`: Indices of low-rank approximated blocks.
- `dense_blocks::Vector{Matrix}`: Dense matrix blocks from direct interactions.
- `approx_matrices::Vector{Tuple{Matrix, Matrix}}`: Low-rank (U, V) approximated matrices.
"""
mutable struct HMatrix
    K::AbstractMatrix                       # Original matrix for hierarchical decomposition
    target_index_map::Vector{Int}           # Index map for target reordering
    source_index_map::Vector{Int}           # Index map for source reordering
    dense_block_indices::Vector{Tuple{Int,Int,Int,Int}}  # Dense block indices
    approx_block_indices::Vector{Tuple{Int,Int,Int,Int}} # Low-rank block indices
    dense_blocks::Vector{Matrix}            # Dense interaction blocks
    approx_matrices::Vector{Tuple{Matrix,Matrix}}  # Low-rank (U, V) matrices
end

"""
    HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5)

Creates a hierarchical matrix (`HMatrix`) from a given matrix `K` and two cluster trees `X` and `Y`.

# Arguments
- `K::AbstractMatrix`: The matrix to decompose hierarchically.
- `X::ClusterTree`: Cluster tree representing the target partitioning.
- `Y::ClusterTree`: Cluster tree representing the source partitioning.
- `eta::Float64`: Admissibility parameter controlling the low-rank approximation.
- `eps::Float64`: Tolerance level for approximation error.

# Returns
- `HMatrix`: The constructed hierarchical matrix with dense and approximated blocks.
"""
function HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5)
    block_tree = BlockTree(X, Y; eta=eta)
    merge_dense_matrices!(block_tree.root)

    dense_blocks, approx_blocks = traverse(block_tree)

    dense_matrices, approx_matrices, dense_block_indices, approx_block_indices = build_matrices(K,
                                                                                                block_tree.target_index_map,
                                                                                                block_tree.source_index_map,
                                                                                                dense_blocks,
                                                                                                approx_blocks;
                                                                                                eps=eps)

    return HMatrix(K, X.index_map, Y.index_map, dense_block_indices, approx_block_indices,
                   dense_matrices, approx_matrices)
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
function build_matrices(K::AbstractMatrix, target_index_map::Vector{Int},
                        source_index_map::Vector{Int}, dense_blocks::Vector,
                        approx_blocks::Vector; eps=1e-5)
    dense_matrices = Matrix[]  # Dense blocks
    approx_matrices = Vector{Tuple{Matrix,Matrix}}()  # Low-rank (U, V) approximation pairs
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

        U, V = ACA_plus(length(target_ids), length(source_ids),
                        I -> K[target_ids[I], source_ids],
                        J -> K[target_ids, source_ids[J]], eps / 10.0)
        Uc, Vc = SVD_recompress(U, V, eps)

        # Check if approximation is beneficial, otherwise store as dense block
        if size(Uc, 1) * size(Uc, 2) + size(Vc, 1) * size(Vc, 2) <
           length(target_ids) * length(source_ids)
            push!(approx_matrices, (Uc, Vc))
            push!(approx_block_indices,
                  (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
        else
            dense_block = K[target_ids, source_ids]
            push!(dense_matrices, dense_block)
            push!(dense_block_indices,
                  (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
        end
    end

    return dense_matrices, approx_matrices, dense_block_indices, approx_block_indices
end

"""
    info(hmatrix::HMatrix) -> Dict

Returns information about the `HMatrix` object, including matrix size, ranks, 
and the compression ratio.

# Arguments
- `hmatrix::HMatrix`: The hierarchical matrix object to analyze.

# Output
Returns a dictionary with information about the hierarchical matrix.
"""
function info(hmatrix::HMatrix)
    # Basic matrix information
    n_rows, n_cols = size(hmatrix.K)
    data_type = eltype(hmatrix.K)

    # Tree statistics
    num_dense_leaves = length(hmatrix.dense_blocks)
    num_approx_leaves = length(hmatrix.approx_matrices)
    num_leaves = num_dense_leaves + num_approx_leaves

    # Sparse block rank statistics
    ranks = [size(U, 2) for (U, _) in hmatrix.approx_matrices]
    min_rank = minimum(ranks)
    max_rank = maximum(ranks)

    # Dense block size statistics
    dense_sizes = [length(block) for block in hmatrix.dense_blocks]
    min_dense_size = minimum(dense_sizes)
    max_dense_size = maximum(dense_sizes)

    # Leaf size statistics (number of elements per leaf)
    approx_sizes = [size(U, 1) * size(U, 2) + size(V, 1) * size(V, 2)
                    for (U, V) in hmatrix.approx_matrices]

    # Compression ratio calculation
    original_size = n_rows * n_cols
    compressed_size = sum(approx_sizes) + sum(dense_sizes)
    compression_ratio = original_size / compressed_size

    # Return dictionary of information
    return Dict("data_type" => data_type, "size" => (n_rows, n_cols),
                "leaves" => num_leaves, "admissible_leaves" => num_approx_leaves,
                "full_leaves" => num_dense_leaves, "min_rank" => min_rank,
                "max_rank" => max_rank, "min_dense_size" => min_dense_size,
                "max_dense_size" => max_dense_size,
                "compression_ratio" => compression_ratio)
end
