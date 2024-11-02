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
    dense_block_indices::Vector{Tuple{Int, Int, Int, Int}}  # Dense block indices
    approx_block_indices::Vector{Tuple{Int, Int, Int, Int}} # Low-rank block indices
    dense_blocks::Vector{Matrix}            # Dense interaction blocks
    U_matrices::Vector{Matrix}              # Low-rank U matrices
    V_matrices::Vector{Matrix}              # Low-rank V matrices
end

"""
    mutable struct HMatrix

Hierarchical matrix structure optimized for GPU-based operations, designed for better parallel performance.

# Fields
- `K::AbstractMatrix`: The matrix for which the hierarchical block structure is constructed.
- `target_index_map::AbstractArray{Int}`: Index map for target reordering.
- `source_index_map::AbstractArray{Int}`: Index map for source reordering.
- `dense_block_indices::AbstractArray{Int, 2}`: Dense block indices in a matrix format.
- `U_block_indices::AbstractArray{Int, 2}`: Indices for U-matrix blocks.
- `V_block_indices::AbstractArray{Int, 2}`: Indices for V-matrix blocks.
- `dense_blocks::AbstractArray{T, 1}`: Dense interaction blocks.
- `U_matrices::AbstractArray{T, 1}`: Low-rank U matrices.
- `V_matrices::AbstractArray{T, 1}`: Low-rank V matrices.
- `Vx_buffer::AbstractArray{T, 1}`: Buffer to store intermediate V*x results.
"""
mutable struct HMatrix{T<:AbstractFloat}
    K::AbstractMatrix{T}
    target_index_map::AbstractArray{Int}
    source_index_map::AbstractArray{Int}
    dense_block_indices::AbstractArray{Int, 2}
    U_block_indices::AbstractArray{Int, 2}
    V_block_indices::AbstractArray{Int, 2}
    dense_blocks::AbstractArray{T, 1}
    U_matrices::AbstractArray{T, 1}
    V_matrices::AbstractArray{T, 1}
    Vx_buffer::AbstractArray{T, 1}
end

"""
    HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5, flatten=true)

Creates a hierarchical matrix (`HMatrix`) from a given matrix `K` and two cluster trees `X` and `Y`.

# Arguments
- `K::AbstractMatrix`: The matrix to decompose hierarchically.
- `X::ClusterTree`: Cluster tree representing the target partitioning.
- `Y::ClusterTree`: Cluster tree representing the source partitioning.
- `eta::Float64`: Admissibility parameter controlling the low-rank approximation.
- `eps::Float64`: Tolerance level for approximation error.
- `flatten::Bool`: Flag to indicate if the matrix should be optimized for GPU operations.
"""
function HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; eta=1.5, eps=1e-5, flatten=true)
    block_tree = BlockTree(X, Y; eta=eta)
    merge_dense_matrices!(block_tree.root)

    # Traverse the block tree to gather dense and approximated blocks
    dense_blocks, approx_blocks = traverse(block_tree)

    # Build the block matrices and indices
    dense_matrices, U_matrices, V_matrices, dense_block_indices, approx_block_indices = build_matrices(
        K, block_tree.target_index_map, block_tree.source_index_map, dense_blocks, approx_blocks; eps=eps
    )

    # Return CPU-based structure if flatten is false
    if !flatten
        return HMatrixCPU(
            K, X.index_map, Y.index_map, dense_block_indices, approx_block_indices,
            dense_matrices, U_matrices, V_matrices
        )
    end

    # Flatten matrices for GPU optimization, using row-major.
    dense_data = vcat([vec(D') for D in dense_matrices]...)
    U_data = vcat([vec(U') for U in U_matrices]...)
    V_data = vcat([vec(V') for V in V_matrices]...)

    # Construct dense block indices
    dense_indices = hcat([collect(t) for t in dense_block_indices]...)
    dense_offsets = [0; cumsum([prod(size(M)) for M in dense_matrices[1:end-1]])]
    dense_indices = vcat(dense_indices, dense_offsets')
    dense_indices = kernel_array(dense_indices)

    # Construct U block indices
    U_indices = hcat([collect(t) for t in approx_block_indices]...)
    U_offsets = [0; cumsum([prod(size(U)) for U in U_matrices[1:end-1]])]
    U_indices = vcat(U_indices, U_offsets')

    # Construct V block indices and adjust U block indices for V matrix rows
    V_indices = []
    V_offset = 0
    U_offset_row = 0
    for i in eachindex(V_matrices)
        r, n = size(V_matrices[i])
        block_data = zeros(Int64, (3, r))
        (_, _, col_start, col_end) = approx_block_indices[i]
        for j in 1:r
            block_data[1, j] = col_start
            block_data[2, j] = col_end
            block_data[3, j] = V_offset
            V_offset += n
        end
        push!(V_indices, block_data)
        # Update U indices for row limits
        U_indices[3, i] = U_offset_row + 1
        U_indices[4, i] = U_offset_row + r
        U_offset_row += r
    end

    U_indices = kernel_array(U_indices)
    V_indices = kernel_array(hcat(V_indices...))
    Vx_buffer = create_zeros(eltype(K), size(V_indices, 2))

    return HMatrix(
        K, kernel_array(X.index_map), kernel_array(Y.index_map),
        dense_indices, U_indices, V_indices,
        kernel_array(dense_data), kernel_array(U_data), kernel_array(V_data), Vx_buffer
    )
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

        U, V = ACA_plus(length(target_ids), length(source_ids),
                        I -> K[target_ids[I], source_ids],
                        J -> K[target_ids, source_ids[J]], eps / 10.0)
        Uc, Vc = SVD_recompress(U, V, eps)

        # Check if approximation is beneficial, otherwise store as dense block
        if size(Uc, 1) * size(Uc, 2) + size(Vc, 1) * size(Vc, 2) < length(target_ids) * length(source_ids)
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
    info(hmatrix::HMatrix) -> Dict

Returns information about the `HMatrix` object, including matrix size, ranks, 
and the compression ratio.

# Arguments
- `hmatrix::HMatrix`: The hierarchical matrix object to analyze.

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
    min_rank = minimum(ranks)
    max_rank = maximum(ranks)

    # Dense block size statistics
    dense_sizes = [length(block) for block in hmatrix.dense_blocks]
    min_dense_size = minimum(dense_sizes)
    max_dense_size = maximum(dense_sizes)

    # Leaf size statistics (number of elements per leaf)
    U_sizes = [size(U, 1) * size(U, 2) for U in hmatrix.U_matrices]
    V_sizes = [size(V, 1) * size(V, 2) for V in hmatrix.V_matrices]

    # Compression ratio calculation
    original_size = n_rows * n_cols
    compressed_size = sum(U_sizes) + sum(V_sizes)+ sum(dense_sizes)
    compression_ratio = original_size / compressed_size

    # Return dictionary of information
    return Dict("data_type" => data_type, "size" => (n_rows, n_cols),
                "leaves" => num_leaves, "admissible_leaves" => num_approx_leaves,
                "full_leaves" => num_dense_leaves, "min_rank" => min_rank,
                "max_rank" => max_rank, "min_dense_size" => min_dense_size,
                "max_dense_size" => max_dense_size,
                "compression_ratio" => compression_ratio)
end
