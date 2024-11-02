
import Base: *

# Optimized `mul!` function for in-place matrix-vector multiplication
function mul!(result::Vector, hmatrix::HMatrixCPU, x::Vector)
    fill!(result, 0)

    # Reorder x based on source index mapping to avoid repeated indexing in loops
    x_ordered = x[hmatrix.source_index_map]
    
    # Process dense blocks
    for i in 1:length(hmatrix.dense_blocks)
        (row_start, row_end, col_start, col_end) = hmatrix.dense_block_indices[i]
        dense_block = hmatrix.dense_blocks[i]
        
        result[row_start:row_end] .+= dense_block * view(x_ordered, col_start:col_end)
    end

    # Process approximate (low-rank) blocks
    for i in 1:length(hmatrix.U_matrices)
        (row_start, row_end, col_start, col_end) = hmatrix.approx_block_indices[i]
        U = hmatrix.U_matrices[i]
        V = hmatrix.V_matrices[i]
        
        result[row_start:row_end] .+= U * (V * view(x_ordered, col_start:col_end))
    end

    # Reorder result according to target index map
    result[hmatrix.target_index_map] .= result
    return result
end

# Overloaded `*` function for HMatrix and Vector
function *(hmatrix::HMatrixCPU, x::Vector)
    result = zeros(eltype(x), size(hmatrix.K, 1))
    mul!(result, hmatrix, x)
    return result
end


@kernel function V_mult_vec_kernel!(result, @Const(V_data), @Const(block_indices), @Const(source_map), @Const(input_vec))
    I = @index(Global)

    @inbounds start_col = block_indices[1, I]
    @inbounds end_col = block_indices[2, I]
    @inbounds offset = block_indices[3, I]

    acc_sum = zero(eltype(result))

    i = offset
    for j in start_col:end_col
        i += 1
        @inbounds acc_sum += V_data[i] * input_vec[source_map[j]]
    end

    @inbounds result[I] = acc_sum
end

@kernel function DU_mult_vec_kernel!(result, @Const(D_blocks), @Const(U_blocks), @Const(dense_block_indices), @Const(U_block_indices), @Const(target_map), @Const(source_map), @Const(Vx_intermediate), @Const(input_vec))
    I = @index(Global)

    acc_sum = zero(eltype(result))

    # Dense block multiplication
    for block in 1:size(dense_block_indices, 2)
        @inbounds start_row = dense_block_indices[1, block]
        @inbounds end_row = dense_block_indices[2, block]
        if I < start_row || I > end_row
            continue
        end

        @inbounds start_col = dense_block_indices[3, block]
        @inbounds end_col = dense_block_indices[4, block]
        @inbounds data_offset = dense_block_indices[5, block]

        i = data_offset + (I - start_row) * (end_col - start_col + 1)
        for j in start_col:end_col
            i += 1
            @inbounds acc_sum += D_blocks[i] * input_vec[source_map[j]]
        end
    end

    # U block multiplication with Vx
    for block in 1:size(U_block_indices, 2)
        @inbounds start_row = U_block_indices[1, block]
        @inbounds end_row = U_block_indices[2, block]
        if I < start_row || I > end_row
            continue
        end

        @inbounds start_col = U_block_indices[3, block]
        @inbounds end_col = U_block_indices[4, block]
        @inbounds data_offset = U_block_indices[5, block]

        i = data_offset + (I - start_row) * (end_col - start_col + 1)
        for j in start_col:end_col
            i += 1
            @inbounds acc_sum += U_blocks[i] * Vx_intermediate[j]
        end
    end

    @inbounds result[target_map[I]] = acc_sum
end



"""
    mul!(result::AbstractArray{T}, hmatrix::HMatrix{T}, x::AbstractArray{T}) where T

Performs a matrix-vector multiplication using a hierarchical matrix (`HMatrix`) structure, storing the result in the provided array `result`.

# Arguments
- `result::AbstractArray{T}`: Preallocated array to store the result of the matrix-vector multiplication.
- `hmatrix::HMatrix{T}`: The hierarchical matrix used for the multiplication.
- `x::AbstractArray{T}`: The input vector to be multiplied.

"""
function mul!(result::AbstractArray{T}, hmatrix::HMatrix{T}, x::AbstractArray{T}) where T
    # Get the size of the original matrix
    m, n = size(hmatrix.K)

    # Select the backend for kernel execution based on the input array type
    backend = KernelAbstractions.get_backend(x)

    # Launch kernel for multiplying V matrices with the input vector
    kernel! = V_mult_vec_kernel!(backend)
    kernel!(
        hmatrix.Vx_buffer,                 # Output buffer for intermediate V * x result
        hmatrix.V_matrices,                # V matrices data
        hmatrix.V_block_indices,           # Indices of V blocks
        hmatrix.source_index_map,          # Source index mapping
        x;                                 # Input vector
        ndrange = length(hmatrix.Vx_buffer) # Set the number of threads for parallel execution
    )

    # Launch kernel for dense and U-matrix multiplications
    kernel! = DU_mult_vec_kernel!(backend)
    kernel!(
        result,                            # Output array for final result
        hmatrix.dense_blocks,              # Dense blocks data
        hmatrix.U_matrices,                # U matrices data
        hmatrix.dense_block_indices,       # Indices of dense blocks
        hmatrix.U_block_indices,           # Indices of U blocks
        hmatrix.target_index_map,          # Target index mapping
        hmatrix.source_index_map,          # Source index mapping
        hmatrix.Vx_buffer,                 # Buffer containing V * x result
        x;                                 # Input vector
        ndrange = m                        # Set the number of threads for parallel execution
    )

    return result
end

"""
    *(hmatrix::HMatrix{T}, x::AbstractArray{T}) where T

Overloaded multiplication operator for `HMatrix`. Performs the matrix-vector multiplication and returns a new result array.

# Arguments
- `hmatrix::HMatrix{T}`: The hierarchical matrix used for the multiplication.
- `x::AbstractArray{T}`: The input vector to be multiplied.

# Returns
- `result::AbstractArray{T}`: The result of the matrix-vector multiplication.
"""
function *(hmatrix::HMatrix{T}, x::AbstractArray{T}) where T
    # Create a zero-initialized result array
    result = create_zeros(T, size(hmatrix.K, 1))
    # Call the in-place multiplication function
    mul!(result, hmatrix, x)
    return result
end
