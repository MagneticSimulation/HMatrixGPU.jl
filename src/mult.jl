
import Base: *

# Optimized `mul!` function for in-place matrix-vector multiplication
function mul!(result::Vector, hmatrix::HMatrix, x::Vector)
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
    for i in 1:length(hmatrix.approx_matrices)
        (row_start, row_end, col_start, col_end) = hmatrix.approx_block_indices[i]
        (U, V) = hmatrix.approx_matrices[i]
        
        result[row_start:row_end] .+= U * (V * view(x_ordered, col_start:col_end))
    end

    # Reorder result according to target index map
    result[hmatrix.target_index_map] .= result
    return result
end

# Overloaded `*` function for HMatrix and Vector
function *(hmatrix::HMatrix, x::Vector)
    result = zeros(eltype(x), size(hmatrix.K, 1))
    mul!(result, hmatrix, x)
    return result
end
