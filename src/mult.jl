
import Base: *
import LinearAlgebra: mul!

"""
    mul!(result::Vector, hmatrix::HMatrixCPU, x::Vector)

In-place matrix-vector multiplication `result = hmatrix * x` for the CPU
reference structure `HMatrixCPU`. `result` is fully overwritten.
"""
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

"""
    *(hmatrix::HMatrixCPU, x::Vector)

Matrix-vector multiplication for the CPU reference structure `HMatrixCPU`,
returning a freshly allocated result vector.
"""
function *(hmatrix::HMatrixCPU, x::Vector)
    result = zeros(eltype(x), size(hmatrix.K, 1))
    mul!(result, hmatrix, x)
    return result
end

# ---------------------------------------------------------------------------
# Backend kernels for the CSR-compressed HMatrix.
#
# Both kernels are plain thread-per-row dot products: no shared memory, no
# workgroup barriers, no warp-level lane assumptions and no atomics. Every
# kernel explicitly bounds-checks its thread index, so kernel launches are
# insensitive to padding threads added by non-divisible ndranges.
# ---------------------------------------------------------------------------

"""
    permute_to_cluster!(x_buffer, source_map, x)

Gather kernel: `x_buffer[i] = x[source_map[i]]`, permuting the input vector
into cluster order so that all subsequent CSR accesses are sequential.
"""
@kernel function permute_to_cluster!(x_buffer, source_map, x)
    i = @index(Global, Linear)
    @inbounds if i <= length(x_buffer)
        x_buffer[i] = x[source_map[i]]
    end
end

"""
    csr_mul_vec_warp!(out, rowptr, colval, val, x)

Workgroup-per-row CSR sparse matrix-vector product with a fixed workgroup size
of 32 (one warp on CUDA): the 32 threads of a workgroup stride through one CSR
row, and the partial sums are reduced with a tree over 32 local-memory slots.

This provides 32x more resident threads than the thread-per-row form, which is
what hides the memory latency of the dot products on GPUs. Since the ndrange is
`32 * rows` and the workgroup size is 32, launches never have padding threads.
The reduction uses a `for` loop over the tree levels (not a `while` loop with a
hoisted counter) so that no kernel variable is consumed outside the segments
guarded by KernelAbstractions' active-lane check.
"""
@kernel function csr_mul_vec_warp!(out, rowptr, colval, val, x)
    row = @index(Group, Linear)
    t = @index(Local, Linear)
    cache = @localmem eltype(out) 32

    @inbounds if row <= length(out)
        p0 = rowptr[row]
        p1 = rowptr[row + 1]
        s = zero(eltype(out))
        for k in (p0 + t):32:p1
            s += val[k] * x[colval[k]]
        end
        cache[t] = s
    end
    @synchronize

    for s in (16, 8, 4, 2, 1)
        if t <= s
            @inbounds cache[t] += cache[t + s]
        end
        @synchronize
    end

    @inbounds if t == 1 && row <= length(out)
        out[row] = cache[1]
    end
end

"""
    near_u_mul_vec_warp!(y, tmap, near_ptr, near_col, near_val, x, u_ptr, u_col, u_val, vx)

Workgroup-per-row fused near-field and far-field kernel (workgroup size 32).
For each matrix row (cluster order) the 32 threads stride through the near-field
and `U` segments of the row, reduce both partial sums, and thread 1 scatters the
total through `tmap`:

```
y[tmap[row]] = Σ near_val·x[near_col] + Σ u_val·vx[u_col]
```

Since the near-field and `U` row sets of the cluster tree partition the matrix
rows and `tmap` is a permutation, every entry of `y` is written exactly once —
no atomics and no pre-zeroing required.
"""
@kernel function near_u_mul_vec_warp!(y, tmap, near_ptr, near_col, near_val, x,
                                      u_ptr, u_col, u_val, vx)
    row = @index(Group, Linear)
    t = @index(Local, Linear)
    cache_near = @localmem eltype(y) 32
    cache_u = @localmem eltype(y) 32

    @inbounds if row <= length(y)
        p0 = near_ptr[row]
        p1 = near_ptr[row + 1]
        s1 = zero(eltype(y))
        for k in (p0 + t):32:p1
            s1 += near_val[k] * x[near_col[k]]
        end
        cache_near[t] = s1

        q0 = u_ptr[row]
        q1 = u_ptr[row + 1]
        s2 = zero(eltype(y))
        for k in (q0 + t):32:q1
            s2 += u_val[k] * vx[u_col[k]]
        end
        cache_u[t] = s2
    end
    @synchronize

    for s in (16, 8, 4, 2, 1)
        if t <= s
            @inbounds cache_near[t] += cache_near[t + s]
            @inbounds cache_u[t] += cache_u[t + s]
        end
        @synchronize
    end

    @inbounds if t == 1 && row <= length(y)
        y[tmap[row]] = cache_near[1] + cache_u[1]
    end
end

"""
    mul!(result::AbstractArray{T}, hmatrix::HMatrix{T}, x::AbstractArray{T}) where T

Performs a matrix-vector multiplication using the CSR-compressed hierarchical
matrix (`HMatrix`), storing the result in the provided array `result`.
`result` is fully overwritten.

# Arguments
- `result::AbstractArray{T}`: Preallocated array to store the result of the
  matrix-vector multiplication (same backend as `x` and the `HMatrix`).
- `hmatrix::HMatrix{T}`: The hierarchical matrix used for the multiplication.
- `x::AbstractArray{T}`: The input vector to be multiplied.
"""
function mul!(result::AbstractArray{T}, hmatrix::HMatrix{T}, x::AbstractArray{T}) where {T}
    backend = KernelAbstractions.get_backend(x)
    if KernelAbstractions.get_backend(hmatrix.near_data) != backend
        error("HMatrix and input vector live on different backends " *
              "($(KernelAbstractions.get_backend(hmatrix.near_data)) vs $backend); " *
              "rebuild the HMatrix on the same backend as the vectors.")
    end

    # Phase 0: permute x into cluster order (sequential accesses afterwards)
    kernel! = permute_to_cluster!(backend, groupsize[])
    kernel!(hmatrix.x_buffer, hmatrix.source_index_map, x; ndrange=hmatrix.n)

    # Phase 1: far field V * x into the rank-row buffer (one warp per row)
    L = length(hmatrix.vx_buffer)
    if L > 0
        kernel! = csr_mul_vec_warp!(backend, 32)
        kernel!(hmatrix.vx_buffer, hmatrix.v_rowptr, hmatrix.v_colval,
                hmatrix.v_data, hmatrix.x_buffer; ndrange=32 * L)
    end

    # Phase 2: fused near-field + U * vx, scattered to the original ordering
    kernel! = near_u_mul_vec_warp!(backend, 32)
    kernel!(result, hmatrix.target_index_map, hmatrix.near_rowptr,
            hmatrix.near_colval, hmatrix.near_data, hmatrix.x_buffer,
            hmatrix.u_rowptr, hmatrix.u_colval, hmatrix.u_data,
            hmatrix.vx_buffer; ndrange=32 * hmatrix.m)

    return result
end

"""
    *(hmatrix::HMatrix{T}, x::AbstractArray{T}) where T

Overloaded multiplication operator for `HMatrix`. Performs the matrix-vector
multiplication and returns a newly allocated result array on the same backend
as `x`.

# Arguments
- `hmatrix::HMatrix{T}`: The hierarchical matrix used for the multiplication.
- `x::AbstractArray{T}`: The input vector to be multiplied.

# Returns
- `result::AbstractArray{T}`: The result of the matrix-vector multiplication.
"""
function *(hmatrix::HMatrix{T}, x::AbstractArray{T}) where {T}
    # the result follows the backend of the matrix, not the global default
    backend = KernelAbstractions.get_backend(hmatrix.near_data)
    result = KernelAbstractions.zeros(backend, T, hmatrix.m)
    mul!(result, hmatrix, x)
    return result
end
