# A simplified version of the python code at https://tbenthompson.com/book/tdes/hmatrix.html.

"""
    argmax_not_in_list(arr, disallowed)

Finds the index of the maximum element in `arr` that is not in `disallowed`. 
If all indices are disallowed, returns -1.
"""
function argmax_not_in_list(arr, disallowed)
    sorted_indices = sortperm(arr; rev=true)  # Sort indices by value in descending order
    for idx in sorted_indices
        if !(idx in disallowed)
            return idx  # Return the first index not in disallowed list
        end
    end
    return -1  # No valid index found
end

"""
    ACA_plus(n_rows, n_cols, calc_rows, calc_cols, eps; max_iter=n_rows)

Adaptive Cross Approximation (ACA) with partial pivoting to approximate a matrix with a given error tolerance.

# Arguments
- `n_rows::Int`: Number of rows in the matrix.
- `n_cols::Int`: Number of columns in the matrix.
- `calc_rows::Function`: Function to calculate selected rows of the original matrix.
- `calc_cols::Function`: Function to calculate selected columns of the original matrix.
- `eps::Float64`: Tolerance level for approximation.
- `max_iter::Int`: Maximum number of iterations (default is `n_rows`).

# Returns
- `U_ACA::Matrix`: Matrix containing the left vectors of the approximation.
- `V_ACA::Matrix`: Matrix containing the right vectors of the approximation.
"""
function ACA_plus(n_rows, n_cols, calc_rows, calc_cols, eps; max_iter=n_rows)
    left_vectors = []  # Store left vectors of the approximation
    right_vectors = []  # Store right vectors of the approximation
    used_row_pivots = Int[]  # Track previously used row pivots
    used_col_pivots = Int[]  # Track previously used column pivots

    # Calculate residual rows for row indices I
    function calc_residual_rows(I)
        residual = calc_rows(I)
        for i in eachindex(left_vectors)
            residual .-= left_vectors[i][I] * right_vectors[i]
        end
        return residual
    end

    # Calculate residual columns for column indices J
    function calc_residual_cols(J)
        residual = calc_cols(J)
        for i in eachindex(right_vectors)
            residual .-= left_vectors[i] * right_vectors[i][J]
        end
        return residual
    end

    max_iter = min(n_rows, n_cols, max_iter)

    residual_row = zeros(n_cols)  # Placeholder for residual row calculation
    residual_col = zeros(n_rows)  # Placeholder for residual column calculation

    for k in 1:max_iter
        abs_row_residual = abs.(residual_row)
        abs_col_residual = abs.(residual_col)

        if k == 1
            row_pivot, col_pivot = 1, 1
        else
            col_pivot = argmax_not_in_list(abs_row_residual, used_col_pivots)
            row_pivot = argmax_not_in_list(abs_col_residual, used_row_pivots)
        end

        col_pivot_val = abs_row_residual[col_pivot]
        row_pivot_val = abs_col_residual[row_pivot]

        if row_pivot_val > col_pivot_val
            residual_row .= calc_residual_rows(row_pivot)
            col_pivot = argmax_not_in_list(abs.(residual_row), used_col_pivots)
            residual_col .= calc_residual_cols(col_pivot)
        else
            residual_col .= calc_residual_cols(col_pivot)
            row_pivot = argmax_not_in_list(abs.(residual_col), used_row_pivots)
            residual_row .= calc_residual_rows(row_pivot)
        end

        push!(used_row_pivots, row_pivot)
        push!(used_col_pivots, col_pivot)

        push!(right_vectors, residual_row / residual_row[col_pivot])
        push!(left_vectors, copy(residual_col))

        step_size = sqrt(sum(left_vectors[end] .^ 2) * sum(right_vectors[end] .^ 2))

        if step_size < eps
            break
        end
    end

    U_ACA = hcat(left_vectors...)
    V_ACA = Matrix(hcat(right_vectors...)')

    return U_ACA, V_ACA
end

"""
    SVD_recompress(U::Matrix, V::Matrix, eps::Float64)

Performs SVD-based recompression of a matrix represented by the product `U * V'`, 
truncating singular values based on a given tolerance `eps`.

# Arguments
- `U::Matrix`: The left matrix in the decomposition.
- `V::Matrix`: The right matrix in the decomposition.
- `eps::Float64`: Tolerance level for truncating small singular values.

# Returns
- `U_SVD::Matrix`: Truncated left singular matrix after recompression.
- `V_SVD::Matrix`: Truncated right singular matrix after recompression.
"""
function SVD_recompress(U::Matrix, V::Matrix, eps::Float64)
    # Perform QR decomposition on U and V'
    QU, RU = qr(U)
    QV, RV = qr(V')

    # Perform SVD on the product of upper triangular matrices from QR decompositions
    W, SIG, Z = svd(RU * RV')

    # Compute cumulative Frobenius norms of the singular values in reverse order
    frobenius_norms = reverse(sqrt.(cumsum(reverse(SIG .^ 2))))

    # Find the truncation rank where the cumulative Frobenius norm falls below eps
    rank_trunc = findfirst(x -> x < eps, frobenius_norms)
    rank_trunc = rank_trunc === nothing ? size(U, 2) : rank_trunc - 1  # Use the full rank if no truncation is needed

    # Compute the recompressed matrices U_SVD and V_SVD up to the truncated rank
    U_SVD = QU * (W[:, 1:rank_trunc] .* SIG[1:rank_trunc]')
    V_SVD = Z[:, 1:rank_trunc]' * QV'

    return U_SVD, V_SVD
end
