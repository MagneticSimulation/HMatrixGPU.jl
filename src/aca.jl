using Random
# A simplified version of the python code at https://tbenthompson.com/book/tdes/hmatrix.html.
# The pivoting strategy follows Grasedyck 2005 ("Adaptive Recompression of
# H-Matrices for BEM"), Construction 2.4: ACA+ with look-ahead pivoting,
# generalized to block rows/columns (vector-valued problems, e.g. the 3x3
# demagnetization tensor, pivot one group of DOFs at a time).

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
    argmax_skip(v, used, dead, tol)

Index of the largest entry of `v` among indices that are neither in `used`
nor marked `dead`, provided its magnitude exceeds `tol`; -1 if none.
"""
function argmax_skip(v, used, dead, tol)
    best = -1
    bestv = tol
    for i in eachindex(v)
        (dead[i] || (i in used)) && continue
        if abs(v[i]) > bestv
            bestv = abs(v[i])
            best = i
        end
    end
    return best
end

"""
    ACA_plus(n_rows, n_cols, calc_rows, calc_cols, eps; max_iter=n_rows, max_rank=0,
             pivot_tol=1e-13, row_block=1, col_block=1)

Adaptive Cross Approximation with the ACA+ look-ahead pivoting strategy
(Grasedyck 2005, Construction 2.4; based on Bebendorf 2000 and
Bebendorf & Rjasanow 2003) and relative error control, generalized to
*block* rows/columns.

The pivoting keeps one reference row group and one reference column group of
residuals. Each step picks the row-group pivot from the reference *column*
group residual and the column-group pivot from the reference *row* group
residual (look-ahead), completes the cross through the larger of the two
candidates, and refreshes the reference when a pivot coincides with it. For
`row_block = col_block = d > 1` (vector-valued problems such as the
demagnetization tensor with three DOFs per cell) the pivot block is `d x d`
and one step captures all `d` components of a cell together — plain scalar
pivoting starves the weak components of anisotropic kernels.

The stopping criterion is *relative*: the iteration stops once the Frobenius
norm of a rank-`block` term drops below `eps` times the Frobenius norm of the
block (estimated from a spread sample of row/column groups), so the achieved
accuracy is independent of the scale of the block.

Following standard practice in H-matrix libraries (Börm–Grasedyck–Hackbusch;
AHMED), the iteration is also bounded by a *storage crossover* rank: once the
group rank reaches `max_rank` (by default `ceil(m*n/(m+n))` in groups, the
rank at which the low-rank factors cost as much as the dense block), the
iteration stops. Blocks whose ε-rank exceeds the crossover are cheaper to
store densely, and the caller falls back to a dense block through its regular
storage comparison.

# Arguments
- `n_rows::Int`, `n_cols::Int`: Number of rows/columns (multiples of the block sizes).
- `calc_rows::Function`: Returns the block rows `dr x n_cols` for a row range.
- `calc_cols::Function`: Returns the block columns `n_rows x dc` for a column range.
- `eps::Float64`: Relative tolerance level for approximation.
- `max_iter::Int`: Maximum number of iterations (default is the group count).
- `max_rank::Int`: Maximum group rank before aborting; `0` selects the storage
  crossover `ceil(m*n/(m+n))` in groups.
- `pivot_tol::Float64`: Pivot blocks with Frobenius norm below `pivot_tol`
  times the estimated block magnitude are treated as zero (protects against
  degenerate blocks with rows that are zero to machine precision).
- `row_block::Int`, `col_block::Int`: Group sizes (DOFs per point on each side).

# Returns
- `U_ACA::Matrix`: `n_rows x (k*col_block)` left factor (residual column groups).
- `V_ACA::Matrix`: `(k*row_block) x n_cols` right factor (residual row groups,
  normalized by the pivot blocks).
- `converged::Bool`: Whether the approximation met the relative stopping
  criterion, or a sampled relative residual check confirmed the block is
  captured to within `10*eps`. When `false`, the caller should store the block
  densely.
"""
function ACA_plus(n_rows, n_cols, calc_rows, calc_cols, eps; max_iter=n_rows,
                  max_rank=0, pivot_tol=1e-13, row_block::Int=1, col_block::Int=1)
    # deterministic per-block RNG: two constructions of the same matrix make
    # identical pivoting decisions (the reference selection is randomized to
    # avoid systematic pivoting pathologies, but reproducibly so)
    rng = Random.MersenneTwister(hash((n_rows, n_cols, row_block, col_block)))
    dr = max(1, row_block)
    dc = max(1, col_block)
    n_rows % dr == 0 || error("n_rows must be divisible by row_block")
    n_cols % dc == 0 || error("n_cols must be divisible by col_block")
    mg = n_rows ÷ dr                # row groups
    ng = n_cols ÷ dc                # column groups
    if max_rank <= 0
        max_rank = ceil(Int, mg * ng / (mg + ng))   # storage crossover, groups
    end
    max_rank = min(max_rank, mg, ng, max_iter)

    # magnitude and Frobenius norm of the block estimated from a spread sample
    # of row/column groups; the magnitude protects the pivoting against groups
    # that are zero to machine precision, the Frobenius estimate makes the
    # stopping criterion independent of an unrepresentative first pivot
    scale = 0.0
    row_fro2 = 0.0
    col_fro2 = 0.0
    row_samples = mg <= 8 ? collect(1:mg) : unique(round.(Int, LinRange(1, mg, 8)))
    col_samples = ng <= 8 ? collect(1:ng) : unique(round.(Int, LinRange(1, ng, 8)))
    for g in row_samples
        v = calc_rows(dr*(g - 1) + 1 : dr*g)          # dr x n_cols
        scale = max(scale, maximum(abs.(v)))
        row_fro2 += sum(abs2, v)
    end
    for g in col_samples
        v = calc_cols(dc*(g - 1) + 1 : dc*g)          # n_rows x dc
        scale = max(scale, maximum(abs.(v)))
        col_fro2 += sum(abs2, v)
    end
    if iszero(scale)
        # numerically zero block: a zero factorization is exact
        return zeros(n_rows, dc), zeros(dr, n_cols), true
    end
    block_fro2 = max(mg * row_fro2 / length(row_samples),
                     ng * col_fro2 / length(col_samples))

    U_blocks = Matrix{Float64}[]   # n_rows x dc residual column groups
    V_blocks = Matrix{Float64}[]   # dr x n_cols residual row groups (normalized)
    used_rows = falses(mg)
    used_cols = falses(ng)
    dead_rows = falses(mg)
    dead_cols = falses(ng)

    # residual helpers (subtract the accumulated rank-block terms)
    calc_residual_rows = function (g)     # dr x n_cols for row group g
        res = calc_rows(dr*(g - 1) + 1 : dr*g)
        for t in eachindex(U_blocks)
            res .-= U_blocks[t][dr*(g - 1) + 1 : dr*g, :] * V_blocks[t]
        end
        return res
    end
    calc_residual_cols = function (g)     # n_rows x dc for column group g
        res = calc_cols(dc*(g - 1) + 1 : dc*g)
        for t in eachindex(V_blocks)
            res .-= U_blocks[t] * V_blocks[t][:, dc*(g - 1) + 1 : dc*g]
        end
        return res
    end

    # best row group of a reference column-group residual (n_rows x dc)
    best_row_group = function (refc)
        best = -1
        bestv = (pivot_tol * scale)^2
        for g in 1:mg
            (used_rows[g] || dead_rows[g]) && continue
            v = sum(abs2, refc[dr*(g - 1) + 1 : dr*g, :])
            v > bestv && (bestv = v; best = g)
        end
        return best, bestv
    end
    # best column group of a reference row-group residual (dr x n_cols)
    best_col_group = function (refw)
        best = -1
        bestv = (pivot_tol * scale)^2
        for g in 1:ng
            (used_cols[g] || dead_cols[g]) && continue
            v = sum(abs2, refw[:, dc*(g - 1) + 1 : dc*g])
            v > bestv && (bestv = v; best = g)
        end
        return best, bestv
    end

    # verify with fresh sampled groups that the current residual is negligible
    # relative to the block magnitude
    residual_negligible = function ()
        for _ in 1:8
            if sum(abs2, calc_residual_rows(rand(rng, 1:mg))) > (pivot_tol * scale)^2
                return false
            end
        end
        for _ in 1:8
            if sum(abs2, calc_residual_cols(rand(rng, 1:ng))) > (pivot_tol * scale)^2
                return false
            end
        end
        return true
    end

    # estimate the relative residual norm from fresh sampled row/column groups:
    # the storage decision for blocks that did not meet the stopping criterion
    sampled_rel_residual = function ()
        r2 = 0.0
        b2 = 0.0
        for _ in 1:8
            g = rand(rng, 1:mg)
            r2 += sum(abs2, calc_residual_rows(g))
            b2 += sum(abs2, calc_rows(dr*(g - 1) + 1 : dr*g))
        end
        for _ in 1:8
            g = rand(rng, 1:ng)
            r2 += sum(abs2, calc_residual_cols(g))
            b2 += sum(abs2, calc_cols(dc*(g - 1) + 1 : dc*g))
        end
        return sqrt(r2 / max(b2, 1e-300))
    end

    # ACA+ setup: one random reference row group r and column group c
    r = rand(rng, 1:mg)
    c = rand(rng, 1:ng)
    ref_row = calc_residual_rows(r)     # dr x n_cols
    ref_col = calc_residual_cols(c)     # n_rows x dc
    scale = max(scale, maximum(abs.(ref_row)), maximum(abs.(ref_col)))

    l = 0                    # accepted rank-block terms
    ref_failures = 0         # consecutive exhausted-reference events
    converged = false        # whether the relative stopping criterion was met
    while l < max_rank
        # look-ahead candidate pivots from the reference residuals
        i, vi = best_row_group(ref_col)
        j, vj = best_col_group(ref_row)
        if i == -1 || j == -1
            # the reference residual can decay to machine zero faster than the
            # block residual (a single column group is captured early); refresh
            # it with a new random index instead of terminating
            ref_failures += 1
            if ref_failures > 3
                converged = residual_negligible() ||
                            sampled_rel_residual() <= 10 * eps
                break
            end
            if i == -1
                c = rand(rng, 1:ng)
                ref_col = calc_residual_cols(c)
                scale = max(scale, maximum(abs.(ref_col)))
            end
            if j == -1
                r = rand(rng, 1:mg)
                ref_row = calc_residual_rows(r)
                scale = max(scale, maximum(abs.(ref_row)))
            end
            continue
        end
        ref_failures = 0

        # complete the cross through the larger of the two candidates; the
        # pivot pair is (i, j) with the missing index taken from the argmax of
        # the corresponding residual, and BOTH the residual column group and
        # the residual row group of the pivot are required for the term
        if vi >= vj
            row_g = i                               # candidate row group i
            row_res = calc_residual_rows(row_g)     # dr x n_cols
            scale = max(scale, maximum(abs.(row_res)))
            col_g, _ = best_col_group(row_res)      # missing column group
            if col_g == -1
                dead_rows[row_g] = true
                continue
            end
            a = calc_residual_cols(col_g)           # n_rows x dc
            scale = max(scale, maximum(abs.(a)))
        else
            col_g = j                               # candidate column group j
            a = calc_residual_cols(col_g)           # n_rows x dc
            scale = max(scale, maximum(abs.(a)))
            row_g, _ = best_row_group(a)            # missing row group
            if row_g == -1
                dead_cols[col_g] = true
                continue
            end
            row_res = calc_residual_rows(row_g)     # dr x n_cols
            scale = max(scale, maximum(abs.(row_res)))
        end

        # pivot block and the normalized update matrix
        P = a[dr*(row_g - 1) + 1 : dr*row_g, :]     # dr x dc
        if sqrt(sum(abs2, P)) <= pivot_tol * scale
            # numerically zero pivot block: the residual is negligible where
            # the look-ahead looks; verify the whole block with fresh samples
            converged = residual_negligible()
            break
        end
        W = pinv(P) * row_res                       # dc x n_cols

        used_rows[row_g] = true
        used_cols[col_g] = true
        # copy: in the look-ahead branches one factor side IS the reference
        # residual, which is updated in place below — the stored factor must
        # not alias it
        push!(U_blocks, copy(a))
        push!(V_blocks, copy(W))
        l += 1

        # update the reference residuals with the new rank-block term
        ref_col .-= a * W[:, dc*(c - 1) + 1 : dc*c]        # dc x dc update
        ref_row .-= a[dr*(r - 1) + 1 : dr*r, :] * W         # dr x n_cols update

        # refresh the reference when a pivot coincides with it
        if row_g == r
            for _ in 1:8
                r = rand(rng, 1:mg)
                ref_row = calc_residual_rows(r)
                scale = max(scale, maximum(abs.(ref_row)))
                maximum(abs.(ref_row)) > pivot_tol * scale && break
            end
        end
        if col_g == c
            for _ in 1:8
                c = rand(rng, 1:ng)
                ref_col = calc_residual_cols(c)
                scale = max(scale, maximum(abs.(ref_col)))
                maximum(abs.(ref_col)) > pivot_tol * scale && break
            end
        end

        # relative stopping: the next term is negligible compared to the
        # Frobenius norm of the block (estimated from the spread samples)
        step_norm2 = sum(abs2, a * W)
        if step_norm2 <= eps^2 * block_fro2
            converged = true
            break
        end
    end

    # Reaching the group-rank cap without the stopping criterion means the
    # block's ε-rank exceeds the storage crossover: `converged` stays false and
    # the caller falls back to a dense block. The break paths above have
    # already set `converged` (sampled-residual check or stopping criterion).

    # the block was exhausted before any rank-1 term was accepted
    if isempty(U_blocks)
        return zeros(n_rows, dc), zeros(dr, n_cols), converged
    end

    U_ACA = hcat(U_blocks...)              # n_rows x (l*dc)
    V_ACA = vcat(V_blocks...)              # (l*dr) x n_cols

    return U_ACA, V_ACA, converged
end

"""
    SVD_recompress(U::Matrix, V::Matrix, eps::Float64)

Performs SVD-based recompression of a matrix represented by the product `U * V'`,
truncating singular values based on a given *relative* tolerance `eps`: the
truncated tail must be smaller than `eps` times the Frobenius norm of the block,
so the achieved accuracy is independent of the block scale.

# Arguments
- `U::Matrix`: The left matrix in the decomposition.
- `V::Matrix`: The right matrix in the decomposition.
- `eps::Float64`: Relative tolerance level for truncating small singular values.

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

    # Truncate at a relative tolerance: the reversed cumulative Frobenius norm
    # of the tail must fall below eps times the Frobenius norm of the block
    block_norm = sqrt(sum(abs2, SIG))
    frobenius_norms = reverse(sqrt.(cumsum(reverse(SIG .^ 2))))

    # Find the truncation rank where the relative tail drops below eps
    rank_trunc = findfirst(x -> x < eps * block_norm, frobenius_norms)
    rank_trunc = rank_trunc === nothing ? size(U, 2) : rank_trunc - 1  # Use the full rank if no truncation is needed

    # Compute the recompressed matrices U_SVD and V_SVD up to the truncated rank
    U_SVD = QU * (W[:, 1:rank_trunc] .* SIG[1:rank_trunc]')
    V_SVD = Z[:, 1:rank_trunc]' * QV'

    return U_SVD, V_SVD
end
