module CUDAExt

using HMatrixGPU
using CUDA
using KernelAbstractions
using LinearAlgebra

CUDA.allowscalar(false)

function set_cuda_backend()
    HMatrixGPU.all_backends[1] = CUDA.CUDABackend()
    HMatrixGPU.set_backend("cuda")
    return nothing
end

# backend follows the data: assembled host arrays follow the device arrays
HMatrixGPU.to_backend(like::CuArray, a::AbstractArray) = a isa CuArray ? a : CuArray(a)

function __init__()
    # importing a GPU package must not switch the backend on machines
    # without a working GPU (e.g. CPU-only CI runners)
    CUDA.functional() && set_cuda_backend()
    return nothing
end

# ---------------- batched GPU assembly (dense kernels) ----------------

HMatrixGPU.gpu_dense_assembly_available(::CUDA.CUDABackend) = CUDA.functional()

"""
    HMatrixGPU.build_matrices_gpu_dense(K_cpu, backend, target_index_map,
        source_index_map, dense_blocks, approx_blocks; eps=1e-5)

Batched GPU assembly of the far field for a *dense* kernel `K_cpu`: the whole
matrix is uploaded to the device once, every far block is evaluated by a
device gather and factorized with a randomized SVD (Halko et al.; the
"randomized range approximation" of Dölz et al.). Blocks whose ε-rank exceeds
the storage crossover `m*n/(m+n)` are reported as dense (they stay in the
near field). All heavy operations are device-side GEMMs and cuSOLVER calls.

The randomized SVD needs no grouped pivoting and is immune to component
anisotropy, so `row_block_size`/`col_block_size` are not used on this path.

# Returns
- `K_gpu`: the uploaded matrix (used by the packing for the near field).
- `dense_block_indices`: near-field blocks plus the high-rank far blocks.
- `U_factors`, `V_factors`: device factors of the low-rank blocks.
- `approx_block_indices`: cluster ranges of the low-rank blocks.
"""
function HMatrixGPU.build_matrices_gpu_dense(K_cpu::Matrix{Float64},
                                             backend::KernelAbstractions.Backend,
                                             target_index_map::Vector{Int},
                                             source_index_map::Vector{Int},
                                             dense_blocks::Vector,
                                             approx_blocks::Vector;
                                             eps::Float64=1e-5)
    K_gpu = CuArray(K_cpu)                      # uploaded once
    ndense = length(dense_blocks)

    dense_far = Tuple{Int,Int,Int,Int}[]
    U_factors = CuMatrix{Float64}[]
    V_factors = CuMatrix{Float64}[]
    approx_block_indices = Vector{Tuple{Int,Int,Int,Int}}()

    # factor arena: all blocks' U factors stacked vertically (padded to the
    # widest block), V factors stacked in the far-CSR rank-row order — so the
    # packing step is pure device-side copies with no host round trip
    sum_m = sum(a.end_idx - a.start_idx for (a, b) in approx_blocks; init=0)
    max_cols = 0
    for (a, b) in approx_blocks
        n = length(source_index_map[b.start_idx:(b.end_idx - 1)])
        crossover = floor(Int, (a.end_idx - a.start_idx) * n /
                              ((a.end_idx - a.start_idx) + n))
        max_cols = max(max_cols, min(n, crossover + 16))
    end
    Ustack = CUDA.zeros(Float64, max(sum_m, 1), max(max_cols, 1))
    Urow_off = zeros(Int, length(approx_blocks) + 1)

    urow = 1                                  # next free row of the U arena
    for (bi, (a, b)) in enumerate(approx_blocks)
        rows = target_index_map[a.start_idx:(a.end_idx - 1)]
        cols = source_index_map[b.start_idx:(b.end_idx - 1)]
        m, n = length(rows), length(cols)
        Bi = K_gpu[CuArray(rows), CuArray(cols)]    # device gather (m x n)

        # storage crossover in columns; range-finder size with oversampling
        crossover = floor(Int, m * n / (m + n))
        l = min(min(m, n), crossover + 16)

        Ω = CUDA.randn(n, l)                        # n x l test matrix
        Y = Bi * Ω                                  # m x l random range

        # two SVDs instead of QR + triangular solve: every operation used
        # (gemm, gesvd) has a native cuSOLVER implementation
        U1, S1, _ = svd(Y)                          # m x l, l, l x l
        l1 = min(l, m)                              # svd(Y) truncates
        Bt = U1[:, 1:l1]' * Bi                      # l1 x n projected block
        U2, S2, V2 = svd(Bt)                        # l1 x n

        # relative Frobenius tail truncation, measured on the projected block
        S2h = Array(S2)
        block_norm = sqrt(sum(abs2, S2h))
        tails = reverse(sqrt.(cumsum(reverse(abs2.(S2h)))))
        r = something(findfirst(x -> x < (eps / 10) * block_norm, tails),
                      length(S2h)) - 1

        if (r + 1) * (m + n) >= m * n
            # ε-rank exceeds the storage crossover: keep the block dense
            push!(dense_far, (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
            continue
        end
        r = max(r, 1)
        Urow_off[bi] = urow
        Ustack[urow:urow + m - 1, 1:r] .= U1[:, 1:l1] * (U2[:, 1:r] .* S2[1:r]')
        push!(U_factors, view(Ustack, urow:urow + m - 1, 1:r))   # m x r (arena)
        push!(V_factors, Matrix(V2[:, 1:r]'))                    # r x n
        urow += m
        push!(approx_block_indices,
              (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
    end

    # combined dense list: the near-field blocks plus the high-rank far blocks
    dense_near = [(a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1)
                  for (a, b) in dense_blocks]
    dense_all = vcat(dense_near, dense_far)
    return K_gpu, dense_all, Ustack, Urow_off, V_factors, approx_block_indices
end

end
