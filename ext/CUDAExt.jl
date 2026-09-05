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

Batched GPU assembly of the compressed matrix for a *dense* kernel `K_cpu`:
the whole matrix is uploaded to the device once, every far block is evaluated
by a device gather and factorized with a randomized SVD
(Halko et al.; the "randomized range approximation" of Dölz et al.), and
blocks whose ε-rank exceeds the storage crossover `m*n/(m+n)` are kept
densely. All heavy operations are device-side GEMMs and cuSOLVER calls.

The randomized SVD needs no grouped pivoting and is immune to component
anisotropy, so `row_block_size`/`col_block_size` are not used on this path.
"""
function HMatrixGPU.build_matrices_gpu_dense(K_cpu::Matrix{Float64},
                                             backend::KernelAbstractions.Backend,
                                             target_index_map::Vector{Int},
                                             source_index_map::Vector{Int},
                                             dense_blocks::Vector,
                                             approx_blocks::Vector;
                                             eps::Float64=1e-5)
    K_gpu = CuArray(K_cpu)                 # uploaded once
    T = Float64
    ndense = length(dense_blocks)
    napp = length(approx_blocks)

    # near field: threaded host extraction (small scattered gathers)
    dense_matrices = Vector{Matrix{T}}(undef, ndense)
    dense_block_indices = Vector{Tuple{Int,Int,Int,Int}}(undef, ndense)
    Threads.@threads for bi in eachindex(dense_blocks)
        (a, b) = dense_blocks[bi]
        tids = view(target_index_map, a.start_idx:(a.end_idx - 1))
        sids = view(source_index_map, b.start_idx:(b.end_idx - 1))
        dense_matrices[bi] = K_cpu[tids, sids]
        dense_block_indices[bi] = (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1)
    end

    # far field: one randomized SVD per block, entirely on the device
    U_matrices = Vector{Matrix{T}}()
    V_matrices = Vector{Matrix{T}}()
    approx_block_indices = Vector{Tuple{Int,Int,Int,Int}}()

    for (a, b) in approx_blocks
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
            push!(dense_matrices, Array(Bi))
            push!(dense_block_indices,
                  (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
            continue
        end
        r = max(r, 1)
        Uf = U1[:, 1:l1] * (U2[:, 1:r] .* S2[1:r]')   # m x r left factor
        Vf = Matrix(V2[:, 1:r]')                     # r x n right factor
        push!(U_matrices, Array(Uf))
        push!(V_matrices, Array(Vf))
        push!(approx_block_indices,
              (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
    end

    return dense_matrices, U_matrices, V_matrices, dense_block_indices, approx_block_indices
end

end
