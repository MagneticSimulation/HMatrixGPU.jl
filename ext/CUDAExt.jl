module CUDAExt

using HMatrixGPU
using CUDA
using KernelAbstractions
using LinearAlgebra
using Random

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
    # the assembly always computes in Float64 (the randomized SVD and the
    # gathers are ill-conditioned in Float32); the caller stores in eltype(K)
    K64 = eltype(K_cpu) === Float64 ? K_cpu : Float64.(K_cpu)
    K_gpu = CuArray(K64)                        # uploaded once
    tmap_d = CuArray(target_index_map)          # uploaded once, sliced per block
    smap_d = CuArray(source_index_map)

    # per-block device factors: U (m x r) materialized, V kept as an n x r
    # view of the SVD result (column j = rank row j — the far-CSR row order,
    # so the packing broadcasts it straight into v_data without transposing)
    U_factors = AbstractMatrix{Float64}[]
    V_factors = AbstractMatrix{Float64}[]
    ranks = Int[]
    dense_far = Tuple{Int,Int,Int,Int}[]
    approx_block_indices = Vector{Tuple{Int,Int,Int,Int}}()

    for (bi, (a, b)) in enumerate(approx_blocks)
        rows = tmap_d[a.start_idx:(a.end_idx - 1)]   # device-side range copy
        cols = smap_d[b.start_idx:(b.end_idx - 1)]
        m, n = length(rows), length(cols)
        Bi = K_gpu[rows, cols]                       # device gather (m x n)

        # storage crossover in columns; range-finder size with oversampling
        crossover = floor(Int, m * n / (m + n))
        l = min(min(m, n), crossover + 16)

        # one-sided randomized range: the basis comes from the small Gram
        # matrix of the test projection (host eigen solve, l x l) so only ONE
        # device SVD per block is needed
        # per-block seeding keeps repeated assemblies of the same matrix
        # bitwise identical (the same invariant as the CPU ACA path); the
        # n x l draw is tiny, so the async host-to-device upload is free
        Ω = CuArray(randn(MersenneTwister(bi), n, l))  # n x l test matrix
        Y = Bi * Ω                                  # m x l random range
        G = Symmetric(Array(Y' * Y))                # l x l Gram (host)
        E = eigen(G)
        λ = E.values
        keep = λ .> maximum(λ) * 1e-12
        l1 = count(keep)
        # a numerically zero block is dropped entirely: its target rows stay
        # covered by the near/U row-set partition, so `near_u_mul_vec_warp!`
        # writes each of them exactly once (as zero)
        l1 == 0 && continue
        Q = Y * CuArray(E.vectors[:, keep] .* (1 ./ sqrt.(λ[keep]))')  # m x l1 orthonormal

        # second Gram pass (CholeskyQR2-style): the first pass loses up to
        # O(eps * lambda_max/lambda_min) orthogonality on fast-decaying spectra
        G2 = Symmetric(Array(Q' * Q))
        E2 = eigen(G2)
        λ2 = E2.values
        keep2 = λ2 .> maximum(λ2) * 1e-12
        l1 = count(keep2)
        l1 == 0 && continue                        # safety valve (λ2 ≈ 1 in theory)
        Q = Q * CuArray(E2.vectors[:, keep2] .* (1 ./ sqrt.(λ2[keep2]))')

        Bt = Q' * Bi                               # l1 x n projected block
        U2, S2, V2 = svd(Bt)                       # the single device SVD

        # relative Frobenius tail truncation, measured on the projected block
        S2h = Array(S2)
        block_norm = sqrt(sum(abs2, S2h))
        tails = reverse(sqrt.(cumsum(reverse(abs2.(S2h)))))
        r = something(findfirst(x -> x < (eps / 10) * block_norm, tails),
                      length(S2h)) - 1

        if S2h[end] > (eps / 10) * block_norm || (r + 1) * (m + n) >= m * n
            # the range could not resolve the truncation (ε-rank beyond the
            # crossover): keep the block dense
            push!(dense_far, (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
            continue
        end
        r = max(r, 1)

        Uf = Q * (U2[:, 1:r] .* S2[1:r]')           # m x r left factor
        Vf = view(V2, 1:n, 1:r)                     # n x r right factor
        push!(U_factors, Uf)
        push!(V_factors, Vf)
        push!(ranks, r)
        push!(approx_block_indices,
              (a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1))
    end

    # combined dense list: the near-field blocks plus the high-rank far blocks
    dense_near = [(a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1)
                  for (a, b) in dense_blocks]
    dense_all = vcat(dense_near, dense_far)
    return K_gpu, dense_all, U_factors, V_factors, ranks, approx_block_indices
end

end
