# ---------------------------------------------------------------------------
# Batched GPU assembly for dense kernels, generic over KernelAbstractions
# backends (moved out of the CUDA extension). Which backends can run it is
# decided at runtime by the device_svd_available probe — no vendor types are
# referenced here. The CUDA extension still provides a CUDA-specialized method
# of build_matrices_gpu_dense until the extensions are removed.
# ---------------------------------------------------------------------------

# capability cache: does a device-side svd work for this backend type?
const svd_capable_backends = Dict{Type,Bool}()

function device_svd_available(B)::Bool
    B isa KernelAbstractions.CPU && return false
    return get!(svd_capable_backends, typeof(B)) do
        try
            A = KernelAbstractions.zeros(B, Float64, 2, 2)
            copyto!(A, [1.0 2.0; 3.0 4.0])
            svd(A)
            KernelAbstractions.synchronize(B)
            true
        catch
            false
        end
    end
end

# block gather: out[ii, jj] = Kmat[tmap[rs + ii - 1], smap[cs + jj - 1]] for the
# cluster ranges (rs:re, cs:ce). Plain index arithmetic instead of device-side
# fancy indexing, whose semantics are not guaranteed across vendors; host
# scalars carry the ranges so nothing but the index maps and the matrix live on
# the device.
@kernel function gather_block!(out, @Const(Kmat), @Const(tmap), @Const(smap),
                               @Const(rs), @Const(re), @Const(cs), @Const(ce))
    ii, jj = @index(Global, NTuple)
    @inbounds if ii <= re - rs + 1 && jj <= ce - cs + 1
        out[ii, jj] = Kmat[tmap[rs + ii - 1], smap[cs + jj - 1]]
    end
end

"""
    build_matrices_gpu_dense(K_cpu, backend, target_index_map, source_index_map,
                             dense_blocks, approx_blocks; eps=1e-5)

Batched GPU assembly of the far field for a *dense* kernel `K_cpu`: the whole
matrix is uploaded to the device once, every far block is gathered by a kernel
and factorized with a single-sided randomized SVD (Halko et al.; the
"randomized range approximation" of Dölz et al.) — one device SVD per block;
the range basis comes from the small Gram matrix of the test projection,
orthogonalized twice (CholeskyQR2-style), with the range cutoff tied to `eps`
so tight tolerances keep resolving below the truncation threshold. The test
matrix `Ω` is seeded per block, so repeated assemblies of the same matrix are
bitwise identical. Blocks whose ε-rank exceeds the storage crossover `m*n/(m+n)`
are reported as dense (they stay in the near field); numerically zero blocks
are dropped (their rows stay covered by the near/U partition). All heavy
operations are device-side GEMMs plus the per-block device `svd`.

Generic over KernelAbstractions backends: any backend passing the
`device_svd_available` probe works. Device-side slicing and fancy indexing are
deliberately not relied upon (the block gather is a plain kernel); the
device-array slicing/view support used by the factorization itself is exactly
what the probe tests.

The randomized SVD needs no grouped pivoting and is immune to component
anisotropy, so `row_block_size`/`col_block_size` are not used on this path.

# Returns
- `K_gpu`: the uploaded matrix (used by the packing for the near field).
- `dense_block_indices`: near-field blocks plus the high-rank far blocks.
- `U_factors`: `m × r` device left factors of the low-rank blocks.
- `V_factors`: `n × r` device views of the right SVD factors (column j = rank
  row j, the far-CSR row order).
- `ranks`: the truncated rank of each low-rank block.
- `approx_block_indices`: cluster ranges of the low-rank blocks.
"""
function build_matrices_gpu_dense(K_cpu::Matrix,
                                  backend::KernelAbstractions.Backend,
                                  target_index_map::Vector{Int},
                                  source_index_map::Vector{Int},
                                  dense_blocks::Vector,
                                  approx_blocks::Vector;
                                  eps::Float64=1e-5)
    # the assembly always computes in Float64 (the randomized SVD and the
    # gathers are ill-conditioned in Float32); the caller stores in eltype(K)
    K64 = eltype(K_cpu) === Float64 ? K_cpu : Float64.(K_cpu)
    K_gpu = move_to_backend(backend, K64)                   # uploaded once
    tmap_d = move_to_backend(backend, target_index_map)     # uploaded once, gathered per block
    smap_d = move_to_backend(backend, source_index_map)
    gather! = gather_block!(backend, groupsize[])

    # per-block device factors: U (m x r) materialized, V kept as an n x r
    # view of the SVD result (column j = rank row j — the far-CSR row order,
    # so the packing broadcasts it straight into v_data without transposing)
    U_factors = AbstractMatrix{Float64}[]
    V_factors = AbstractMatrix{Float64}[]
    ranks = Int[]
    dense_far = Tuple{Int,Int,Int,Int}[]
    approx_block_indices = Vector{Tuple{Int,Int,Int,Int}}()

    # range truncation tied to the block tolerance: Q must reach below the
    # (eps/10) tail threshold, so keep singular directions down to ~eps/50
    # (the sqrt(l) slack covers a worst-case flat junk tail; a flat spectrum
    # also inflates ||B||_F by the same factor, so the two worst cases do not
    # coincide). The floor keeps CholeskyQR2's first pass inside its stability
    # bound kappa(G) = 1/lambda_cut <~ 1e16.
    λ_cut = max((eps / 50)^2, 1e-16)

    for (bi, (a, b)) in enumerate(approx_blocks)
        rs, re = a.start_idx, a.end_idx - 1
        cs, ce = b.start_idx, b.end_idx - 1
        m, n = re - rs + 1, ce - cs + 1
        Bi = KernelAbstractions.zeros(backend, Float64, m, n)   # device gather (m x n)
        gather!(Bi, K_gpu, tmap_d, smap_d, rs, re, cs, ce; ndrange=(m, n))

        # storage crossover in columns; range-finder size with oversampling
        crossover = floor(Int, m * n / (m + n))
        l = min(min(m, n), crossover + 16)

        # one-sided randomized range: the basis comes from the small Gram
        # matrix of the test projection (host eigen solve, l x l) so only ONE
        # device SVD per block is needed
        # per-block seeding keeps repeated assemblies of the same matrix
        # bitwise identical (the same invariant as the CPU ACA path); the
        # n x l draw is tiny, so the async host-to-device upload is free
        Ω = move_to_backend(backend, randn(MersenneTwister(bi), n, l))  # n x l test matrix
        Y = Bi * Ω                                  # m x l random range
        G = Symmetric(Array(Y' * Y))                # l x l Gram (host)
        E = eigen(G)
        λ = E.values
        keep = λ .> maximum(λ) * λ_cut
        l1 = count(keep)
        # a numerically zero block is dropped entirely: its target rows stay
        # covered by the near/U row-set partition, so `near_u_mul_vec_warp!`
        # writes each of them exactly once (as zero)
        l1 == 0 && continue
        Q = Y * move_to_backend(backend, E.vectors[:, keep] .* (1 ./ sqrt.(λ[keep]))')  # m x l1 orthonormal

        # second Gram pass (CholeskyQR2-style): the first pass loses up to
        # O(eps * lambda_max/lambda_min) orthogonality on fast-decaying spectra
        G2 = Symmetric(Array(Q' * Q))
        E2 = eigen(G2)
        λ2 = E2.values
        keep2 = λ2 .> maximum(λ2) * 1e-12
        l1 = count(keep2)
        l1 == 0 && continue                        # safety valve (λ2 ≈ 1 in theory)
        Q = Q * move_to_backend(backend, E2.vectors[:, keep2] .* (1 ./ sqrt.(λ2[keep2]))')

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

    if length(dense_far) > length(approx_blocks) / 2
        @warn "GPU dense assembly: $(length(dense_far))/$(length(approx_blocks)) " *
              "far blocks fell back to dense storage at eps=$eps — compression " *
              "will be poor; consider a looser eps or a matrix-free K (CPU ACA path)" maxlog = 1
    end

    # combined dense list: the near-field blocks plus the high-rank far blocks
    dense_near = [(a.start_idx, a.end_idx - 1, b.start_idx, b.end_idx - 1)
                  for (a, b) in dense_blocks]
    dense_all = vcat(dense_near, dense_far)
    return K_gpu, dense_all, U_factors, V_factors, ranks, approx_block_indices
end
