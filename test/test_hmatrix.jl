using Random
using LinearAlgebra
using SparseArrays
using KernelAbstractions
using HMatrixGPU
using Test
Random.seed!(10)

N = 1000;

X = [[sin(i * 2π / N), cos(i * 2π / N), 0] for i in 1:N]
pts = hcat(X...)

struct MyCustomMatrix <: AbstractMatrix{Float64}
    X::Matrix{Float64}
    Y::Matrix{Float64}
end

Base.size(K::MyCustomMatrix) = size(K.X, 2), size(K.Y, 2)

function Base.getindex(K::MyCustomMatrix, i::Int, j::Int)
    d = norm(K.X[:, i] .- K.Y[:, j])
    return d < 1e-12 ? 1.0 : -0.5 / pi * log(d)
end

K = MyCustomMatrix(pts, pts)

cluster = ClusterTree(pts; max_points_per_leaf=64)

# ---------------------------------------------------------------------------
# build + matvec, run on every available backend via test_functions
# ---------------------------------------------------------------------------
function test_hmatrix_matvec(B)
    hmatrix = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, backend=B)

    d = info(hmatrix)
    @test d["compression_ratio"] > 3
    @test d["leaves"] == d["admissible_leaves"] + d["full_leaves"]
    @test Base.size(hmatrix) == (N, N)
    @test !any(isnan, Array(hmatrix.near_data))
    @test !any(isnan, Array(hmatrix.v_data))
    @test !any(isnan, Array(hmatrix.u_data))

    x = rand(N)
    xd = HMatrixGPU.create_zeros(B, Float64, N)
    copyto!(xd, x)
    @test norm(K * x - Array(hmatrix * xd)) / norm(K * x) < 1e-4

    y = HMatrixGPU.create_zeros(B, Float64, N)
    mul!(y, hmatrix, xd)
    @test norm(K * x - Array(y)) / norm(K * x) < 1e-4
end

# the explicit idiom that replaces the removed like= keyword: place the matrix
# next to a reference array via backend = KernelAbstractions.get_backend(...)
function test_backend_from_data_idiom(B)
    like = HMatrixGPU.create_zeros(B, Float32, 0)
    h = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6,
                backend=KernelAbstractions.get_backend(like))
    @test nameof(typeof(h.near_data)) == nameof(typeof(like))
    x = rand(N)
    @test isapprox(K * x, Array(h * HMatrixGPU.to_backend(like, x)); rtol=1e-4)
end

test_functions("HMatrix", test_hmatrix_matvec, test_backend_from_data_idiom)

function test_float32_end_to_end(B)
    K32 = Float32.(K)
    x32 = rand(Float32, N)
    y_ref = K32 * x32

    h32 = HMatrix(K32, cluster, cluster; eta=1.5, eps=1e-6, backend=B)
    @test eltype(h32.near_data) == Float32

    x32d = HMatrixGPU.create_zeros(B, Float32, N)
    copyto!(x32d, x32)
    y32 = Array(h32 * x32d)
    @test eltype(y32) == Float32
    @test norm(Float64.(y32 .- y_ref)) / norm(Float64.(y_ref)) < 1e-3

    y32d = HMatrixGPU.create_zeros(B, Float32, N)
    mul!(y32d, h32, x32d)
    @test norm(Float64.(Array(y32d) .- y_ref)) / norm(Float64.(y_ref)) < 1e-3
end

test_functions("Float32 end-to-end", test_float32_end_to_end)

function test_backend_keyword(B)
    x = rand(N)             # own vector: file-scope x is shadowed by other files
    @test_throws ErrorException HMatrix(K, cluster, cluster;
                                        backend="nonsense")

    # explicit names and objects; loading CUDA has no side effect on HMatrixGPU
    h_cpu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, backend="cpu")
    @test h_cpu.near_data isa Array

    # "gpu" is deliberately not a backend name (no auto-detection)
    @test_throws ErrorException HMatrixGPU.backend_from_name("gpu")

    if CUDA.functional()
        h_cu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, backend="cuda")
        @test h_cu.near_data isa CuArray
        x_cu = CuArray(x)
        @test isapprox(h_cpu * x, Array(h_cu * x_cu); rtol=1e-4)

        # v1.1: device data no longer decides the placement — a CuArray K
        # without keywords lands on the CPU (per-block queries download)
        pts_small = reduce(hcat, [[sin(i * 2π / 40), cos(i * 2π / 40), 0.0]
                                  for i in 1:40])
        cl_small = ClusterTree(pts_small; max_points_per_leaf=16)
        K_cu = CuArray(randn(40, 40))
        h_df = HMatrix(K_cu, cl_small, cl_small; eta=1.5, eps=1e-6)
        @test h_df.near_data isa Array
    end
end
test_functions("backend keyword", test_backend_keyword)

# ---------------------------------------------------------------------------
# CPU and GPU instances coexist in one process and their matvecs interleave:
# with no global backend state the instances can never interfere (DESIGN §F.3)
# ---------------------------------------------------------------------------
function test_backend_coexistence(B)
    h_cpu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, backend="cpu")
    x = rand(N)
    y1 = h_cpu * x
    @test norm(K * x - y1) / norm(K * x) < 1e-4
    if !(B isa KernelAbstractions.CPU)
        h_gpu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, backend=B)
        xg = HMatrixGPU.create_zeros(B, Float64, N); copyto!(xg, x)
        @test isapprox(Array(h_gpu * xg), y1; rtol=1e-10)
        y2 = h_cpu * x                 # interleave back to the CPU instance
        @test y2 == y1                 # bitwise: instances never interfere
        @test Array(h_gpu * xg) == Array(h_gpu * xg)
    end
end
test_functions("backend coexistence", test_backend_coexistence)

# ---------------------------------------------------------------------------
# CPU-only checks: the CSR packing must reproduce the assembled blocks
# ---------------------------------------------------------------------------
hmatrix = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6)
x = rand(N)

# rebuild the per-block factors through the assembly pipeline for reference
bt = BlockTree(cluster, cluster; eta=1.5)
HMatrixGPU.merge_dense_matrices!(bt.root)
dense_blocks, approx_blocks = HMatrixGPU.traverse(bt)
dense_mats, Umats, Vmats, dense_idx, approx_idx = HMatrixGPU.build_matrices(
    K, bt.target_index_map, bt.source_index_map, dense_blocks, approx_blocks;
    eps=1e-6)

@test hmatrix.ndense == length(dense_mats)
@test hmatrix.napprox == length(Umats)
@test hmatrix.ranks == [size(U, 2) for U in Umats]
@test all(U -> !any(isnan, U), Umats)
@test all(V -> !any(isnan, V), Vmats)
@test all(D -> !any(isnan, D), dense_mats)

yf = zeros(N)
mul!(yf, hmatrix, x)
@test norm(K * x - yf) / norm(K * x) < 1e-4

# the rank-row buffer must equal the stacked V * x products of the blocks
function vx_from_blocks(Vmats, approx_idx, smap, x)
    x_ordered = x[smap]
    out = Float64[]
    for i in eachindex(Vmats)
        (rs, re, cs, ce) = approx_idx[i]
        append!(out, Vmats[i] * view(x_ordered, cs:ce))
    end
    return out
end
@test isapprox(Array(hmatrix.vx_buffer),
               vx_from_blocks(Vmats, approx_idx, Array(hmatrix.source_index_map), x);
               atol=1e-7)

# the near-field CSR must reproduce the dense blocks of the assembly.
# CSR rows are in cluster order and column indices are cluster positions;
# scatter both back to original indices and compare with `sparsify_hmatrix`.
function near_matrix_from_csr(h)
    ptr = Array(h.near_rowptr)
    col = Array(h.near_colval)
    val = Array(h.near_data)
    tmap = Array(h.target_index_map)
    smap = Array(h.source_index_map)
    A = spzeros(length(ptr) - 1, h.n)
    for i in 1:(length(ptr) - 1)
        for k in (ptr[i] + 1):ptr[i + 1]
            A[tmap[i], smap[col[k]]] += val[k]
        end
    end
    return A
end
S_ref = HMatrixGPU.sparsify_hmatrix(K, cluster, cluster; eta=1.5)
@test isapprox(near_matrix_from_csr(hmatrix), S_ref; atol=1e-12)

# ---------------------------------------------------------------------------
# GPU dense assembly: dense fallback, empty approx path and determinism.
# Registered for the CPU platform only — the function builds explicitly with
# backend="cuda" and returns early when no functional GPU is present.
# ---------------------------------------------------------------------------
function test_gpu_dense_assembly_fallback(B)
    CUDA.functional() || return

    # dense Float64 log kernel + an incompressible random patch on the
    # first quarter arc: some far blocks fall back, others stay low-rank;
    # ring geometry at eta=1.5 guarantees cluster rows shared by several
    # approx blocks (interleaved u segments). eps=1e-4 is the FD demag
    # working point; the tight-eps mixed scenario is covered by
    # test_gpu_dense_assembly_tight_eps below.
    rng = MersenneTwister(42)
    Kd = Matrix{Float64}(undef, N, N)
    for j in 1:N, i in 1:N
        Kd[i, j] = K[i, j]
    end
    q = 1:div(N, 4)
    Kd[q, q] .+= randn(rng, length(q), length(q))

    bt = BlockTree(cluster, cluster; eta=1.5)
    HMatrixGPU.merge_dense_matrices!(bt.root)
    dense_blocks, approx_blocks = HMatrixGPU.traverse(bt)

    H = HMatrix(Kd, cluster, cluster; eta=1.5, eps=1e-4, backend="cuda")
    @test H.ndense > length(dense_blocks)          # ≥1 far block fell back
    @test 0 < H.napprox < length(approx_blocks)    # ≥1 kept low-rank

    x = rand(N)
    xd = CuArray(x)
    @test norm(Kd * x - Array(H * xd)) / norm(Kd * x) < 1e-4

    # incompressible everywhere: every far block falls back (empty approx
    # path: L=0, no per-block launches, mul! skips phase 1)
    Kr = randn(rng, N, N)
    Hr = HMatrix(Kr, cluster, cluster; eta=1.5, eps=1e-4, backend="cuda")
    @test Hr.napprox == 0
    @test norm(Kr * x - Array(Hr * xd)) / norm(Kr * x) < 1e-4

    # deterministic assembly: rebuild → identical ranks, bitwise-identical y
    H2 = HMatrix(Kd, cluster, cluster; eta=1.5, eps=1e-4, backend="cuda")
    @test H2.ranks == H.ranks
    @test Array(H2 * xd) == Array(H * xd)
end
test_functions("GPU dense assembly fallback", test_gpu_dense_assembly_fallback;
               platforms=["CPU"])

function test_gpu_dense_assembly_tight_eps(B)
    CUDA.functional() || return

    # same mixed matrix as the fallback test, but at eps=1e-6: before the
    # eps-aware range cutoff every far block fell back at this tolerance;
    # now the smooth blocks must compress again while the random patch
    # still falls back (crossover guard is eps-independent)
    rng = MersenneTwister(43)
    Kd = Matrix{Float64}(undef, N, N)
    for j in 1:N, i in 1:N
        Kd[i, j] = K[i, j]
    end
    q = 1:div(N, 4)
    Kd[q, q] .+= randn(rng, length(q), length(q))

    bt = BlockTree(cluster, cluster; eta=1.5)
    HMatrixGPU.merge_dense_matrices!(bt.root)
    dense_blocks, approx_blocks = HMatrixGPU.traverse(bt)

    H = HMatrix(Kd, cluster, cluster; eta=1.5, eps=1e-6, backend="cuda")
    @test H.ndense > length(dense_blocks)          # patch still falls back
    @test H.napprox > 0                            # smooth blocks compress at 1e-6

    x = rand(N)
    xd = CuArray(x)
    @test norm(Kd * x - Array(H * xd)) / norm(Kd * x) < 1e-5

    H2 = HMatrix(Kd, cluster, cluster; eta=1.5, eps=1e-6, backend="cuda")
    @test H2.ranks == H.ranks
    @test Array(H2 * xd) == Array(H * xd)
end
test_functions("GPU dense assembly tight eps", test_gpu_dense_assembly_tight_eps;
               platforms=["CPU"])

@test_throws ErrorException HMatrix(complex.(K[1:4, 1:4]), cluster, cluster)
