using Random
using LinearAlgebra
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

Base.size(K::MyCustomMatrix) = size(K.X, 2), 3 * size(K.Y, 2)

function Base.getindex(K::MyCustomMatrix, i::Int, vj::Int)
    J = div(vj - 1, 3) + 1
    j = mod(vj - 1, 3) + 1
    R = K.X[:, i] .- K.Y[:, J]
    r = norm(R)
    return r < 1e-12 ? 1.0 : R[j] / r^2
end

K = MyCustomMatrix(pts, pts)

cluster_targets = ClusterTree(pts; max_points_per_leaf=64)
cluster_source = ClusterTree(pts; max_points_per_leaf=64, dims=3)

# build + matvec with a multi-component source, run on every available
# backend via test_functions
function test_hmatrix_vector_matvec(B)
    hmatrix = HMatrix(K, cluster_targets, cluster_source; eta=1.5, eps=1e-6,
                      row_block_size=1, col_block_size=3, backend=B)

    @test length(Set(cluster_targets.index_map)) == N
    @test length(Set(cluster_source.index_map)) == 3 * N
    @test maximum(cluster_source.index_map) == 3 * N
    @test minimum(cluster_source.index_map) == 1

    d = info(hmatrix)
    @test d["compression_ratio"] > 3
    @test !any(isnan, Array(hmatrix.near_data))
    @test !any(isnan, Array(hmatrix.v_data))
    @test !any(isnan, Array(hmatrix.u_data))

    x = rand(3 * N)
    xd = HMatrixGPU.create_zeros(B, Float64, 3 * N)
    copyto!(xd, x)
    @test norm(K * x - Array(hmatrix * xd)) / norm(K * x) < 1e-4
end

test_functions("HMatrix vector", test_hmatrix_vector_matvec)

# ---------------------------------------------------------------------------
# default ACA block sizes follow the trees' dims: for dims=3 trees the default
# construction must take the same (3,3) grouped-pivoting path as the explicit
# row_block_size=3, col_block_size=3 one — the ACA rng is seeded with
# hash((n_rows, n_cols, row_block, col_block)), so both assemblies are
# identical (CPU path; a dense mixed-component log kernel keeps real ACA work)
# ---------------------------------------------------------------------------
M3 = 500
pts3m = reduce(hcat, [[sin(i * 2π / M3), cos(i * 2π / M3), 0.0] for i in 1:M3])
Klog = [-0.5 / π * log(max(norm(pts3m[:, i] .- pts3m[:, j]), 1e-12))
        for i in 1:M3, j in 1:M3]
K3 = kron([1.0 0.4 0.2; 0.4 1.0 0.4; 0.2 0.4 1.0], Klog)
ct3 = ClusterTree(pts3m; max_points_per_leaf=64, dims=3)
cs3 = ClusterTree(pts3m; max_points_per_leaf=64, dims=3)

Hd = HMatrix(K3, ct3, cs3; eta=1.5, eps=1e-6)
He = HMatrix(K3, ct3, cs3; eta=1.5, eps=1e-6, row_block_size=3, col_block_size=3)
@test Hd.ranks == He.ranks
@test Hd.napprox > 0                      # real ACA work, not all-dense fallback
x3 = rand(3 * M3)
@test Hd * x3 == He * x3                  # bitwise: identical assembly
@test norm(K3 * x3 - Hd * x3) / norm(K3 * x3) < 1e-4
