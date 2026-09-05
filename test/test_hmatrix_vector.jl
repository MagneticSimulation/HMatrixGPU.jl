using Random
using LinearAlgebra
using HMatrixGPU
using Test
Random.seed!(10)

set_backend("cpu")
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
function test_hmatrix_vector_matvec()
    hmatrix = HMatrix(K, cluster_targets, cluster_source; eta=1.5, eps=1e-6,
                      row_block_size=1, col_block_size=3)

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
    xd = HMatrixGPU.create_zeros(Float64, 3 * N)
    copyto!(xd, x)
    @test norm(K * x - Array(hmatrix * xd)) / norm(K * x) < 1e-4
end

test_functions("HMatrix vector", test_hmatrix_vector_matvec)
