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

Base.size(K::MyCustomMatrix) = size(K.X, 2), 3*size(K.Y, 2)

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
hmatrix = HMatrix(K, cluster_targets, cluster_source; eta=1.5, eps=1e-6, flatten=false)


@test length(Set(cluster_source.index_map)) == 3*N
@test maximum(cluster_source.index_map) == 3*N
@test minimum(cluster_source.index_map) == 1

d = info(hmatrix)

@test d["compression_ratio"] > 3

for M in hmatrix.dense_blocks
    @test !any(isnan, M)
end

for U in hmatrix.U_matrices
    @test !any(isnan, U)
end

for V in hmatrix.V_matrices
    @test !any(isnan, V)
end

x = rand(3*N)
@test isapprox(K * x, hmatrix * x; atol=1e-5)


h_flatten = HMatrix(K, cluster_targets, cluster_source; eta=1.5, eps=1e-6, flatten=true)


ids = [c[3] + c[2] - c[1] + 1 for c in eachcol(h_flatten.V_block_indices)]
@test maximum(ids) == length(h_flatten.V_matrices) 


@test isapprox(hmatrix * x, h_flatten * x; atol=1e-8)
