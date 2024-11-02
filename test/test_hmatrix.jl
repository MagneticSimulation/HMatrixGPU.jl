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

Base.size(K::MyCustomMatrix) = size(K.X, 2), size(K.Y, 2)

function Base.getindex(K::MyCustomMatrix, i::Int, j::Int)
    d = norm(K.X[:, i] .- K.Y[:, j])
    return d < 1e-12 ? 1.0 : -0.5 / pi * log(d)
end

K = MyCustomMatrix(pts, pts)

cluster = ClusterTree(pts; max_points_per_leaf=64)
hmatrix = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6)

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

x = rand(N)
@test isapprox(K * x, hmatrix * x; atol=1e-5)