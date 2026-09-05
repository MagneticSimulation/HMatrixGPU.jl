using Random
using LinearAlgebra
using SparseArrays
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

Base.size(K::MyCustomMatrix) = size(K.X, 2), size(K.Y, 2)

function Base.getindex(K::MyCustomMatrix, i::Int, j::Int)
    d = norm(K.X[:, i] .- K.Y[:, j])
    return d < 1e-12 ? 1.0 : -0.5 / pi * log(d)
end

K = MyCustomMatrix(pts, pts)

cluster = ClusterTree(pts; max_points_per_leaf=64)
hmatrix = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, flatten=false)

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
y = rand(N)
mul!(y, hmatrix, x)
@test norm(K * x - hmatrix * x) / norm(K * x) < 1e-4
@test norm(K * x - y) / norm(K * x) < 1e-4

# ---------------------------------------------------------------------------
# CSR-compressed structure
# ---------------------------------------------------------------------------
h_flatten = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, flatten=true)

# block statistics must match the reference structure
d2 = info(h_flatten)
@test d2["leaves"] == d["leaves"]
@test d2["admissible_leaves"] == d["admissible_leaves"]
@test d2["full_leaves"] == d["full_leaves"]
@test d2["min_rank"] == d["min_rank"]
@test d2["max_rank"] == d["max_rank"]
@test isapprox(d2["compression_ratio"], d["compression_ratio"]; rtol=1e-12)
@test h_flatten.ranks == [size(U, 2) for U in hmatrix.U_matrices]
@test Base.size(h_flatten) == (N, N)

# end-to-end product must match the reference structure and the exact kernel
@test isapprox(hmatrix * x, h_flatten * x; atol=1e-8)

yf = zeros(N)
mul!(yf, h_flatten, x)
@test norm(K * x - yf) / norm(K * x) < 1e-4

# the rank-row buffer must equal the stacked V * x products of the reference
function vx_from_reference(hmatrix::HMatrixGPU.HMatrixCPU, x::Vector)
    x_ordered = x[hmatrix.source_index_map]
    out = Float64[]
    for i in eachindex(hmatrix.V_matrices)
        (rs, re, cs, ce) = hmatrix.approx_block_indices[i]
        append!(out, hmatrix.V_matrices[i] * view(x_ordered, cs:ce))
    end
    return out
end
@test isapprox(Array(h_flatten.vx_buffer), vx_from_reference(hmatrix, x); atol=1e-9)

# the near-field CSR must reproduce the dense blocks of the reference structure.
# CSR rows are in cluster order and column indices are cluster positions;
# scatter both back to original indices and compare with `sparsify_hmatrix`.
function near_matrix_from_csr(h_flatten)
    ptr = Array(h_flatten.near_rowptr)
    col = Array(h_flatten.near_colval)
    val = Array(h_flatten.near_data)
    tmap = Array(h_flatten.target_index_map)
    smap = Array(h_flatten.source_index_map)
    A = spzeros(length(ptr) - 1, h_flatten.n)
    for i in 1:(length(ptr) - 1)
        for k in (ptr[i] + 1):ptr[i + 1]
            A[tmap[i], smap[col[k]]] += val[k]
        end
    end
    return A
end
S_ref = HMatrixGPU.sparsify_hmatrix(K, cluster, cluster; eta=1.5)
@test isapprox(near_matrix_from_csr(h_flatten), S_ref; atol=1e-12)

# ---------------------------------------------------------------------------
# Float32 end-to-end (the ACA computes in Float64, factors store as Float32)
# ---------------------------------------------------------------------------
@testset "Float32 end-to-end" begin
    K32 = Float32.(K)
    x32 = rand(Float32, N)
    y_ref = K32 * x32
    h32 = HMatrix(K32, cluster, cluster; eta=1.5, eps=1e-6, flatten=false)
    y32 = h32 * x32
    @test eltype(y32) == Float32
    @test norm(Float64.(y32 .- y_ref)) / norm(Float64.(y_ref)) < 1e-3
    h32f = HMatrix(K32, cluster, cluster; eta=1.5, eps=1e-6, flatten=true)
    @test eltype(h32f.near_data) == Float32
    y32f = h32f * x32
    @test norm(Float64.(y32f .- y_ref)) / norm(Float64.(y_ref)) < 1e-3
end

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
@using_gpu()
set_backend("cuda")
if CUDA.functional()
    h_gpu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, flatten=true)
    x_gpu = CuArray(x)
    y_gpu = h_gpu * x_gpu
    @test isapprox(hmatrix * x, Array(y_gpu); atol=1e-9)

    y_gpu2 = CUDA.zeros(Float64, N)
    mul!(y_gpu2, h_gpu, x_gpu)
    @test norm(K * x - Array(y_gpu2)) / norm(K * x) < 1e-4

    # backend follows the data: build with the global backend on CPU but
    # `like` on the GPU — the factor arrays must land next to `like`
    set_backend("cpu")
    h_like = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, flatten=true,
                     like=CUDA.zeros(Float32, 0))
    @test h_like.near_data isa CuArray
    @test h_like.source_index_map isa CuArray
    y_like = h_like * CuArray(x)
    @test isapprox(hmatrix * x, Array(y_like); rtol=1e-4)
end
