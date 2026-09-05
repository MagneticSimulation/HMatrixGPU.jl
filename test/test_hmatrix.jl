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
hmatrix = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6)

d = info(hmatrix)

@test d["compression_ratio"] > 3
@test d["leaves"] == d["admissible_leaves"] + d["full_leaves"]
@test Base.size(hmatrix) == (N, N)
@test !any(isnan, Array(hmatrix.near_data))
@test !any(isnan, Array(hmatrix.v_data))
@test !any(isnan, Array(hmatrix.u_data))

x = rand(N)
y = rand(N)
mul!(y, hmatrix, x)
@test norm(K * x - hmatrix * x) / norm(K * x) < 1e-4
@test norm(K * x - y) / norm(K * x) < 1e-4

# ---------------------------------------------------------------------------
# the CSR packing must reproduce the assembled blocks
# ---------------------------------------------------------------------------
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

# the deprecated alias must forward to HMatrix
h_dep = HMatrixGPU.HMatrixCPU(K, cluster, cluster; eta=1.5, eps=1e-6)
@test h_dep isa HMatrix
@test isapprox(Array(h_dep.near_data), Array(hmatrix.near_data); atol=0.0)

# ---------------------------------------------------------------------------
# Float32 end-to-end (the ACA computes in Float64, factors store as Float32)
# ---------------------------------------------------------------------------
@testset "Float32 end-to-end" begin
    K32 = Float32.(K)
    x32 = rand(Float32, N)
    y_ref = K32 * x32
    h32 = HMatrix(K32, cluster, cluster; eta=1.5, eps=1e-6)
    @test eltype(h32.near_data) == Float32
    y32 = h32 * x32
    @test eltype(y32) == Float32
    @test norm(Float64.(y32 .- y_ref)) / norm(Float64.(y_ref)) < 1e-3
    mul!(y32, h32, x32)
    @test norm(Float64.(y32 .- y_ref)) / norm(Float64.(y_ref)) < 1e-3
end

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
@using_gpu()
set_backend("cuda")
if CUDA.functional()
    h_gpu = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6)
    x_gpu = CuArray(x)
    y_gpu = h_gpu * x_gpu
    @test isapprox(hmatrix * x, Array(y_gpu); atol=1e-9)

    y_gpu2 = CUDA.zeros(Float64, N)
    mul!(y_gpu2, h_gpu, x_gpu)
    @test norm(K * x - Array(y_gpu2)) / norm(K * x) < 1e-4

    # backend follows the data: build with the global backend on CPU but
    # `like` on the GPU — the factor arrays must land next to `like`
    set_backend("cpu")
    h_like = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6,
                     like=CUDA.zeros(Float32, 0))
    @test h_like.near_data isa CuArray
    @test h_like.source_index_map isa CuArray
    y_like = h_like * CuArray(x)
    @test isapprox(K * x, Array(y_like); rtol=1e-4)
end
