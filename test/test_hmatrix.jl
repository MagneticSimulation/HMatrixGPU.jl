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
@test isapprox(K * x, hmatrix * x; atol=1e-5)

h_flatten = HMatrix(K, cluster, cluster; eta=1.5, eps=1e-6, flatten=true)


ids = [c[3] + c[2] - c[1] + 1 for c in eachcol(h_flatten.V_block_indices)]
@test maximum(ids) == length(h_flatten.V_matrices) 


function dense_multiply(hmatrix::HMatrixGPU.HMatrixCPU, x::Vector)
    result = zeros(eltype(hmatrix.K), size(hmatrix.K, 1))

    x_ordered = x[hmatrix.source_index_map]

    for i in 1:length(hmatrix.dense_blocks)
        (row_start, row_end, col_start, col_end) = hmatrix.dense_block_indices[i]
        dense_block = hmatrix.dense_blocks[i]
        result[row_start:row_end] .+= dense_block * view(x_ordered, col_start:col_end)
    end

    result[hmatrix.target_index_map] .= result
    return result
end

function dense_multiply(hmatrix::HMatrixGPU.HMatrix, x::Vector)
    result = zeros(eltype(hmatrix.K), size(hmatrix.K, 1))

    source_map = hmatrix.source_index_map
    target_map = hmatrix.target_index_map
    indices = hmatrix.dense_block_indices
    Ds_array = hmatrix.dense_blocks
    
    for block = 1:size(indices, 2)
        row_start = indices[1, block]
        row_end = indices[2, block]
        col_start = indices[3, block]
        col_end = indices[4, block]
        offset = indices[5, block]

        for i = row_start:row_end
            sum = 0.0
            I =  offset + (i - row_start) * (col_end - col_start + 1)
            for j = col_start:col_end
                I += 1
                sum += Ds_array[I] * x[source_map[j]]
            end

            result[target_map[i]] += sum
        end
    end

    return result
end

a1 = dense_multiply(hmatrix, x)
b1 = dense_multiply(h_flatten, x)
@test isapprox(a1, b1; atol=1e-12)

function V_multiply(hmatrix::HMatrixGPU.HMatrixCPU, x::Vector)
    result = zeros(eltype(hmatrix.K), size(hmatrix.K, 1))

    x_ordered = x[hmatrix.source_index_map]

    results = []

    for i in 1:length(hmatrix.V_matrices)
        (row_start, row_end, col_start, col_end) = hmatrix.approx_block_indices[i]
        V = hmatrix.V_matrices[i]

        result = V * view(x_ordered, col_start:col_end)
        push!(results, result)
    end

    return vcat(results...)
end


function V_multiply(hmatrix::HMatrixGPU.HMatrix, x::Vector)
    result = zeros(eltype(hmatrix.K), size(hmatrix.K, 1))

    source_map = hmatrix.source_index_map
    target_map = hmatrix.target_index_map
    indices = hmatrix.V_block_indices
    Vs_array = hmatrix.V_matrices

    for i = 1:size(indices, 2)
        col_start = indices[1, i]
        col_end = indices[2, i]
        offset = indices[3, i]

        sum = 0.0
        I = offset
        for j in col_start:col_end
            I += 1
            sum += Vs_array[I] * x[source_map[j]]
        end
        hmatrix.Vx_buffer[i] = sum
        
    end

    return hmatrix.Vx_buffer
end


result1 = V_multiply(hmatrix, x)
vx = V_multiply(h_flatten, x)
@test isapprox(result1, vx; atol=1e-10)

@test isapprox(hmatrix * x, h_flatten * x; atol=1e-8)
