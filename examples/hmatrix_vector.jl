using Random
using LinearAlgebra
using HMatrixGPU
using Test
Random.seed!(10)

using KernelAbstractions
using StaticArrays

using CUDA

function dipolar_tensor_xx(x::Float64, y::Float64, z::Float64)
    R = x * x + y * y + z * z
    if R == 0
        return 0.0
    else
        return -(2 * x * x - y * y - z * z) / (R * R * sqrt(R)) / (4 * pi)
    end
end

function dipolar_tensor_xy(x::Float64, y::Float64, z::Float64)
    R = x * x + y * y + z * z
    if R == 0
        return 0.0
    else
        return -3 * x * y / (R * R * sqrt(R)) / (4 * pi)
    end
end

function dipolar_tensor_yy(x::Float64, y::Float64, z::Float64)
    return dipolar_tensor_xx(y, x, z)
end

function dipolar_tensor_zz(x::Float64, y::Float64, z::Float64)
    return dipolar_tensor_xx(z, y, x)
end

function dipolar_tensor_xz(x::Float64, y::Float64, z::Float64)
    return dipolar_tensor_xy(x, z, y)
end

function dipolar_tensor_yz(x::Float64, y::Float64, z::Float64)
    return dipolar_tensor_xy(y, z, x)
end

function dipolar_tensor(x, y, z)
    xx = dipolar_tensor_xx(x, y, z)
    xy = dipolar_tensor_xy(x, y, z)
    xz = dipolar_tensor_xz(x, y, z)
    yy = dipolar_tensor_yy(x, y, z)
    yz = dipolar_tensor_yz(x, y, z)
    zz = dipolar_tensor_zz(x, y, z)
    return @SMatrix [xx xy xz; xy yy yz; xz yz zz]
end

N = 2000;
pts = CUDA.rand(Float64, 3, N)

@kernel function __kernel!(output, @Const(X), @Const(Y), @Const(idx), @Const(idy))
    i, j = @index(Global, NTuple)
    I = 3 * i - 2
    J = 3 * j - 2

    Ic = idx[I]
    Jc = idy[J]
    x = X[Ic] - Y[Jc]
    y = X[Ic + 1] - Y[Jc + 1]
    z = X[Ic + 2] - Y[Jc + 2]

    N = dipolar_tensor(x, y, z)
    output[I, J] = N[1, 1]
    output[I, J + 1] = N[1, 2]
    output[I, J + 2] = N[1, 3]
    output[I + 1, J] = N[2, 1]
    output[I + 1, J + 1] = N[2, 2]
    output[I + 1, J + 2] = N[2, 3]
    output[I + 2, J] = N[3, 1]
    output[I + 2, J + 1] = N[3, 2]
    output[I + 2, J + 2] = N[3, 3]
end

@kernel function __kernel2!(output, @Const(X), @Const(Y), I3, @Const(idy))
    j = @index(Global)
    J = idy[3 * j - 2]

    I = 3 * div(I3 - 1, 3) + 1
    i = I3 - I + 1

    x = X[I] - Y[J]
    y = X[I + 1] - Y[J + 1]
    z = X[I + 2] - Y[J + 2]

    N = dipolar_tensor(x, y, z)
    output[3 * j - 2] = N[i, 1]
    output[3 * j - 1] = N[i, 2]
    output[3 * j] = N[i, 3]
end

@kernel function __kernel3!(output, @Const(X), @Const(Y), @Const(idx), J3)
    i = @index(Global)
    I = idx[3 * i - 2]

    J = 3 * div(J3 - 1, 3) + 1
    j = J3 - J + 1

    x = X[I] - Y[J]
    y = X[I + 1] - Y[J + 1]
    z = X[I + 2] - Y[J + 2]
    N = dipolar_tensor(x, y, z)
    output[3 * i - 2] = N[1, j]
    output[3 * i - 1] = N[2, j]
    output[3 * i] = N[3, j]
end

struct MyCustomMatrix <: AbstractMatrix{Float64}
    X::AbstractMatrix{Float64} # target points
    Y::AbstractMatrix{Float64} # source points
end

Base.size(K::MyCustomMatrix) = 3 * size(K.X, 2), 3 * size(K.Y, 2)

function Base.getindex(K::MyCustomMatrix, idx::AbstractArray{Int}, idy::AbstractArray{Int})
    output = HMatrixGPU.create_zeros(Float64, length(idx), length(idy))
    kernel! = __kernel!(HMatrixGPU.default_backend[], 256)
    M, N = length(idx), length(idy)
    kernel!(output, K.X, K.Y, idx, idy; ndrange=(div(M, 3), div(N, 3)))
    return Array(output)
end

function Base.getindex(K::MyCustomMatrix, I::Int, idy::AbstractArray{Int})
    output = HMatrixGPU.create_zeros(Float64, length(idy))
    kernel! = __kernel2!(HMatrixGPU.default_backend[], 256)
    N = length(idy)
    kernel!(output, K.X, K.Y, I, idy; ndrange=div(N, 3))
    return Array(output)
end

function Base.getindex(K::MyCustomMatrix, idx::AbstractArray{Int}, J::Int)
    output = HMatrixGPU.create_zeros(Float64, length(idx))
    kernel! = __kernel3!(HMatrixGPU.default_backend[], 256)
    M = length(idx)
    kernel!(output, K.X, K.Y, idx, J; ndrange=div(M, 3))
    return Array(output)
end

K = MyCustomMatrix(pts, pts)

cluster_targets = ClusterTree(Array(pts); max_points_per_leaf=64, dims=3)
cluster_source = ClusterTree(Array(pts); max_points_per_leaf=64, dims=3)

@time hmatrix = HMatrix(K, cluster_targets, cluster_source; eta=1.0, eps=1e-6, flatten=true,
                        index_map_using_cpu=false)

# disable the test for now, as the flattened HMatrix only supports gpus
# @test isapprox(hmatrix * x, h_flatten * x; atol=1e-8)
