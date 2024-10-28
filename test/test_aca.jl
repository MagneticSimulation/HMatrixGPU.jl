using Random
using LinearAlgebra
using HMatrixGPU
using Test
Random.seed!(10)

N = 100;
K = zeros(N, N)
for i in 1:N, j in 1:N
    if i != j
        K[i, j] = N / (i - j)^2
    end
end

nrows = 30
block = K[(end - nrows + 1):end, 1:nrows]

epsilon = 1e-6
U, V = ACA_plus(size(block, 1), size(block, 2), I -> block[I, :], J -> block[:, J],
                epsilon / 10.0)

x = rand(size(block, 2))
y_true = block * x
y_aca = U * (V * x)

max_diff = maximum(abs.(y_aca - y_true))
frob_err = norm(U * V - block, 2)

@test size(U, 2) < size(block, 2)
@test frob_err < epsilon / 100
@test max_diff < epsilon / 100

U_SVD, V_SVD = HMatrixGPU.SVD_recompress(U, V, epsilon / 10)
println(size(U), size(U_SVD))

y_aca = U_SVD * (V_SVD * x)

max_diff = maximum(abs.(y_aca - y_true))
frob_err = norm(U * V - block, 2)
@test frob_err < epsilon / 100
@test max_diff < epsilon / 100
@test size(U, 2) >= size(U_SVD, 2)
