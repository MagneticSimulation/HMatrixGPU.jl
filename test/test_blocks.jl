using LinearAlgebra
using KernelAbstractions
using HMatrixGPU
using Test

# ---------------------------------------------------------------------------
# hmatrix_blocks: the CSR index arrays must reconstruct leaf blocks that tile
# the matrix exactly — the packing invariants behind the plot(H) recipe (and
# now hard errors inside hmatrix_blocks). Only the index arrays are downloaded,
# so the same assertions run for CPU-resident and GPU-resident matrices.
# ---------------------------------------------------------------------------

N = 600
pts = [(sin(2π * i / N), cos(2π * i / N), 0.0) for i in 1:N]
g(x, y) = (d = norm(x - y); d < 1e-12 ? 0.0 : -log(d) / (2π))

function test_blocks_tiling(B)
    H = HMatrix(g, pts; eta=1.5, eps=1e-6, backend=B)
    dense, lowrank = hmatrix_blocks(H)

    # the near-field CSR holds every dense-block entry exactly once
    nnz_dense = sum(length(r) * length(c) for (r, c) in dense; init=0)
    @test nnz_dense == length(H.near_data)

    # the dense + low-rank leaves cover every matrix entry exactly
    area = nnz_dense +
           sum(length(r) * length(c) for (r, c, _) in lowrank; init=0)
    @test area == H.m * H.n

    # a ring at eta=1.5 exercises both block kinds, all in bounds, and the
    # low-rank storage must actually win on every admissible leaf
    @test !isempty(dense) && !isempty(lowrank)
    for (r, c) in dense
        @test 1 <= first(r) && last(r) <= H.m && 1 <= first(c) && last(c) <= H.n
    end
    for (r, c, rank) in lowrank
        @test 1 <= first(r) && last(r) <= H.m && 1 <= first(c) && last(c) <= H.n
        @test (length(r) + length(c)) * rank < length(r) * length(c)
    end
end

test_functions("hmatrix blocks tiling", test_blocks_tiling)
