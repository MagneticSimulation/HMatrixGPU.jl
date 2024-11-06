using HMatrixGPU
using Test

@testset "HMatrixGPU.jl" begin
    include("test_tree.jl")
    include("test_block.jl")
    include("test_aca.jl")
    include("test_hmatrix.jl")
    include("test_hmatrix_vector.jl")
end
