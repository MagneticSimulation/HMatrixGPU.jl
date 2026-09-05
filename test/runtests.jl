using HMatrixGPU
using Test

set_backend("cpu")
include("test_utils.jl")

@testset "HMatrixGPU.jl" begin
    include("test_tree.jl")
    include("test_block.jl")
    include("test_aca.jl")
    include("test_hmatrix.jl")
    include("test_hmatrix_vector.jl")
end
