using Random
using LinearAlgebra
using KernelAbstractions
using HMatrixGPU
using Test

# ---------------------------------------------------------------------------
# High-level API: KernelMatrix + function-kernel HMatrix constructors.
# Eight test groups per TASK_highlevel_api.md §F4 (groups 6/7 merged into the
# v1.1 device-input test); every function takes the platform backend B and is
# registered through test_functions (task-4 harness).
#
# The reference kernels use explicit component arithmetic and are shared by
# the high-level path and the hand-written low-level structs below, so both
# assembly paths query bitwise-identical values — the rank/matvec comparisons
# are sharp (any arithmetic difference could flip an ACA pivot).
# ---------------------------------------------------------------------------

Nh = 500
pts_hl = [(sin(2π * i / Nh), cos(2π * i / Nh), 0.0) for i in 1:Nh]   # point vector
pts_hl_mat = stack(pts_hl)                                           # d x N matrix

# scalar 2D Laplace log kernel (total: the self-pair guard lives inside)
function log_kernel(x, y)
    dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
    d = sqrt(dx * dx + dy * dy + dz * dz)
    return d < 1e-12 ? 0.0 : -log(d) / (2π)
end

# vector demag-type tensor kernel: returns a 3x3 matrix (total: self pair zeroed)
function demag_tensor(x, y)
    rx = x[1] - y[1]; ry = x[2] - y[2]; rz = x[3] - y[3]
    r2 = rx * rx + ry * ry + rz * rz
    out = zeros(3, 3)
    r2 < 1e-24 && return out
    for e in 1:3, c in 1:3
        Rc = c == 1 ? rx : (c == 2 ? ry : rz)
        Rd = e == 1 ? rx : (e == 2 ? ry : rz)
        delta = c == e ? 1.0 : 0.0
        out[c, e] = -(3.0 * Rc * Rd - r2 * delta) / (4π * r2^2.5)
    end
    return out
end

# hand-written lazy kernel — the pre-high-level way (struct + size + getindex)
struct LazyLogKernel <: AbstractMatrix{Float64}
    X::Matrix{Float64}
    Y::Matrix{Float64}
end
Base.size(K::LazyLogKernel) = (size(K.X, 2), size(K.Y, 2))
Base.getindex(K::LazyLogKernel, i::Int, j::Int) =
    log_kernel(view(K.X, :, i), view(K.Y, :, j))
Base.getindex(K::LazyLogKernel, I::AbstractVector{Int}, J::AbstractVector{Int}) =
    (out = Matrix{Float64}(undef, length(I), length(J));
     for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
         out[ii, jj] = K[i, j]
     end;
     out)

# hand-written flattened vector kernel: DOF index sets, 3 DOF per point
struct LazyDemagKernel <: AbstractMatrix{Float64}
    P::Matrix{Float64}
end
Base.size(K::LazyDemagKernel) = (3 * size(K.P, 2), 3 * size(K.P, 2))
function Base.getindex(K::LazyDemagKernel, i::Int, j::Int)
    v = demag_tensor(view(K.P, :, div(i - 1, 3) + 1), view(K.P, :, div(j - 1, 3) + 1))
    return v[rem(i - 1, 3) + 1, rem(j - 1, 3) + 1]
end
Base.getindex(K::LazyDemagKernel, I::AbstractVector{Int}, J::AbstractVector{Int}) =
    (out = Matrix{Float64}(undef, length(I), length(J));
     for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
         out[ii, jj] = K[i, j]
     end;
     out)

# ---------------------------------------------------------------------------
# 1. high-level vs hand-written low-level (scalar kernel): same parameters —
#    identical ranks, matvec agreement at 1e-12·norm, relerr < 1e-5
# ---------------------------------------------------------------------------
function test_highlevel_vs_lowlevel_scalar(B)
    H1 = HMatrix(log_kernel, pts_hl; eta=1.5, eps=1e-6, backend=B)
    K = LazyLogKernel(pts_hl_mat, pts_hl_mat)
    Xc = ClusterTree(pts_hl_mat; max_points_per_leaf=32)
    H2 = HMatrix(K, Xc, Xc; eta=1.5, eps=1e-6, backend=B)
    @test H1.ranks == H2.ranks

    x = rand(Nh)
    ref = Matrix(K) * x                       # exact dense kernel
    xd = HMatrixGPU.create_zeros(B, Float64, Nh); copyto!(xd, x)
    y1 = Array(H1 * xd)
    y2 = Array(H2 * xd)
    @test isapprox(y1, y2; atol=1e-12 * norm(ref))
    @test norm(ref - y1) / norm(ref) < 1e-5
end

# ---------------------------------------------------------------------------
# 2. vector kernel dims=3: the auto-default block sizes follow the trees' dims
#    (3, 3) and must assemble identically to the explicit row_block_size=3,
#    col_block_size=3 low-level construction (same ACA rng seed)
# ---------------------------------------------------------------------------
function test_highlevel_vector_dims3(B)
    H1 = HMatrix(demag_tensor, pts_hl; dims=3, eta=1.5, eps=1e-4, backend=B)
    X3 = ClusterTree(pts_hl_mat; max_points_per_leaf=32, dims=3)
    H2 = HMatrix(LazyDemagKernel(pts_hl_mat), X3, X3; eta=1.5, eps=1e-4,
                 row_block_size=3, col_block_size=3, backend=B)
    @test size(H1) == (3 * Nh, 3 * Nh)
    @test H1.napprox > 0                     # real ACA work on the admissible blocks
    @test H1.ranks == H2.ranks

    x = rand(3 * Nh)
    xd = HMatrixGPU.create_zeros(B, Float64, 3 * Nh); copyto!(xd, x)
    @test Array(H1 * xd) == Array(H2 * xd)   # bitwise: identical assembly

    ref = demag_tensor |>
          g -> KernelMatrix(g, pts_hl, pts_hl; dims=3)
    refd = ref[1:(3 * Nh), 1:(3 * Nh)] * x   # exact dense kernel (batch query)
    @test norm(refd - Array(H1 * xd)) / norm(refd) < 1e-4
end

# ---------------------------------------------------------------------------
# 3. dual-form point sets: matrix and point-vector inputs must normalize to
#    the same coordinates exactly (stack of the vector = the matrix), so the
#    assemblies are identical
# ---------------------------------------------------------------------------
function test_highlevel_dual_form(B)
    Hv = HMatrix(log_kernel, pts_hl; eta=1.5, eps=1e-6, backend=B)
    Hm = HMatrix(log_kernel, pts_hl_mat; eta=1.5, eps=1e-6, backend=B)
    @test Hv.ranks == Hm.ranks
    x = rand(Nh)
    xd = HMatrixGPU.create_zeros(B, Float64, Nh); copyto!(xd, x)
    @test Array(Hv * xd) == Array(Hm * xd)   # bitwise
end

# ---------------------------------------------------------------------------
# 4. construction-time validation probe: type/dims mismatches surface at
#    KernelMatrix construction, not mid-assembly
# ---------------------------------------------------------------------------
function test_highlevel_probe_errors(B)
    @test_throws ErrorException KernelMatrix(demag_tensor, pts_hl, pts_hl; dims=1)
    @test_throws ErrorException KernelMatrix(log_kernel, pts_hl, pts_hl; dims=3)
end

# ---------------------------------------------------------------------------
# 5. dims conflict between the explicit kwarg and custom trees
# ---------------------------------------------------------------------------
function test_highlevel_dims_conflict(B)
    X3 = ClusterTree(pts_hl_mat; max_points_per_leaf=32, dims=3)
    Y3 = ClusterTree(pts_hl_mat; max_points_per_leaf=32, dims=3)
    @test_throws ErrorException HMatrix(log_kernel, X3, Y3; dims=1)
    H = HMatrix(demag_tensor, X3, Y3)        # dims=nothing follows the trees
    @test size(H) == (3 * Nh, 3 * Nh)
end

# ---------------------------------------------------------------------------
# 6. device point sets (even mixed-device) build on any platform's backend:
#    they are downloaded once for the host-side trees, but the placement is
#    explicit-only — no keyword lands the factor arrays on the CPU (CUDA
#    platform only; the CPU construction is covered by the other groups)
# ---------------------------------------------------------------------------
function test_highlevel_device_input(B)
    B isa KernelAbstractions.CPU && return
    # device (even mixed) point sets still build — downloaded once for the
    # host-side trees — but the placement is explicit-only: no keyword → CPU
    H = HMatrix(log_kernel, CuArray(pts_hl_mat), pts_hl_mat; eta=1.5, eps=1e-6)
    @test H.near_data isa Array
    H2 = HMatrix(log_kernel, CuArray(pts_hl_mat); eta=1.5, eps=1e-6, backend=B)
    @test H2.near_data isa CuArray
end

# ---------------------------------------------------------------------------
# 7. do-block smoke test: the syntax sugar HMatrix(pts; ...) do x, y ... end
#    must build, report and validate like the named-function form
# ---------------------------------------------------------------------------
function test_highlevel_do_block_smoke(B)
    H = HMatrix(pts_hl; eta=1.5, eps=1e-6, backend=B) do x, y
        dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
        d = sqrt(dx * dx + dy * dy + dz * dz)
        d < 1e-12 ? 0.0 : -log(d) / (2π)
    end
    st = info(H)
    @test st["size"] == (Nh, Nh)
    @test st["leaves"] == st["admissible_leaves"] + st["full_leaves"]
    x = rand(Nh)
    ref = Matrix(KernelMatrix(log_kernel, pts_hl, pts_hl)) * x
    xd = HMatrixGPU.create_zeros(B, Float64, Nh); copyto!(xd, x)
    y = Array(H * xd)
    @test norm(ref - y) / norm(ref) < 1e-5
end

# ---------------------------------------------------------------------------
# 8. batch getindex semantics: the (Vector, Vector) block query must agree
#    with the elementwise entries, including the dims=3 memoized path
# ---------------------------------------------------------------------------
function test_highlevel_batch_getindex(B)
    Xt = pts_hl_mat[:, 1:100]
    Ys = pts_hl_mat[:, 60:160]              # rectangular, partially overlapping
    K = KernelMatrix(log_kernel, Xt, Ys)
    I = [17, 3, 100, 3, 41, 2]              # repeated + unsorted
    J = [11, 1, 11, 99, 5, 5]
    @test size(K[I, J]) == (length(I), length(J))
    @test K[I, J] == [K[i, j] for i in I, j in J]

    K3 = KernelMatrix(demag_tensor, Xt, Xt; dims=3)
    I3 = [4, 5, 6, 13, 14, 2, 3, 1]         # whole cells then interleaved DOFs
    J3 = [7, 8, 9, 1, 3, 2]
    @test K3[I3, J3] == [K3[i, j] for i in I3, j in J3]
    @test K3[1:9, 4:6] == [K3[i, j] for i in 1:9, j in 4:6]   # whole-cell block
end

test_functions("high-level scalar vs low-level", test_highlevel_vs_lowlevel_scalar)
test_functions("high-level vector dims=3", test_highlevel_vector_dims3)
test_functions("high-level dual form", test_highlevel_dual_form)
test_functions("high-level kernel probe errors", test_highlevel_probe_errors)
test_functions("high-level tree dims conflict", test_highlevel_dims_conflict)
test_functions("high-level device input", test_highlevel_device_input;
               platforms=["CUDA"])
test_functions("high-level do-block smoke", test_highlevel_do_block_smoke)
test_functions("high-level batch getindex", test_highlevel_batch_getindex)
