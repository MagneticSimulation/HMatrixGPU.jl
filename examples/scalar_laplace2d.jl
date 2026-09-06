# =============================================================================
# Scalar BEM: 2D Laplace single-layer potential (log kernel) on a ring
#
# Scene: N boundary points on the unit circle in the z = 0 plane, the textbook
#        example of a boundary-element matrix for the 2D Laplace equation.
#        Kernel:  G(x, y) = -log|x - y| / (2π)
#
# Run:   julia --project=. examples/scalar_laplace2d.jl [N]     (default N=4000)
#
# The kernel K is a *lazy* AbstractMatrix (matrix-free): the H-matrix assembly
# never materializes K and only queries the blocks it needs through getindex.
# The same HMatrixGPU code runs unchanged on CPU and GPU — the closing section
# rebuilds the matrix on a CUDA device when one is available.
# =============================================================================

using HMatrixGPU
using KernelAbstractions
using LinearAlgebra
using Random
using Printf

# ---- configuration ----------------------------------------------------------
N = isempty(ARGS) ? 4000 : parse(Int, ARGS[1])
eta = 1.5          # admissibility: a block is far when dist > eta*(r_X + r_Y)
eps = 1e-6         # relative per-block tolerance (independent of kernel scale)

# best-of-n wall time (a full sync happens inside f for GPU results)
function best_of(f, n)
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

# ---- point set: analytic ring (deterministic, no RNG) -----------------------
pts = reduce(hcat, [[sin(2π*i/N), cos(2π*i/N), 0.0] for i in 1:N])

# ---- lazy kernel (matrix-free) ----------------------------------------------
# Only two methods are required: a scalar entry (used to materialize the dense
# reference below) and a batch entry (the only form the ACA assembly queries).
struct Laplace2D <: AbstractMatrix{Float64}
    X::Matrix{Float64}      # target coordinates, 3 x nx
    Y::Matrix{Float64}      # source coordinates, 3 x ny
end

Base.size(K::Laplace2D) = (size(K.X, 2), size(K.Y, 2))

# Scalar entry G(x, y) = -log|x-y|/(2π). The coincident-point self term is a
# measure-zero singular integral that a real BEM treats analytically; zeroing
# it is fine for this example.
function Base.getindex(K::Laplace2D, i::Int, j::Int)
    dx = K.X[1, i] - K.Y[1, j]
    dy = K.X[2, i] - K.Y[2, j]
    dz = K.X[3, i] - K.Y[3, j]
    d = sqrt(dx * dx + dy * dy + dz * dz)
    return d < 1e-12 ? 0.0 : -log(d) / (2π)
end

# Batch entry (index sets, not necessarily contiguous): a double loop over the
# scalar formula keeps the example simple. A real application would evaluate
# the kernel in vectorized form or with a device kernel — see
# examples/hmatrix_vector.jl for the KernelAbstractions version.
function Base.getindex(K::Laplace2D, I::AbstractVector{Int}, J::AbstractVector{Int})
    out = Matrix{Float64}(undef, length(I), length(J))
    for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        out[ii, jj] = K[i, j]
    end
    return out
end

K = Laplace2D(pts, pts)

# ---- cluster trees (leaf size 64) -------------------------------------------
Xc = ClusterTree(pts; max_points_per_leaf=64)
Yc = ClusterTree(pts; max_points_per_leaf=64)

# ---- build (matrix-free K always assembles through the CPU ACA path) --------
t_asm = @elapsed H = HMatrix(K, Xc, Yc; eta=eta, eps=eps)
st = info(H)

# ---- validate against the dense reference ------------------------------------
Kdense = Matrix(K)      # O(N^2) materialization — feasible at example sizes only
x = rand(MersenneTwister(42), N)
rex = Kdense * x
y = H * x
t_mv = best_of(() -> H * x, 20)
relerr = norm(rex - y) / norm(rex)
relerr < 1e-5 || error("validation failed: relerr=$relerr")

# ---- report -------------------------------------------------------------------
@printf("[laplace2d] N=%d backend=CPU eta=%g eps=%g\n", N, eta, eps)
@printf("  assembly %.1fs | leaves %d (near %d / approx %d) | rank %d-%d | compression %.1fx\n",
        t_asm, st["leaves"], st["full_leaves"], st["admissible_leaves"],
        st["min_rank"], st["max_rank"], st["compression_ratio"])
@printf("  matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n", 1e3 * t_mv, relerr)

# ---- optional GPU section ------------------------------------------------------
# The compressed factors land on the CUDA device (lazy K still assembles via
# the CPU ACA); the matvec then runs as fused GPU kernels. `eps` is a relative
# tolerance, so the accuracy is identical on both backends. No global backend
# state: the landing backend is chosen per construction.
HMatrixGPU.@using_gpu()             # loads whichever GPU package the env provides
B = try HMatrixGPU.backend_from_name("cuda") catch
    KernelAbstractions.CPU()        # examples target CUDA (the validated vendor)
end
if !(B isa KernelAbstractions.CPU)
    t_asm_g = @elapsed Hg = HMatrix(K, Xc, Yc; eta=eta, eps=eps, backend=B)
    xg = HMatrixGPU.create_zeros(B, Float64, size(K, 2)); copyto!(xg, x)
    yg = Array(Hg * xg)             # Array() also synchronizes
    t_mv_g = best_of(() -> Array(Hg * xg), 20)
    relerr_g = norm(Kdense * x - yg) / norm(Kdense * x)
    relerr_g < 1e-5 || error("GPU validation failed: $relerr_g")
    @printf("  gpu: assembly %.1fs | matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n",
            t_asm_g, 1e3 * t_mv_g, relerr_g)
end
