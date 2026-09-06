# =============================================================================
# Scalar BEM: 3D Laplace single-layer potential (1/r kernel) on a sphere
#
# Scene: N points distributed on the unit sphere with the deterministic
#        Fibonacci (golden-angle spiral) rule — the typical geometry of a
#        closed-surface boundary-integral equation in 3D.
#        Kernel:  G(x, y) = 1 / (4π |x - y|)
#
# Run:   julia --project=. examples/scalar_laplace3d.jl [N]     (default N=4000)
#
# The kernel K is a *lazy* AbstractMatrix (matrix-free): the H-matrix assembly
# never materializes K and only queries the blocks it needs through getindex.
# The same HMatrixGPU code runs unchanged on CPU and GPU — the closing section
# rebuilds the matrix on a CUDA device when one is available.
#
# explicit/low-level mode: full control over storage and batched evaluation;
# see scalar_laplace2d.jl for the high-level form of the same workflow.
# =============================================================================

using HMatrixGPU
using KernelAbstractions
using LinearAlgebra
using Random
using Printf

# best-of-n wall time (a full sync happens inside f for GPU results)
function best_of(f, n)
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

# ---- configuration ----------------------------------------------------------
N = isempty(ARGS) ? 4000 : parse(Int, ARGS[1])
eta = 1.5          # admissibility: a block is far when dist > eta*(r_X + r_Y)
eps = 1e-6         # relative per-block tolerance (independent of kernel scale)

# ---- point set: Fibonacci sphere (analytic formula, no RNG) ------------------
# z walks linearly from +1 to -1 while the azimuth advances by the golden
# angle, giving an even, fully deterministic coverage of the sphere.
pts = Matrix{Float64}(undef, 3, N)
for i in 1:N
    z = 1 - 2 * (i - 0.5) / N
    r = sqrt(max(0.0, 1 - z^2))
    θ = π * (1 + sqrt(5)) * i
    pts[1, i] = r * cos(θ)
    pts[2, i] = r * sin(θ)
    pts[3, i] = z
end

# ---- lazy kernel (matrix-free) ----------------------------------------------
# Only two methods are required: a scalar entry (used to materialize the dense
# reference below) and a batch entry (the only form the ACA assembly queries).
struct Laplace3D <: AbstractMatrix{Float64}
    X::Matrix{Float64}      # target coordinates, 3 x nx
    Y::Matrix{Float64}      # source coordinates, 3 x ny
end

Base.size(K::Laplace3D) = (size(K.X, 2), size(K.Y, 2))

# Scalar entry G(x, y) = 1/(4π|x-y|). The coincident-point self term is a
# singular integral that a real BEM treats analytically; zeroing it is fine
# for this example.
function Base.getindex(K::Laplace3D, i::Int, j::Int)
    dx = K.X[1, i] - K.Y[1, j]
    dy = K.X[2, i] - K.Y[2, j]
    dz = K.X[3, i] - K.Y[3, j]
    d = sqrt(dx * dx + dy * dy + dz * dz)
    return d < 1e-12 ? 0.0 : 1 / (4π * d)
end

# Batch entry (index sets, not necessarily contiguous): a double loop over the
# scalar formula keeps the example simple. A real application would evaluate
# the kernel in vectorized form or with a device kernel — see
# examples/hmatrix_vector.jl for the KernelAbstractions version.
function Base.getindex(K::Laplace3D, I::AbstractVector{Int}, J::AbstractVector{Int})
    out = Matrix{Float64}(undef, length(I), length(J))
    for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        out[ii, jj] = K[i, j]
    end
    return out
end

K = Laplace3D(pts, pts)

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
@printf("[laplace3d] N=%d backend=CPU eta=%g eps=%g\n", N, eta, eps)
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

# Teaching note on `eta`: the admissibility condition dist > eta*(r_X + r_Y)
# decides which blocks are treated as low rank. A smaller eta marks more
# blocks admissible (higher compression) but those blocks then need higher
# ACA ranks to reach the same `eps` — on a closed surface like this sphere,
# the near-field (dense) fraction shrinks accordingly.
