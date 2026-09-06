# =============================================================================
# Spatial statistics: Gaussian covariance kernel (Gaussian process / kriging)
#
# Scene: N points drawn from a seeded RNG in the unit cube [0,1]^3 — the
#        sampling geometry of a spatial-statistics / Gaussian-process
#        regression problem. Kernel (isotropic Gaussian, σ = 1, ℓ = 0.1):
#
#        C(x, y) = σ² exp(-|x - y|² / (2 ℓ²))
#
# Run:   julia --project=. examples/covariance_gaussian.jl [N]  (default N=4000)
#
# The kernel is extremely smooth, so admissible (far) blocks are low rank and
# their entries decay exponentially with separation. At these parameters the
# storage is dominated by the near-field blocks (56% of the leaves in 3D), so
# the compression is moderate — the H-matrix benefit here is the O(N log N)
# work and O(N) storage of the compressed matvec instead of the O(N²) dense
# product, which is what matters at scale in a GP/kriging iterative solve.
# Because the H-matrix tolerance `eps` is
# relative, the overall scale σ is irrelevant for the compression (try
# σ = 1e3 — the relerr is unchanged). The kernel is a *lazy* AbstractMatrix
# (matrix-free): the assembly never materializes the dense N x N covariance
# matrix, which is exactly the O(N²) storage that H-matrices avoid in GP
# applications. The closing section rebuilds the matrix on a CUDA device when
# one is available.
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
sigma = 1.0        # kernel scale σ (does not affect the compression)
ell = 0.1          # correlation length ℓ

# ---- point set: seeded uniform samples in [0,1]^3 (deterministic) ------------
pts = rand(MersenneTwister(7), 3, N)

# ---- lazy kernel (matrix-free) ----------------------------------------------
# Only two methods are required: a scalar entry (used to materialize the dense
# reference below) and a batch entry (the only form the ACA assembly queries).
# Other covariances (e.g. Matérn family) only change the scalar formula.
struct GaussianCovariance <: AbstractMatrix{Float64}
    X::Matrix{Float64}      # coordinates, 3 x N
end

Base.size(K::GaussianCovariance) = (size(K.X, 2), size(K.X, 2))

function Base.getindex(K::GaussianCovariance, i::Int, j::Int)
    dx = K.X[1, i] - K.X[1, j]
    dy = K.X[2, i] - K.X[2, j]
    dz = K.X[3, i] - K.X[3, j]
    return sigma^2 * exp(-(dx * dx + dy * dy + dz * dz) / (2 * ell^2))
end

# Batch entry (index sets, not necessarily contiguous): a double loop over the
# scalar formula keeps the example simple. A real application would evaluate
# the kernel in vectorized form or with a device kernel — see
# examples/hmatrix_vector.jl for the KernelAbstractions version.
function Base.getindex(K::GaussianCovariance, I::AbstractVector{Int}, J::AbstractVector{Int})
    out = Matrix{Float64}(undef, length(I), length(J))
    for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        out[ii, jj] = K[i, j]
    end
    return out
end

K = GaussianCovariance(pts)

# ---- cluster trees (leaf size 64) -------------------------------------------
Xc = ClusterTree(pts; max_points_per_leaf=64)
Yc = ClusterTree(pts; max_points_per_leaf=64)

# ---- build (matrix-free K always assembles through the CPU ACA path) --------
# In a GP/kriging workflow this compressed C is what makes the matvec of an
# iterative solve (or a log-det computation) cost O(N log N) instead of O(N²).
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
@printf("[covariance] N=%d backend=CPU eta=%g eps=%g\n", N, eta, eps)
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
