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
# High-level API: the physics is written once as a kernel function g — it
# receives the *coordinates* of a target/source point pair — and the library
# derives the lazy kernel matrix, the cluster trees and the compressed
# H-matrix from it. No struct, no size, no getindex boilerplate. (The explicit
# low-level mode — a hand-written lazy AbstractMatrix with full control over
# storage and batched evaluation — is shown in examples/scalar_laplace3d.jl.)
#
# The kernel is extremely smooth, so admissible (far) blocks are low rank and
# their entries decay exponentially with separation. At these parameters the
# storage is dominated by the near-field blocks (56% of the leaves in 3D), so
# the compression is moderate — the H-matrix benefit here is the O(N log N)
# work and O(N) storage of the compressed matvec instead of the O(N²) dense
# product, which is what matters at scale in a GP/kriging iterative solve.
# Because the H-matrix tolerance `eps` is relative, the overall scale σ is
# irrelevant for the compression (try σ = 1e3 — the relerr is unchanged). The
# assembly never materializes the dense N x N covariance matrix, which is
# exactly the O(N²) storage that H-matrices avoid in GP applications. The
# closing section rebuilds the matrix on a CUDA device when one is available.
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

# ---- build (high-level): the physics is the do-block -------------------------
# A Gaussian kernel is smooth and bounded, so g is naturally total — no
# self-pair guard is needed (compare the log kernel in scalar_laplace2d.jl).
# The library wraps the block in a lazy KernelMatrix, builds the cluster trees
# and assembles — one call, no bookkeeping.
t_asm = @elapsed H = HMatrix(pts; eta=eta, eps=eps, max_points_per_leaf=64) do x, y
    dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
    sigma^2 * exp(-(dx * dx + dy * dy + dz * dz) / (2 * ell^2))
end
st = info(H)

# ---- validate against the dense reference ------------------------------------
# The same physics as a named function (bitwise-identical arithmetic — keep in
# sync with the do-block above), wrapped in the library's exported lazy
# KernelMatrix and materialized: O(N^2) storage, feasible at example sizes only.
function gaussian_cov_ref(x, y)
    dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
    sigma^2 * exp(-(dx * dx + dy * dy + dz * dz) / (2 * ell^2))
end
Kdense = Matrix(KernelMatrix(gaussian_cov_ref, pts, pts))
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
# The compressed factors land on the CUDA device (the kernel is evaluated on
# the host during assembly); the matvec then runs as fused GPU kernels. `eps`
# is a relative tolerance, so the accuracy is identical on both backends. No
# global backend state: the landing backend is chosen per construction. The
# named-function form here is equivalent to the do-block used above — both
# produce the same compressed matrix (same ranks, same relerr).
HMatrixGPU.@using_gpu()             # loads whichever GPU package the env provides
B = try HMatrixGPU.backend_from_name("cuda") catch
    KernelAbstractions.CPU()        # examples target CUDA (the validated vendor)
end
if !(B isa KernelAbstractions.CPU)
    t_asm_g = @elapsed Hg = HMatrix(gaussian_cov_ref, pts; eta=eta, eps=eps,
                                    max_points_per_leaf=64, backend=B)
    xg = HMatrixGPU.create_zeros(B, Float64, length(x)); copyto!(xg, x)
    yg = Array(Hg * xg)             # Array() also synchronizes
    t_mv_g = best_of(() -> Array(Hg * xg), 20)
    relerr_g = norm(Kdense * x - yg) / norm(Kdense * x)
    relerr_g < 1e-5 || error("GPU validation failed: $relerr_g")
    @printf("  gpu: assembly %.1fs | matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n",
            t_asm_g, 1e3 * t_mv_g, relerr_g)
end
