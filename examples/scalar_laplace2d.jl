# =============================================================================
# Scalar BEM: 2D Laplace single-layer potential (log kernel) on a ring
#
# Scene: N boundary points on the unit circle in the z = 0 plane, the textbook
#        example of a boundary-element matrix for the 2D Laplace equation.
#        Kernel:  G(x, y) = -log|x - y| / (2π)
#
# Run:   julia --project=. examples/scalar_laplace2d.jl [N]     (default N=4000)
#
# High-level API: the physics is written once as a kernel function g — it
# receives the *coordinates* of a target/source point pair — and the library
# derives the lazy kernel matrix, the cluster trees and the compressed
# H-matrix from it. No struct, no size, no getindex boilerplate. The same
# HMatrixGPU code runs unchanged on CPU and GPU — the closing section rebuilds
# the matrix on a CUDA device when one is available. (The explicit low-level
# mode — a hand-written lazy AbstractMatrix with full control over storage and
# batched evaluation — is shown in examples/scalar_laplace3d.jl.)
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

# ---- point set: analytic ring (deterministic, no RNG), one point per element -
pts = [(sin(2π * i / N), cos(2π * i / N), 0.0) for i in 1:N]

# ---- build (high-level): the physics is the do-block -------------------------
# The kernel must be *total*: the coincident-point self term is a measure-zero
# singular integral that a real BEM treats analytically, so regularizing it is
# part of g (the guard below). The library wraps the block in a lazy
# KernelMatrix, builds the cluster trees and assembles — one call, no
# bookkeeping.
t_asm = @elapsed H = HMatrix(pts; eta=eta, eps=eps, max_points_per_leaf=64) do x, y
    dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
    d = sqrt(dx * dx + dy * dy + dz * dz)
    d < 1e-12 ? 0.0 : -log(d) / (2π)
end
st = info(H)

# ---- validate against the dense reference ------------------------------------
# The same physics as a named function (bitwise-identical arithmetic — keep in
# sync with the do-block above), wrapped in the library's exported lazy
# KernelMatrix and materialized: O(N^2) storage, feasible at example sizes only.
function laplace2d_ref(x, y)
    dx = x[1] - y[1]; dy = x[2] - y[2]; dz = x[3] - y[3]
    d = sqrt(dx * dx + dy * dy + dz * dz)
    return d < 1e-12 ? 0.0 : -log(d) / (2π)
end
Kdense = Matrix(KernelMatrix(laplace2d_ref, pts, pts))
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
    t_asm_g = @elapsed Hg = HMatrix(laplace2d_ref, pts; eta=eta, eps=eps,
                                    max_points_per_leaf=64, backend=B)
    xg = HMatrixGPU.create_zeros(B, Float64, length(x)); copyto!(xg, x)
    yg = Array(Hg * xg)             # Array() also synchronizes
    t_mv_g = best_of(() -> Array(Hg * xg), 20)
    relerr_g = norm(Kdense * x - yg) / norm(Kdense * x)
    relerr_g < 1e-5 || error("GPU validation failed: $relerr_g")
    @printf("  gpu: assembly %.1fs | matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n",
            t_asm_g, 1e3 * t_mv_g, relerr_g)
end
