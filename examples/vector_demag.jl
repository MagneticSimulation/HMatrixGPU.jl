# =============================================================================
# Micromagnetics: point dipolar demagnetization tensor (vector kernel, 3 DOF)
#
# Scene: N sample points inside a thin-film slab (1 x 0.5 x 0.02 — strongly
#        anisotropic, the geometry that motivates grouped pivoting). The
#        kernel is the point dipole-dipole tensor acting between cell
#        magnetizations, so the matrix is 3N x 3N (three DOF per point):
#
#        N_cd(x, y) = -(3 R_c R_d - r² δ_cd) / (4π r^5),   R = x - y
#
# Run:   julia --project=. examples/vector_demag.jl [N]         (default N=2000)
#
# High-level API with one knob: the do-block kernel returns a 3x3 matrix and
# `dims=3` says so — the library then builds the trees per cell, flattens the
# DOFs as K[3(p-1)+c, 3(q-1)+e] = g(x_p, y_q)[c, e] and defaults the ACA block
# sizes to the cell (3, 3). That grouped pivoting matters for anisotropic
# vector kernels — in a thin film the in-plane components dominate the tensor,
# and scalar per-column pivoting would starve the weak out-of-plane components.
# The compressed matvec keeps flat vectors in/out (3N entries). Real
# micromagnetics replaces the point dipole by the Newell analytic cell
# integral (which regularizes the self term and the near field); the point
# dipole with a zeroed self term keeps the example self-contained. The
# explicit low-level mode of the same workflow is hmatrix_vector.jl (device-
# side kernel evaluation).
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
N = isempty(ARGS) ? 2000 : parse(Int, ARGS[1])
eta = 1.5          # admissibility: a block is far when dist > eta*(r_X + r_Y)
eps = 1e-4         # relative per-block tolerance (practical demag benchmark level)
dims = 3           # DOF per point (magnetization components) — the single knob

# ---- point set: seeded uniform samples in a thin-film slab (deterministic) ---
p = rand(MersenneTwister(11), 3, N)
pts = p .* [1.0; 0.5; 0.02]     # x ∈ [0,1], y ∈ [0,0.5], z ∈ [0,0.02]

# ---- build (high-level): the physics is the do-block -------------------------
# The kernel receives the coordinates x, y of a cell-center pair and must be
# *total*: the coincident-point self term is regularized by the Newell cell
# integral in real micromagnetics; zeroing it is fine for this example.
t_asm = @elapsed H = HMatrix(pts; dims=dims, eta=eta, eps=eps,
                             max_points_per_leaf=64) do x, y
    rx = x[1] - y[1]; ry = x[2] - y[2]; rz = x[3] - y[3]
    r2 = rx * rx + ry * ry + rz * rz
    Ncd = zeros(3, 3)
    r2 < 1e-24 && return Ncd
    for d in 1:3, c in 1:3
        Rc = c == 1 ? rx : (c == 2 ? ry : rz)
        Rd = d == 1 ? rx : (d == 2 ? ry : rz)
        delta = c == d ? 1.0 : 0.0
        Ncd[c, d] = -(3.0 * Rc * Rd - r2 * delta) / (4π * r2^2.5)
    end
    Ncd
end
st = info(H)

# ---- validate against the dense reference ------------------------------------
# The same physics as a named function (bitwise-identical arithmetic — keep in
# sync with the do-block above), wrapped in the library's exported lazy
# KernelMatrix. The block query Kref[1:3N, 1:3N] materializes the O(N²)
# reference (feasible at example sizes only) with one g call per point pair —
# the whole 3x3 cell is filled at once (the same memoized path the ACA uses).
function demag_tensor_ref(x, y)
    rx = x[1] - y[1]; ry = x[2] - y[2]; rz = x[3] - y[3]
    r2 = rx * rx + ry * ry + rz * rz
    Ncd = zeros(3, 3)
    r2 < 1e-24 && return Ncd
    for d in 1:3, c in 1:3
        Rc = c == 1 ? rx : (c == 2 ? ry : rz)
        Rd = d == 1 ? rx : (d == 2 ? ry : rz)
        delta = c == d ? 1.0 : 0.0
        Ncd[c, d] = -(3.0 * Rc * Rd - r2 * delta) / (4π * r2^2.5)
    end
    Ncd
end
Kref = KernelMatrix(demag_tensor_ref, pts, pts; dims=dims)
Kdense = Kref[1:(dims * N), 1:(dims * N)]
x = rand(MersenneTwister(42), dims * N)
rex = Kdense * x
y = H * x
t_mv = best_of(() -> H * x, 20)
relerr = norm(rex - y) / norm(rex)
relerr < 1e-5 || error("validation failed: relerr=$relerr")

# ---- report -------------------------------------------------------------------
@printf("[demag-vector] N=%d (%dx%d) backend=CPU eta=%g eps=%g\n", N, dims * N, dims * N, eta, eps)
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
    t_asm_g = @elapsed Hg = HMatrix(demag_tensor_ref, pts; dims=dims, eta=eta,
                                    eps=eps, max_points_per_leaf=64, backend=B)
    xg = HMatrixGPU.create_zeros(B, Float64, length(x)); copyto!(xg, x)
    yg = Array(Hg * xg)             # Array() also synchronizes
    t_mv_g = best_of(() -> Array(Hg * xg), 20)
    relerr_g = norm(Kdense * x - yg) / norm(Kdense * x)
    relerr_g < 1e-5 || error("GPU validation failed: $relerr_g")
    @printf("  gpu: assembly %.1fs | matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n",
            t_asm_g, 1e3 * t_mv_g, relerr_g)
end

# Teaching notes:
# - `dims=3` is the only knob: the trees count DOFs (3 per point, whole cells
#   in every node), the kernel matrix is flattened automatically, and the ACA
#   pivots one cell (3 components) at a time — grouped pivoting follows the
#   same `dims` without further settings.
# - The compression ratio of the vector demag kernel (a few ×) is far below
#   that of smooth scalar kernels: the 1/r^3 tensor decays slowly and its
#   thin-film anisotropy raises the numerical rank of every admissible block.
#   eps=1e-4 is the practical benchmark level; tightening eps trades storage
#   for accuracy roughly linearly in the rank.
# - Large-scale version: hmatrix_vector.jl evaluates the queried blocks with a
#   KernelAbstractions kernel directly on the device (explicit low-level mode).
