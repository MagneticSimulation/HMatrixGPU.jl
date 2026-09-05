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
# Index convention:  K[3(p-1)+c, 3(q-1)+d] = N_cd(pts[:,p] - pts[:,q]) with
# c, d ∈ 1:3. Both cluster trees are built with dims=3 so every tree node
# covers whole cells (DOF ranges are multiples of 3).
#
# Real micromagnetics replaces the point dipole by the Newell analytic
# cell integral (which regularizes the self term and the near field); the
# point dipole with a zeroed self term keeps the example self-contained.
# =============================================================================

using HMatrixGPU
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
dims = 3           # DOF per point (magnetization components)

# ---- point set: seeded uniform samples in a thin-film slab (deterministic) ---
p = rand(MersenneTwister(11), 3, N)
pts = p .* [1.0; 0.5; 0.02]     # x ∈ [0,1], y ∈ [0,0.5], z ∈ [0,0.02]

# ---- lazy kernel (matrix-free) ----------------------------------------------
# Only two methods are required: a scalar entry (used to materialize the dense
# reference below) and a batch entry (the only form the ACA assembly queries).
struct DipolarDemag <: AbstractMatrix{Float64}
    P::Matrix{Float64}      # cell-center coordinates, 3 x N (host)
end

Base.size(K::DipolarDemag) = (dims * size(K.P, 2), dims * size(K.P, 2))

# Scalar entry for DOF (i, j): cell pair (p, q) and component pair (c, d).
# The coincident-point self term is regularized by the Newell cell integral in
# real micromagnetics; zeroing it is fine for this example.
function Base.getindex(K::DipolarDemag, i::Int, j::Int)
    p = (i - 1) ÷ 3 + 1
    q = (j - 1) ÷ 3 + 1
    c = (i - 1) % 3 + 1
    d = (j - 1) % 3 + 1
    rx = K.P[1, p] - K.P[1, q]
    ry = K.P[2, p] - K.P[2, q]
    rz = K.P[3, p] - K.P[3, q]
    r2 = rx * rx + ry * ry + rz * rz
    r2 < 1e-24 && return 0.0
    Rc = c == 1 ? rx : (c == 2 ? ry : rz)
    Rd = d == 1 ? rx : (d == 2 ? ry : rz)
    delta = c == d ? 1.0 : 0.0
    return -(3.0 * Rc * Rd - r2 * delta) / (4π * r2^2.5)
end

# Batch entry (index sets, not necessarily contiguous): a double loop over the
# scalar formula keeps the example simple. A real application would evaluate
# the tensor in vectorized form or with a device kernel — see
# examples/hmatrix_vector.jl for the KernelAbstractions version.
function Base.getindex(K::DipolarDemag, I::AbstractVector{Int}, J::AbstractVector{Int})
    out = Matrix{Float64}(undef, length(I), length(J))
    for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        out[ii, jj] = K[i, j]
    end
    return out
end

K = DipolarDemag(pts)

# ---- cluster trees: dims=3 puts whole cells (3 DOF) in every tree node -------
Xc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)
Yc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)

# ---- build (matrix-free K always assembles through the CPU ACA path) --------
# row_block_size = col_block_size = 3: the ACA pivots one cell (3 components)
# at a time. This grouped pivoting matters for anisotropic vector kernels —
# in a thin film the in-plane components dominate the tensor, and scalar
# per-column pivoting would starve the weak out-of-plane components.
t_asm = @elapsed H = HMatrix(K, Xc, Yc; eta=eta, eps=eps,
                             row_block_size=dims, col_block_size=dims)
st = info(H)

# ---- validate against the dense reference ------------------------------------
# Kdense is 3N x 3N = 288 MB at the default N — O(N²) materialization is only
# feasible at example sizes.
Kdense = Matrix(K)
x = rand(MersenneTwister(42), 3 * N)
rex = Kdense * x
y = H * x
t_mv = best_of(() -> H * x, 20)
relerr = norm(rex - y) / norm(rex)
relerr < 1e-5 || error("validation failed: relerr=$relerr")

# ---- report -------------------------------------------------------------------
@printf("[demag-vector] N=%d (%dx%d) backend=CPU eta=%g eps=%g\n", N, 3 * N, 3 * N, eta, eps)
@printf("  assembly %.1fs | leaves %d (near %d / approx %d) | rank %d-%d | compression %.1fx\n",
        t_asm, st["leaves"], st["full_leaves"], st["admissible_leaves"],
        st["min_rank"], st["max_rank"], st["compression_ratio"])
@printf("  matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n", 1e3 * t_mv, relerr)

# ---- optional GPU section ------------------------------------------------------
# The compressed factors land on the CUDA device (lazy K still assembles via
# the CPU ACA); the matvec then runs as fused GPU kernels. `eps` is a relative
# tolerance, so the accuracy is identical on both backends.
HMatrixGPU.@using_gpu()             # loads whichever GPU package is installed
if set_backend("cuda")              # returns false (clean skip) without a GPU
    t_asm_g = @elapsed Hg = HMatrix(K, Xc, Yc; eta=eta, eps=eps,
                                    row_block_size=dims, col_block_size=dims,
                                    backend="cuda")
    xg = HMatrixGPU.create_zeros(Float64, size(K, 2)); copyto!(xg, x)
    yg = Array(Hg * xg)             # Array() also synchronizes
    t_mv_g = best_of(() -> Array(Hg * xg), 20)
    relerr_g = norm(Kdense * x - yg) / norm(Kdense * x)
    relerr_g < 1e-5 || error("GPU validation failed: $relerr_g")
    @printf("  gpu: assembly %.1fs | matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n",
            t_asm_g, 1e3 * t_mv_g, relerr_g)
    set_backend("cpu")
end

# Teaching notes:
# - `dims=3` in the ClusterTree makes tree indices count DOFs (3 per point),
#   so a leaf of 64 points covers 192 consecutive DOFs and every ACA query
#   carries whole cells.
# - The compression ratio of the vector demag kernel (a few ×) is far below
#   that of smooth scalar kernels: the 1/r^3 tensor decays slowly and its
#   thin-film anisotropy raises the numerical rank of every admissible block.
#   eps=1e-4 is the practical benchmark level; tightening eps trades storage
#   for accuracy roughly linearly in the rank.
