# =============================================================================
# Advanced: matrix-free dipolar demag tensor with device-side kernel evaluation
#
# Scene: the same physical quantity as examples/vector_demag.jl — the point
#        dipolar demagnetization tensor between N cells (3 DOF each, so the
#        matrix is 3N x 3N) — but evaluated the advanced way: the batched
#        getindex that the ACA assembly queries is a KernelAbstractions
#        kernel running on the *chosen backend* (GPU when available). The
#        kernel function is never materialized; only the queried blocks are
#        ever computed. Kernel (R = x - y):
#
#        N_cd(x, y) = -(3 R_c R_d - r² δ_cd) / (4π r^5)
#
# Run:   julia --project=. examples/hmatrix_vector.jl [N]       (default N=2000)
#
# GPU behavior: the script picks its backend once at startup — `using CUDA` is
# never written here; `HMatrixGPU.@using_gpu()` loads whichever GPU package is
# installed in the current environment and `backend_from_name("cuda")` resolves
# it (falling back to the CPU without a functional GPU, so the entire script
# runs portably on a CPU-only system — same code, same validation). The chosen
# backend object `B` is passed explicitly everywhere (construction
# `backend=B`, arrays via create_zeros/kernel_array), and the batch getindex
# reads its launch backend from the data itself — there is no global state.
# `Array(out)` is the only synchronization needed.
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

# ---- backend selection (portable: CPU-only systems run the same script) ------
HMatrixGPU.@using_gpu()
B = try HMatrixGPU.backend_from_name("cuda") catch
    KernelAbstractions.CPU()        # examples target CUDA (the validated vendor)
end

# ---- configuration ----------------------------------------------------------
N = isempty(ARGS) ? 2000 : parse(Int, ARGS[1])
eta = 1.0          # admissibility: a block is far when dist > eta*(r_X + r_Y)
eps = 1e-6         # relative per-block tolerance (independent of kernel scale)
dims = 3           # DOF per point (magnetization components)

# ---- point set: seeded uniform samples (deterministic) -----------------------
pts = rand(MersenneTwister(10), 3, N)

# ---- device-side block evaluation -------------------------------------------
# queries from dims=3 trees arrive as whole cells: three consecutive DOFs per
# cell, components in order — one work item per cell pair evaluates the
# geometry once and writes the 3x3 block (the host asserts the invariant)
@kernel function eval_dipolar_cells!(out, @Const(X), @Const(Y), @Const(idx),
                                     @Const(idy), @Const(nci), @Const(ncj))
    ci, cj = @index(Global, NTuple)          # cell positions within the query
    @inbounds if ci <= nci && cj <= ncj      # padding guard
        ia = div(idx[3 * ci - 2] - 1, 3) + 1  # ONE div per cell (first DOF)
        jb = div(idy[3 * cj - 2] - 1, 3) + 1
        rx = X[1, ia] - Y[1, jb]; ry = X[2, ia] - Y[2, jb]; rz = X[3, ia] - Y[3, jb]
        r2 = rx * rx + ry * ry + rz * rz
        # N_cd = (3 R_c R_d - r² δ_cd) · t with t = -1/(4π r^5): one division
        # per cell pair, and r^5 = r2*r2*sqrt(r2) keeps the software pow()
        # sequence out of the kernel (sqrt is a single hardware instruction)
        t = r2 >= 1e-24 ? -1.0 / (4π * r2 * r2 * sqrt(r2)) : 0.0
        for cb in 1:3, ca in 1:3              # position within the triple IS
            Rca = ca == 1 ? rx : (ca == 2 ? ry : rz)   # the component (1,2,3
            Rcb = cb == 1 ? rx : (cb == 2 ? ry : rz)   # in order) — no mod
            delta = ca == cb ? 1.0 : 0.0
            out[3 * ci - 3 + ca, 3 * cj - 3 + cb] = (3.0 * Rca * Rcb - r2 * delta) * t
        end
    end
end

# Lazy kernel: holds the coordinates on the chosen backend (kernel_array moves
# them there) plus a host copy for the dense reference below.
struct DipolarDemagGPU <: AbstractMatrix{Float64}
    Xd::AbstractMatrix{Float64}     # target coordinates, 3 x N, backend of B
    Yd::AbstractMatrix{Float64}     # source coordinates, 3 x N, backend of B
    P::Matrix{Float64}              # host copy (dense reference materialization)
end

Base.size(K::DipolarDemagGPU) = (dims * size(K.Xd, 2), dims * size(K.Yd, 2))

# Scalar entry (host formula) — used by Matrix(K) for the dense reference.
function Base.getindex(K::DipolarDemagGPU, i::Int, j::Int)
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
    return -(3.0 * Rc * Rd - r2 * delta) / (4π * r2 * r2 * sqrt(r2))
end

# Batch entry — the only form the ACA assembly queries. Output and index sets
# are created on the backend the coordinates live on (data-driven — the launch
# backend comes from the data, no global backend is read), the block is
# evaluated by the kernel above, and the result is returned to the host
# (assembly consumes blocks on the host; a real large-scale application could
# keep them on the device).
function Base.getindex(K::DipolarDemagGPU, I::AbstractVector{Int}, J::AbstractVector{Int})
    bk = KernelAbstractions.get_backend(K.Xd)
    out = HMatrixGPU.create_zeros(bk, Float64, length(I), length(J))
    Id = HMatrixGPU.to_backend(out, collect(I))
    Jd = HMatrixGPU.to_backend(out, collect(J))
    length(I) % 3 == 0 && length(J) % 3 == 0 ||
        error("whole-cell queries expected (dims=3 trees, block sizes 3); " *
              "got $(length(I)) x $(length(J))")
    nci, ncj = length(I) ÷ 3, length(J) ÷ 3
    kernel! = eval_dipolar_cells!(bk, 256)
    kernel!(out, K.Xd, K.Yd, Id, Jd, nci, ncj; ndrange=(nci, ncj))
    return Array(out)       # Array() also synchronizes on GPUs
end

K = DipolarDemagGPU(HMatrixGPU.kernel_array(B, pts), HMatrixGPU.kernel_array(B, pts), pts)
println("  kernel coordinates live on: ", typeof(K.Xd))

# ---- cluster trees: dims=3 puts whole cells (3 DOF) in every tree node -------
Xc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)
Yc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)

# ---- build (lazy K always assembles through the CPU ACA; the block sizes
#      default to the trees' dims — one cell per pivot group) ------------------
t_asm = @elapsed H = HMatrix(K, Xc, Yc; eta=eta, eps=eps, backend=B)
st = info(H)

# ---- validate against the dense reference ------------------------------------
# Kdense is 3N x 3N = 288 MB at the default N — O(N²) materialization is only
# feasible at example sizes.
Kdense = Matrix(K)
x = rand(MersenneTwister(42), 3 * N)
xg = HMatrixGPU.create_zeros(B, Float64, size(K, 2)); copyto!(xg, x)
yg = Array(H * xg)              # Array() also synchronizes
t_mv = best_of(() -> Array(H * xg), 20)
relerr = norm(Kdense * x - yg) / norm(Kdense * x)
relerr < 1e-5 || error("validation failed: relerr=$relerr")

# ---- report -------------------------------------------------------------------
backend_name = B isa KernelAbstractions.CPU ? "CPU" : "CUDA"
@printf("[demag-device] N=%d (%dx%d) backend=%s eta=%g eps=%g\n",
        N, 3 * N, 3 * N, backend_name, eta, eps)
@printf("  assembly %.1fs | leaves %d (near %d / approx %d) | rank %d-%d | compression %.1fx\n",
        t_asm, st["leaves"], st["full_leaves"], st["admissible_leaves"],
        st["min_rank"], st["max_rank"], st["compression_ratio"])
@printf("  matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n", 1e3 * t_mv, relerr)
