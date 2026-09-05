# =============================================================================
# Advanced: matrix-free dipolar demag tensor with device-side kernel evaluation
#
# Scene: the same physical quantity as examples/vector_demag.jl — the point
#        dipolar demagnetization tensor between N cells (3 DOF each, so the
#        matrix is 3N x 3N) — but evaluated the advanced way: the batched
#        getindex that the ACA assembly queries is a KernelAbstractions
#        kernel running on the *active backend* (GPU when available). The
#        kernel function is never materialized; only the queried blocks are
#        ever computed. Kernel (R = x - y):
#
#        N_cd(x, y) = -(3 R_c R_d - r² δ_cd) / (4π r^5)
#
# Run:   julia --project=. examples/hmatrix_vector.jl [N]       (default N=2000)
#
# GPU behavior: `HMatrixGPU.@using_gpu()` loads whichever GPU package is
# installed (CUDA/AMDGPU/oneAPI/Metal) and `set_backend("cuda")` activates it;
# on a CPU-only system set_backend returns false and the entire script runs
# portably on the CPU — same code, same validation. There is no `using CUDA`
# anywhere: the library's portable hooks (create_zeros / to_backend /
# kernel_array) place every array on the active backend, and `Array(out)` is
# the only synchronization needed.
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
gpu_ok = set_backend("cuda")    # false → everything below runs on the CPU

# ---- configuration ----------------------------------------------------------
N = isempty(ARGS) ? 2000 : parse(Int, ARGS[1])
eta = 1.0          # admissibility: a block is far when dist > eta*(r_X + r_Y)
eps = 1e-6         # relative per-block tolerance (independent of kernel scale)
dims = 3           # DOF per point (magnetization components)

# ---- point set: seeded uniform samples (deterministic) -----------------------
pts = rand(MersenneTwister(10), 3, N)

# ---- device-side block evaluation -------------------------------------------
# One thread per matrix element: the query index sets are arbitrary (the ACA
# issues single-row group queries of length 3 and (view, view) dense blocks),
# so per-element mapping is correct for *any* query shape — a kernel that
# assumed the index length is divisible by 3 would silently return zero rows
# for a 1-row query. The few extra sqrt calls per thread are irrelevant at
# example sizes.
@kernel function eval_dipolar_block!(out, @Const(X), @Const(Y), @Const(idx),
                                     @Const(idy), @Const(nrows), @Const(ncols))
    ii, jj = @index(Global, NTuple)
    @inbounds if ii <= nrows && jj <= ncols     # padding guard (launches round
        a, b = idx[ii], idy[jj]                 # ndrange up to the group size)
        ia, ca = div(a - 1, 3) + 1, mod(a - 1, 3)   # cell index / component
        jb, cb = div(b - 1, 3) + 1, mod(b - 1, 3)
        rx = X[1, ia] - Y[1, jb]
        ry = X[2, ia] - Y[2, jb]
        rz = X[3, ia] - Y[3, jb]
        r2 = rx * rx + ry * ry + rz * rz
        val = 0.0
        if r2 >= 1e-24                          # self term: zeroed (Newell's
            Rca = ca == 0 ? rx : (ca == 1 ? ry : rz)   # analytic cell integral
            Rcb = cb == 0 ? rx : (cb == 1 ? ry : rz)   # regularizes it in real
            delta = ca == cb ? 1.0 : 0.0               # micromagnetics)
            val = -(3.0 * Rca * Rcb - r2 * delta) / (4π * r2^2.5)
        end
        out[ii, jj] = val
    end
end

# Lazy kernel: holds the coordinates on the *active backend* (kernel_array
# moves them to the GPU when one is active) plus a host copy for the dense
# reference below.
struct DipolarDemagGPU <: AbstractMatrix{Float64}
    Xd::AbstractMatrix{Float64}     # target coordinates, 3 x N, active backend
    Yd::AbstractMatrix{Float64}     # source coordinates, 3 x N, active backend
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
    return -(3.0 * Rc * Rd - r2 * delta) / (4π * r2^2.5)
end

# Batch entry — the only form the ACA assembly queries. Output and index sets
# are created on the active backend, the block is evaluated by the kernel
# above, and the result is returned to the host (assembly consumes blocks on
# the host; a real large-scale application could keep them on the device).
function Base.getindex(K::DipolarDemagGPU, I::AbstractVector{Int}, J::AbstractVector{Int})
    out = HMatrixGPU.create_zeros(Float64, length(I), length(J))
    Id = HMatrixGPU.to_backend(out, collect(I))
    Jd = HMatrixGPU.to_backend(out, collect(J))
    kernel! = eval_dipolar_block!(HMatrixGPU.default_backend[], 256)
    kernel!(out, K.Xd, K.Yd, Id, Jd, length(I), length(J);
            ndrange=(length(I), length(J)))
    return Array(out)       # Array() also synchronizes on GPUs
end

K = DipolarDemagGPU(HMatrixGPU.kernel_array(pts), HMatrixGPU.kernel_array(pts), pts)
println("  kernel coordinates live on: ", typeof(K.Xd))

# ---- cluster trees: dims=3 puts whole cells (3 DOF) in every tree node -------
Xc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)
Yc = ClusterTree(pts; max_points_per_leaf=64, dims=dims)

# ---- build (lazy K always assembles through the CPU ACA; the factors land on
#      the active backend, so the matvec below runs on the GPU when active) ----
t_asm = @elapsed H = HMatrix(K, Xc, Yc; eta=eta, eps=eps,
                             row_block_size=dims, col_block_size=dims)
st = info(H)

# ---- validate against the dense reference ------------------------------------
# Kdense is 3N x 3N = 288 MB at the default N — O(N²) materialization is only
# feasible at example sizes.
Kdense = Matrix(K)
x = rand(MersenneTwister(42), 3 * N)
xg = HMatrixGPU.create_zeros(Float64, size(K, 2)); copyto!(xg, x)
yg = Array(H * xg)              # Array() also synchronizes
t_mv = best_of(() -> Array(H * xg), 20)
relerr = norm(Kdense * x - yg) / norm(Kdense * x)
relerr < 1e-5 || error("validation failed: relerr=$relerr")

# ---- report -------------------------------------------------------------------
backend_name = HMatrixGPU.default_backend[] isa KernelAbstractions.CPU ? "CPU" : "CUDA"
@printf("[demag-device] N=%d (%dx%d) backend=%s eta=%g eps=%g\n",
        N, 3 * N, 3 * N, backend_name, eta, eps)
@printf("  assembly %.1fs | leaves %d (near %d / approx %d) | rank %d-%d | compression %.1fx\n",
        t_asm, st["leaves"], st["full_leaves"], st["admissible_leaves"],
        st["min_rank"], st["max_rank"], st["compression_ratio"])
@printf("  matvec %.2f ms (best of 20) | relerr %.1e  [PASS]\n", 1e3 * t_mv, relerr)
