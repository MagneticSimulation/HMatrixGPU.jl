# Manual

```@meta
CurrentModule = HMatrixGPU
```

```@contents
Depth = 2
```

## Overview

The package compresses a dense kernel matrix `K` into an [`HMatrix`](@ref) and
computes `y = H * x` with workgroup-per-row device kernels. `K` can be provided
in two ways:

- **High-level (recommended)** — you write the kernel as a *function*
  `g(x, y)` of two point coordinates and hand the library the points. The
  library clusters the points, wraps `g` into a lazy matrix (`KernelMatrix`),
  and drives the whole assembly. The bookkeeping that is normally boilerplate
  (a custom `AbstractMatrix` struct, `size`, `getindex`, cluster trees, block
  sizes) is gone.
- **Explicit low-level** — you provide `K` yourself as any `AbstractMatrix`
  and build the trees yourself. This mode is what the high-level
  mode is a thin layer over, and stays the right tool when you need index-level
  control, custom storage (e.g. `Float32`), or device-side block evaluation.

Either way the result is the same [`HMatrix`](@ref): a single CSR-resident
structure whose matvec runs on every backend.

## Point sets

Every point-set entry — [`ClusterTree`](@ref), `KernelMatrix`, and the
high-level [`HMatrix`](@ref) constructors — accepts both forms:

- a **`d × N` matrix** (one point per column). It is used as-is, by reference —
  no copy, no conversion. This is the native format of MicroMagnetic.jl and of
  all existing tests and examples.
- a **vector of points** — each element is one point: a d-element tuple,
  `Vector`, or `SVector`. It is converted once, at construction, into the
  `d × N` layout (`stack`, each element becomes one column). The orientation
  is unambiguous.

Internally everything is `d × N`; the two forms produce bit-identical trees
and matrices.

**Coordinates are used exactly as given**: the library performs no numerical
preprocessing — no scaling, no shifting, no normalization. The geometry
reaches the kernel function untouched, and the achieved accuracy does not
depend on the scale of the kernel (that is what the *relative* tolerance
`eps` guarantees).

`N × d` input (one point per *row*) is not accepted: in column-major storage
a single point's coordinates would not be contiguous, it breaks the cluster
tree contract and the ecosystem convention (tdes, HMatrix.jl, MicroMagnetic.jl
are all `d × N`), and for square matrices the orientation is undecidable. If a
matrix input looks transposed (row count ≥ 10× the column count, few columns),
construction emits a warning once — a hint, nothing is converted:

```text
┌ Warning: point coordinates should be d x N (one point per column); got 2000 x 3 — possible transposed input
└ @ HMatrixGPU
```

Construction convention: write point vectors, or `rand(d, N)` for random
geometries.

## Kernels

### The contract

The high-level entry builds the matrix from a function. The contract is one
sentence: **`K[i, j] = g(x_i, y_j)` — `g` receives the *coordinates* (column
views) of the i-th target point and the j-th source point, not indices.**

```julia
# named function ...
g(x, y) = (d = norm(x - y); d < 1e-12 ? 0.0 : -log(d)/(2π))
H = HMatrix(g, pts; eta = 1.5, eps = 1e-6)

# ... and the do-block form are exactly equivalent
H = HMatrix(pts; eta = 1.5, eps = 1e-6) do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)
end
```

`x` and `y` inside the block are d-element column views of the two point sets
(no copy; the values are exactly the coordinates you passed in, in whatever
form you passed them).

### Vector kernels: `dims` and the DOF layout

For a vector-valued kernel (`g` returns a `d × d` matrix, `dims = d`), each
point carries `d` degrees of freedom, and the matrix is the `dN × dN` operator
with the **interleaved layout** (a single global convention for the whole
package):

```text
K[d*(p-1) + c, d*(q-1) + e] = g(x_p, y_q)[c, e]      (dims = d)
K[p, q]                     = g(x_p, y_q)            (dims = 1)
```

so the DOFs of each point occupy consecutive positions (`x₁,y₁,z₁, x₂,y₂,z₂,
...` for `d = 3`) and matvecs take flat vectors in and out — the same layout
the CSR kernels use. You declare `dims` once, at construction; the trees, the
grouped pivoting (below) and the matrix layout all follow.

### `g` must be a total function

The library evaluates `g` on the point pairs the block tree asks for; it never
inspects the geometry. Singular pairs (coincident target and source points)
are the kernel's responsibility — regularize them inside `g`:

```julia
g(x, y) = (R = x - y; r2 = dot(R, R);        # demag tensor, dims = 3
           r2 < 1e-24 ? zeros(3, 3) : -(3.0 * R * R' - r2 * I) / (4π * r2^2.5))
```

The library does not guess a regularization for you (explicit is better than
implicit), but you are not on your own either: a **construction-time probe**
calls `g` once on an off-diagonal pair (first target × last source, chosen to
avoid singular self-pairs) and checks the return value against `dims` — a
scalar for `dims = 1`, a `d × d` matrix for `dims = d`. A mismatch errors
immediately at construction instead of blowing up in the middle of an
assembly.

## The two API tiers and when to leave the high-level one

The high-level constructors do the bookkeeping: trees, the lazy kernel
wrapper, block sizes that follow `dims`, the type probe. The bookkeeping has a
small price: measured on the example gallery, the thin wrapper adds ≈ 30% to
the assembly time (every block query travels through `KernelMatrix`'s batched
`getindex`, which calls `g` once per point pair on host views). Ranks,
compression and products are bit-identical to the equivalent low-level
construction — only the assembly time differs. The low-level constructor

```julia
HMatrix(K::AbstractMatrix, X::ClusterTree, Y::ClusterTree; kwargs...)
```

takes `K` as *you* define it — any `AbstractMatrix` with any
internal representation, storage eltype, and evaluation strategy — and is the
right tool when that price matters or when you need:

- **index-level control** — e.g. a kernel whose entries are not a function of
  two coordinates (mesh-based integral kernels with their own DOF layout);
- **custom storage precision** — a kernel with `eltype(K) = Float32` stores
  its factors in `Float32` (assembly always computes in `Float64`);
- **device-side block evaluation** — a batched `getindex` that launches a
  KernelAbstractions kernel on the GPU, so ACA block queries never
  materialize on the host; `examples/hmatrix_vector.jl` is the complete
  worked example of this pattern. When writing such a kernel, evaluate one
  *cell pair* per work item (with `dims = d` and the default block sizes the
  queries arrive as whole cells, so the geometry is computed once and the
  whole `d × d` block written), and decompose fractional powers by hand —
  `r⁵ = r²·r²·√r²` costs two multiplications and one hardware `sqrt`, while
  `r2^2.5` compiles to a software `pow` sequence.

The high-level mode deliberately does *not* offer an index-form do-block
(`do i, j`): it would be indistinguishable in the signature from the
coordinate form, and passing indices re-opens the point-set orientation trap
that the coordinate form closes.

## Trees

[`ClusterTree`](@ref) partitions the points by recursive geometric bisection:
a node with more than `max_points_per_leaf` points is split along the longest
axis of its bounding box; leaf sizes therefore bound the near-field block
sizes. Typical values: 32–128.

`max_points_per_leaf` controls the tree; `dims` (default 1) declares the DOFs
per point: with `dims = d` the index map is expanded so that each point's `d`
DOFs are consecutive and every node range covers whole points' worth of DOFs.
The tree stores the `d × N` coordinates it was built from (a matrix input is
kept by reference; a point-vector input is converted once), so the high-level
constructors and the diagnostics can reuse them. Trees are host objects:
device point sets are downloaded once at construction — tree construction and
high-level kernel evaluation are host-side algorithms.

## Admissibility, `eps`, and block sizes

Two clusters form an *admissible* pair (eligible for a low-rank
approximation) when

```text
dist(X.center, Y.center) > eta * (X.radius + Y.radius)
```

Smaller `eta` classifies more blocks as admissible: fewer dense operations,
more (and smaller) low-rank blocks. Inadmissible pairs are subdivided until
one side is a leaf; those leaf pairs are the near field (dense blocks).

**`eps` is a relative, per-block tolerance**: the ACA stopping criterion and
the SVD recompression truncate at fractions of each block's Frobenius norm,
so accuracy is independent of the scale of the kernel. The ACA runs one order
tighter than the recompression so the accumulated matvec error stays at the
level of `eps`. Use `1e-6`–`1e-8` when the product feeds a field or energy
computation. Blocks whose ε-rank exceeds the storage crossover `m*n/(m+n)` —
or that fail a sampled residual check — are stored **dense**: accuracy never
depends on a half-converged factorization. Pivoting is deterministic per
block (seeded from the block dimensions), so repeated constructions are
bit-identical.

**ACA block sizes follow `dims` by default.** For vector kernels the ACA
pivots on `d × d` groups (set `row_block_size`/`col_block_size` to override;
the counts must be multiples of them). Grouped pivoting matters for
*anisotropic* kernels: the components of a `3 × 3` demag tensor differ by
orders of magnitude (for a thin film `N_zz` dominates), and scalar per-column
pivoting starves the weak components — one group step captures all components
of a cell pair together, the factors have `k*d` nontrivial columns/rows, and
the accuracy of every component is controlled jointly. With the default in
place a vector user says `dims = 3` once and nothing else; the GPU dense fast
path (below) ignores both block sizes — its randomized SVD needs no pivoting
and is immune to component anisotropy.

## Backends and devices

The landing device of an [`HMatrix`](@ref) is fixed at construction by the
explicit `backend=` keyword (no global default exists):

```text
backend= (name or Backend object)  >  CPU() when no keyword is given
```

- `backend=` accepts a name (`"cuda"`, ...) or a backend object
  (`CUDA.CUDABackend()`); the name is resolved strictly and errors when the
  vendor package is not loaded or has no functional device (see
  [getting started](getting-started.md)).
- Device-resident inputs are legal but never decide the placement: a device
  `K` or point set (`CuArray(pts)`) is downloaded **once** during
  construction (trees and high-level kernel evaluation are host-side
  algorithms), and the matrix lands on `backend=`, else `CPU()`.
- To place an instance next to an existing array, ask for its backend
  explicitly:

```julia
h = HMatrix(K, X, Y; backend = KernelAbstractions.get_backend(x), ...)
```

- What ends up on the device is the three CSR operators, the index maps and
  the product buffers (moved at construction), plus the internal quantities
  of the GPU dense fast path; the host coordinates (`pts`, tree coordinates,
  `KernelMatrix` targets and sources) always stay on the host.

**The matvec has a hard same-device contract**: `x`, `y` and the
[`HMatrix`](@ref) must live on the same device. `x` is never transferred for
you — nothing is moved implicitly, at construction or per call. A mismatch is
detected with a clear error (this check is the safety net for a
mis-requested placement); `H * x` allocates the result on the matrix's
backend. `LinearAlgebra.mul!(y, H, x)` fully overwrites `y`.

Because nothing is global, CPU and GPU instances — different backends, even
different vendors — coexist in one process and can be multiplied in
alternation without interfering.

## Assembly paths and the GPU dense fast path

Which assembly runs depends on `K`, not on the interface tier:

- **Lazy / matrix-free `K`** (a `KernelMatrix`, or any other `AbstractMatrix`
  that only defines `getindex`): always the CPU ACA path — blocks are
  evaluated on demand and factorized by ACA+ with grouped pivoting; the
  factors are then placed on the landing backend. No `eps` restriction.
- **Dense host `K::Matrix` on a GPU backend** with a working device SVD
  (probed once per backend type, at construction): the batched GPU dense
  path — the whole matrix is uploaded once and every far-field block is
  factorized by a single-sided randomized SVD on the device. The path
  compresses normally for `eps ≳ 1e-6`; at tighter tolerances its
  compression degrades and it warns once when more than half of the far
  blocks fall back to dense storage (see [known issues](known-issues.md)).
  If the device-SVD probe fails, the path is not taken and assembly falls
  back to the CPU ACA.
- **Dense device-resident `K`** (e.g. a `CuArray`): not on the fast path —
  it assembles through the CPU ACA, which queries the device matrix
  block by block (works, but slow at scale).

## Diagnostics

`info` returns statistics: for a [`ClusterTree`](@ref) the depth and the
point-count range of the leaves; for an [`HMatrix`](@ref) a `Dict` with
matrix size, near-field/far-field block counts, ACA rank range and the
compression ratio.

`sparsify_hmatrix` (not exported; call as `HMatrixGPU.sparsify_hmatrix`)
returns a `SparseMatrixCSC` containing exactly the near-field blocks of the
block tree — everything that would be stored densely. It accepts any
`AbstractMatrix` kernel, so it composes naturally with `KernelMatrix`.

### Block structure and plotting

`hmatrix_blocks(H)` reconstructs the leaf blocks of the assembled
[`HMatrix`](@ref) from its CSR index arrays — a vector of `(rows, cols)` for
the dense blocks and `(rows, cols, rank)` for the low-rank ones. It works
for GPU-resident matrices too: only the index arrays are downloaded.

A plotting example — build the 2D log-kernel ring matrix and render its
block structure with `plot(H)`. The recipe comes from RecipesBase, so it
works with any recipe-aware plotting backend; here with Plots.jl:

```@example blockplot
using HMatrixGPU, Plots, LinearAlgebra

N = 400
pts = [(sin(2π*i/N), cos(2π*i/N), 0.0) for i in 1:N]
H = HMatrix(pts; eta = 1.5, eps = 1e-6, max_points_per_leaf = 32) do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)
end
nothing #hide
```

Teal = dense (near-field) blocks, amber = low-rank (far-field) blocks,
origin `(0, 0)` at the bottom left (matrix entry `(1, 1)` sits there):

```@example blockplot
plot(H)
```

The plot is deliberately bare — no title, no axis labels — so the figure can
be captioned externally; extra `kwargs...` are forwarded to the backend
(`size`, `dpi`, ...).

## Performance tuning

| Knob | Effect |
| :--- | :----- |
| `max_points_per_leaf` | Leaf size of the cluster tree. Larger leaves → fewer, denser near-field blocks and faster assembly; smaller leaves → deeper trees, more low-rank blocks, better asymptotics. Typical values: 32–128. |
| `eta` | Admissibility. Smaller (e.g. `1.0`) → more blocks go low-rank (less FLOPs in the matvec, more assembly work); larger (e.g. `2.0`) → more dense blocks. Typical values: 1.0–2.0. |
| `eps` | *Relative* per-block approximation tolerance (see above): accuracy is independent of the kernel scale. Use `1e-6`–`1e-8` for field/energy computations. |
| `svd_recompress` | SVD recompression trades assembly time for lower ranks (faster matvec, less memory). Disable with `svd_recompress = false` when assembly time dominates. |
| `row_block_size` / `col_block_size` | ACA group sizes; default to the trees' `dims` (grouped `d × d` pivoting for vector kernels). Override only for special DOF layouts. Ignored on the GPU dense path. |
| `HMatrixGPU.set_groupsize(n)` | Workgroup size of the permutation kernel of the matvec (default 512). A global performance knob that only affects launch configuration — it never changes results; the CSR matvec kernels use a fixed 32-thread workgroup per row. Most users never need to touch it. |
