# Internals: design of the stable parts

This page documents the parts of the implementation that are considered
**stable and standard** — the pieces a contributor should understand before
changing anything, and the invariants they must preserve. The design follows
the classical H-matrix construction (Börm–Grasedyck–Hackbusch 2003; Bebendorf
2000; Grasedyck 2005); where a deliberate deviation exists it is called out.

```@meta
CurrentModule = HMatrixGPU
```

## Overview

```
coordinates ──► ClusterTree ──► BlockTree ──► build_matrices ──► build_csr_hmatrix ──► HMatrix
   (points)      geometric       adaptive       ACA+ per block      three CSR ops      matvec
                 bisection       single-side    look-ahead,          (Int32 indices,     kernels
                                 split          grouped, error-      cluster-position    (warp/row)
                                                controlled           columns)
```

Everything downstream of the cluster tree works on *cluster-ordered* indices;
the two permutations (`target_index_map`, `source_index_map`) translate between
cluster order and the original ordering at the boundaries of the computation.

## Cluster tree (`src/tree.jl`)

`ClusterTree` is a **geometric KD-tree** over the point coordinates:

- Each node holds a contiguous range of the `index_map` permutation
  (`start_idx` inclusive, `end_idx` exclusive) plus the cluster `center`
  (mean of the points) and `radius` (largest distance from the center to a
  corner of the node's bounding box).
- A node with more than `max_points_per_leaf` points is split along the
  **longest axis of its bounding box at the box midpoint**; points strictly
  below the midpoint go to the left child, the rest to the right child.
- The tree stores the `d × N` `coordinates` it was built from. Point-set
  inputs are normalized to that layout by `_point_matrix`: matrices pass
  through by reference, vectors of points are `stack`ed once. No numeric
  preprocessing happens anywhere — the coordinates are used as given.

**Invariants (must hold after any change):**

1. Both children of an internal node are non-empty. This is guaranteed by the
   midpoint rule (`min < midpoint ≤ max` on the split axis) and by turning
   nodes whose points are *all identical* (zero extent on every axis) into
   leaves — such nodes cannot be subdivided.
2. The leaf check (`end - start ≤ max_points_per_leaf`) runs **before** any
   bounding-box computation: empty or small clusters must never reach code
   that reduces over the points.
3. `sort(index_map) == collect(1:N)`: the map is a permutation.

The `dims` keyword handles vector-valued DOFs: with `dims = d` the index map
expands so that the `d` components of point `p` occupy consecutive positions
(`d*p-d+1 ... d*p`), and every node range is rescaled to cover the expanded
indices. The geometry (center/radius) still refers to the point positions.
After any `dims = d` construction, all cluster ranges are multiples of `d` —
this is what makes grouped `d × d` ACA pivoting safe.

Deliberate deviation from the textbook: the split plane is the box midpoint
(standard geometric bisection) rather than the coordinate mean — the mean can
be pulled aside by outliers and, for degenerate point sets, used to leave one
side empty (the bug this rule replaced).

## Block tree (`src/block.jl`)

`BlockTree` pairs a target cluster with a source cluster and decides
**admissibility**:

```
admissible  ⟺  dist(c_X, c_Y) > η · (r_X + r_Y)
```

with the center/radius taken from the cluster trees. An inadmissible pair is
split along the side with the **larger radius** (single-side split; if one side
is already a leaf, the other is split), recursively, until either the
admissibility test passes or both sides are leaves — those become near-field
(dense) blocks.

Two standard refinements are implemented:

- `merge_dense_matrices!` merges every fully-dense subtree into a single
  larger near-field block (fewer segments per matrix row, which the CSR layout
  benefits from directly).
- The single-side adaptive split is an accepted variant of the block cluster
  tree (the symmetric both-sides split of the classical construction is the
  other variant); it keeps the near-field blocks smaller for elongated
  geometries.

There is deliberately **no** `is_leaf` shortcut on admissibility for the root:
the whole matrix is never treated as one low-rank block.

## The three CSR operators (`src/hmatrix.jl`)

The compressed matrix stores everything in three flat CSR structures on the
landing backend:

| operator | rows | columns | content |
| :--- | :--- | :--- | :--- |
| near `D` | matrix rows (cluster order) | near-field columns (cluster positions) | the dense blocks |
| far `V` | rank rows `L = Σ rank` | far-field columns (cluster positions) | the `V` factors, one row per rank row |
| far `U` | matrix rows (cluster order) | rank rows `1:L` | the `U` factors |

Design points that must be preserved:

- **Column indices are cluster positions** (`Int32`). The input vector is
  permuted into cluster order once per product (`x_buffer`), so every kernel
  access to `x` is sequential; nothing else needs the original ordering.
- Rows of all three operators are in cluster order; the fused kernel scatters
  the result through `target_index_map` at the end.
- The near-field row sets and the `U` row sets **partition** the matrix rows
  (each row appears in exactly one near-field position per overlapping block,
  and all its blocks are visited); combined with `target_index_map` being a
  permutation this means the fused kernel writes every entry of `y` **exactly
  once** — no atomics, no zero-filling, no race conditions.
- Index arrays are `Int32` (halves index traffic of the memory-bound matvec);
  `build_csr_hmatrix` guards against overflow.
- Assembly (`build_csr_hmatrix`) is a two-pass count-then-fill over the block
  lists, so the CSR row lengths need not be known in advance.

## Matrix–vector kernels (`src/mult.jl`)

Three kernels, all *workgroup-per-row* with a fixed group size of 32 (one warp
on CUDA):

1. `permute_to_cluster!` — one thread per DOF, coalesced gather.
2. `csr_mul_vec_warp!` — one warp per rank row: the 32 threads stride through
   the row, partial sums land in a 32-slot static shared-memory array, and a
   `for s in (16, 8, 4, 2, 1)` loop reduces it.
3. `near_u_mul_vec_warp!` — one warp per matrix row: strides through the
   near-field and `U` segments of the row, reduces both, thread 1 scatters
   through `tmap`.

Rules any new kernel must follow:

- **No unguarded global access.** Every kernel checks its own index
  (`i <= length(...)`) before touching data, so padding threads from a
  non-divisible ndrange can never fault. (The ndrange of the warp kernels is
  `32 × rows`, always divisible by the group size.)
- **No state that outlives a guarded segment.** KernelAbstractions hoists
  statements into active-lane-guarded segments; a variable assigned inside a
  guarded segment and read outside it is undefined on padding threads. Loop
  counters for device-side reductions must live in `for` loops, not
  `while` loops with a hoisted counter.
- **No shared memory, no barriers, no atomics.** The workgroup reduction uses
  static shared memory but no cross-workgroup synchronization; correctness
  never depends on warp-synchronous execution.
- Row arithmetic is identical on every backend: with the same factors, the CPU
  and GPU products are bit-identical (verified by the test suite).

## Cross approximation (`src/aca.jl`)

`ACA_plus` is an ACA+ variant with look-ahead pivoting (Grasedyck 2005,
Construction 2.4) generalized to *grouped* pivoting:

- One reference row group `r` and one reference column group `c` are drawn
  from a per-block deterministic RNG (seeded from the block dimensions), so
  repeated constructions of the same matrix are bit-identical.
- Each step picks the row-group pivot from the reference *column* residual and
  the column-group pivot from the reference *row* residual (look-ahead),
  completes the cross through the larger candidate, and refreshes the
  reference when a pivot coincides with it. This is immune to blocks whose
  first rows are zero to machine precision (the double-layer kernel on
  coplanar surfaces), which defeat plain partial pivoting.
- With `row_block = col_block = d > 1` (vector-valued kernels) the pivot block
  is `d × d` and one step captures all components of a cell pair together;
  `pinv(P)` supports rectangular groups. Scalar problems use `d = 1`, where
  the update reduces to the classical `a ⊗ b/d`.
- The residual is always computed in `Float64` regardless of the query
  precision; the caller converts the factors to `eltype(K)`.

Stopping and storage:

- **Relative stopping**: a term is accepted while its norm exceeds `eps` times
  the block Frobenius norm estimated from the spread sample.
- **Storage crossover cap**: the group rank stops at `ceil(m·n/(m+n))` — the
  rank where the factors would cost as much as the dense block. Blocks that
  hit the cap without converging, or that fail a sampled residual check, are
  reported with `converged = false` and stored **densely** by the caller:
  accuracy never depends on a half-converged factorization.
- **Significance guards**: pivots below `pivot_tol · scale` (scale = running
  maximum over sampled and computed residuals) are treated as zero, with
  dead-group bookkeeping — this is what protects the factorization on blocks
  containing rows that are zero to machine precision.

`SVD_recompress` truncates the factors with a *relative* Frobenius tolerance
(`eps/10` by the caller's layering), which is the optimal truncated
factorization at that tolerance.

## GPU dense assembly (`src/assembly_gpu.jl`)

For a **dense host `K::Matrix`** on a non-CPU backend whose device SVD works,
far-field assembly takes a batched device path (`build_matrices_gpu_dense`);
otherwise it falls back to the per-block CPU ACA. The capability probe
`device_svd_available(B)` runs one tiny SVD on the backend once per backend
type and caches the verdict — the probe is also what keeps vendor LAPACK out
of the package's own dependency graph.

Algorithm per far-field block `(rows, cols)` (the whole `K` and both index
maps are uploaded to the device once):

1. **Gather** the block from the uploaded matrix with the `gather_block!`
   kernel (device ranges of the two cluster maps bound the block; per-block
   emission, host scalars pass the ranges).
2. **One-sided randomized range finder**: draw the `n × l` test matrix `Ω`
   from `MersenneTwister(bi)` (seeded by the block index — repeated
   assemblies of the same matrix are bitwise identical, the same invariant as
   the CPU ACA path), project `Y = B·Ω`, and orthogonalize through the small
   `l × l` Gram matrix of the test projection (host eigensolve). This yields
   an orthonormal range basis with only **one device SVD per block** (Halko
   et al.; the randomized range approximation of Dölz et al.).
3. **Second Gram pass** (CholeskyQR2-style): the first pass loses up to
   `O(eps·κ)` orthogonality on fast-decaying spectra; re-orthogonalizing `Q`
   through `Q'Q` restores it.
4. **Range cutoff tied to `eps`**: eigen-directions below
   `λ_cut = max((eps/50)², 1e-16)` are dropped. The `(eps/50)²` tie keeps the
   range below the `eps/10` truncation threshold at every supported
   tolerance; the `1e-16` floor keeps the first Gram pass inside its
   stability bound (`κ(G) = 1/λ_cut ≲ 1e16`). The l-vector size is
   `crossover + 16` where the crossover is the storage crossover `m·n/(m+n)`.
5. **Truncation and dual guard**: the projected block `Q'B` gets the single
   device SVD; the rank `r` truncates the relative Frobenius tail at `eps/10`.
   The block stays **dense** when either guard fires: `S2h[end] >
   (eps/10)·‖B‖_F` (the range could not resolve the truncation — ε-rank
   beyond what the range finder captured) or `(r+1)·(m+n) ≥ m·n` (no storage
   win over the dense block). Both guards feed the same near-field list, so
   accuracy never rests on a half-resolved range.
6. **Zero-block drop**: a numerically zero block (`l1 == 0`) is dropped
   entirely; its rows stay covered by the near/U row-set partition, so the
   fused matvec kernel still writes each of them exactly once (as zero).
7. **Factors**: `U = Q·(U₂[:,1:r]·S₂[1:r]')` (`m × r`, materialized) and
   `V = view(V₂, 1:n, 1:r)` — the SVD's right factor is already in far-CSR
   row order (column j = rank row j), so packing broadcasts it without a
   transpose.

If more than half of the far blocks hit the dense guards (tight `eps`), the
assembly emits a one-time warning (`maxlog = 1`); see
[known issues](known-issues.md). `row_block_size`/`col_block_size` do not
apply on this path — the randomized SVD needs no pivoting and is immune to
component anisotropy.

**Packing** (`build_csr_hmatrix_gpu`): the near-field values are gathered
from the uploaded `K` by the `fill_near_data!` kernel; each low-rank block's
`U` factor is written into the CSR `u_data` array in **block-row order** — a
host-side `useg` table records, per matrix row, the `u_data` position the
row's factor rows occupy, and `fill_u_block!` scatters the device factor into
exactly those positions. No intermediate dense assembly is ever materialized
on the host.

## Backend resolution

The package keeps no backend state; every constructor resolves the landing
backend through `_resolve_landing(like, backend, datas...)`:

1. `like=` — the backend of the given array;
2. `backend=` — a backend object, or a name resolved by `backend_from_name`;
3. the device of the primary data (skipping host arrays; mixed devices error);
4. `CPU()`.

`backend_from_name` maps names through a small table (`"cuda"`/`"nvidia"` →
CUDA, `"amd"`/`"roc"`/`"amdgpu"` → AMDGPU, `"oneapi"`/`"intel"` → oneAPI,
`"metal"`/`"apple"` → Metal, `"cpu"` → CPU) and looks up the *already-loaded*
vendor module via `Base.loaded_modules` — a `Vector{PkgId}` on Julia ≤ 1.11
and a `Dict{PkgId,Module}` on ≥ 1.12, handled by `_loaded_vendor_module`.
The backend constructors live in the vendor packages (`CUDA.CUDABackend()`,
...), not in the KernelAbstractions namespace, so no vendor type is ever
referenced by the package itself. The resolver is deliberately strict: an
unknown name, a vendor package that is not loaded, or a loaded package
without a functional device are all hard errors — there is no soft probing
and no auto-detection anywhere in the package; callers that want a fallback
`try`/`catch` around it themselves.

`move_to_backend(B, a)` uploads a host array to `B` (a no-op when it already
lives there); `to_backend(like, a)` is its data-following convenience form.
The matvec kernels read the backends off the arrays they are given — no
global is consulted anywhere in the product path.

## What may change

The cluster/block trees, the CSR layout, the kernel scheme and the ACA+
algorithm are considered settled. Expected areas of future work (see the
project roadmap): device-resident dense `K` on the GPU dense fast path,
Float32 performance paths, transpose/adjoint products, and the `H²` /
nested-basis variant.
