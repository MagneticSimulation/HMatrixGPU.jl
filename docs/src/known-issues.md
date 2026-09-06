# Known issues and limitations

## ~~GPU matrix–vector product crashes~~ — resolved

**Status: fixed.** The GPU matvec was rewritten (blocks packed into three
Int32 CSR operators — near field, far-field `V` and far-field `U` — multiplied
by fused workgroup-per-row kernels). The full test suite, including the CUDA
matvec test, passes, and the product reaches ~570–770 GB/s of effective
bandwidth on an A100 at N = 20k–50k (about 9.6× faster than the previous
implementation).

For the record, the old crash had two stacked causes:

1. KernelAbstractions executes a kernel body segment under an active-lane
   check, but statements are hoisted per segment — the old kernels' `j::Int =
   8; while j > 0` tree reduction left the counter assignment inside the
   guarded segment and the loop condition outside, so threads beyond the
   `ndrange` hit an undefined variable on the device.
2. Independently of padding threads, the 16/4-lane + shared-memory + `while`
   reduction machinery triggered illegal accesses even when the launch had no
   padding threads, while the underlying block layout and index arithmetic were
   verified correct with plain thread-per-row kernels.

The new kernels avoid this bug class by construction: no shared-memory
barriers, no lane assumptions, explicit index guards, and an ndrange
(`32 × rows`) that is always divisible by the workgroup size.

## GPU dense assembly: supported eps range

For a dense kernel matrix on a GPU backend (with a working device SVD),
far-field assembly takes the GPU fast path (single-sided randomized SVD with
per-block packing). The range-finder cutoff is tied to the block tolerance,
so the path compresses normally for `eps ≳ 1e-6` (at `eps = 1e-6` the FEM–BEM
cube reaches 2.6× compression with a relative error of 3.5e-10). At tighter
tolerances the compression degrades — far blocks fall back to dense storage —
and the assembly emits a one-time warning once more than half of the far
blocks fall back. The path only applies to dense host `K::Matrix` on a GPU
backend: function kernels and other lazy (matrix-free) `K` always use the CPU
ACA path, which has no eps restriction, and a device-resident dense `K` (e.g.
a `CuArray`) is not on the fast path in v1.0 (it assembles through the CPU
ACA, querying the device matrix block by block).

## Float64 only at the high level; Float32 storage at the low level

The **high-level v1.0 interface** — the function-kernel `HMatrix`
constructors and `KernelMatrix` — is `Float64`-only: point sets and the
values `g` returns must be `Float64`.

The **low-level explicit mode** supports `Float32` *storage* end to end:
assemble from a custom kernel with `eltype(K) = Float32` and the factors are
stored in `Float32` (the assembly always computes in `Float64` internally and
converts at storage time), with `Float32` vectors in and out of the matvec —
regression-tested on the CPU and through the GPU dense path (which also
computes in `Float64` internally and stores in `eltype(K)`).

## No transpose/adjoint products

Only the forward product `H * x` is implemented. This is sufficient for the
target application (the FEM–BEM boundary matrix is symmetric), but `H'` or
`transpose(H)` products are not available yet.

## Lazy (matrix-free) assembly is CPU-bound

A function kernel (`KernelMatrix`, the high-level constructors) and any other
lazy `AbstractMatrix` assemble through the CPU ACA: the queried blocks are
evaluated on the host and the resulting factors are placed on the target
backend — the *matvec* runs on the device, the assembly queries do not. The
only device-side piece available for lazy kernels is the block *evaluation*
itself: a custom low-level kernel whose batched `getindex` launches a
KernelAbstractions kernel (the advanced pattern in
[`examples/hmatrix_vector.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/hmatrix_vector.jl)).
A fully device-side assembly exists only for dense host `K::Matrix` (the GPU
dense fast path above).
