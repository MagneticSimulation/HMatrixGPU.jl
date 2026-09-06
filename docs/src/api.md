# API reference

Public interface of HMatrixGPU.jl. Docstrings of the shipped code are rendered
below; every entry also appears in context on the [manual](manual.md) page.

```@meta
CurrentModule = HMatrixGPU
```

## Trees

```@docs
ClusterTree
```

## Kernel matrix (lazy, matrix-free)

<!-- PHASE2: @docs KernelMatrix -->
`KernelMatrix(g, targets, sources; dims = 1)` wraps a kernel function `g` into
a lazy `AbstractMatrix{Float64}` with the entry contract
`K[i,j] = g(x_i, y_j)` (coordinates, not indices) and the interleaved DOF
layout `K[d(p-1)+c, d(q-1)+e] = g(x_p, y_q)[c, e]` for `dims = d`. It is the
lazy kernel behind the high-level [`HMatrix`](@ref) constructors and can also
be held and passed around on its own (see the [manual](manual.md)).

## Hierarchical matrix

```@docs
HMatrix
```

<!-- PHASE2: @docs HMatrix 构造器族(任务5:函数核 + 点集 / 矩形 / 函数核 + 自定义树 / 自带 lazy K + 点集;签名与契约措辞以 API_DESIGN §4.2 为准)。
     Phase 1 不能渲染基线的低层构造器 docstring:它对 v1.0 已删除的全局
     backend API 有一处 @ref(无法解析),任务4/5 会重写该契约段。 -->

The high-level constructors are documented on the
[manual](manual.md) page (kernels, point sets, `dims`).

## Products

```@docs
mul!
*
```

## Diagnostics

```@docs
info
sparsify_hmatrix
```

## Backends and utilities

```@docs
to_backend
```

<!-- PHASE2: @docs backend_from_name -->
<!-- PHASE2: @docs move_to_backend -->
<!-- PHASE2: @docs create_zeros -->
<!-- PHASE2: @docs create_ones -->
<!-- PHASE2: @docs kernel_array -->
<!-- PHASE2: @docs set_groupsize -->
<!-- PHASE2: @docs @using_gpu -->

The backend resolution rules (the `like=` > `backend=` > data > `CPU()`
chain, strict name resolution, no global state) are described on the
[manual](manual.md) page; `HMatrixGPU.backend_from_name(name)` is the strict
name resolver the `backend=` keyword uses.

## Low-rank approximation (internal, documented for reference)

`ACA_plus` is the adaptive cross approximation used for every low-rank block
on the CPU assembly path. It is an internal algorithm — documented here for
reference and because the test suite relies on it.

```@docs
ACA_plus
```
