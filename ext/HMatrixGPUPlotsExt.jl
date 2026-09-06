module HMatrixGPUPlotsExt

using HMatrixGPU: HMatrix, hmatrix_blocks
import HMatrixGPU: plot_hmatrix
using Plots

# fill colors: teal = dense (near field), amber = low-rank (far field)
const DENSE_FILL = "#2a9d8f"
const LOWRANK_FILL = "#f4a261"
const EMPTY_FILL = "#f5f5f5"

# rasterize rectangles into a height x width cell grid (0 empty, 1 dense, 2 low-rank)
function _block_grid(dense, lowrank, m, n; width, height)
    grid = zeros(Int, height, width)
    paint = (rects, val) -> begin
        for rect in rects
            r, c = rect[1], rect[2]
            g1 = clamp(fld((first(r) - 1) * height, m) + 1, 1, height)
            g2 = clamp(cld(last(r) * height, m), 1, height)
            h1 = clamp(fld((first(c) - 1) * width, n) + 1, 1, width)
            h2 = clamp(cld(last(c) * width, n), 1, width)
            grid[g1:max(g1, g2), h1:max(h1, h2)] .= val
        end
    end
    paint(dense, 1)
    paint(lowrank, 2)
    return grid
end

"""
    plot_hmatrix(H::HMatrix; nx=600, ny=600, kwargs...) -> Plots.Plot

Bare heatmap of the block distribution: teal = dense (near-field) blocks,
amber = low-rank (far-field) blocks; the origin (0, 0) is the bottom-left
corner (matrix entry (1, 1) sits there) — no title/axis labels, so a caption
can be provided externally. `nx`/`ny` set the raster resolution; extra
`kwargs...` go to `Plots.heatmap` (e.g. `size`, `dpi`, `title`). Save the
result with `savefig(p, "hmatrix_pattern.png")`.
"""
function plot_hmatrix(H::HMatrix; nx=600, ny=600, kwargs...)
    m, n = size(H)
    dense, lowrank = hmatrix_blocks(H)
    grid = _block_grid(dense, lowrank, m, n; width=nx, height=ny)
    xs = range(0, n; length=nx)            # axis in matrix (cluster) indices
    ys = range(0, m; length=ny)
    heatmap(xs, ys, grid; aspect_ratio=:equal,
            xlims=(0, n), ylims=(0, m),
            color=cgrad([EMPTY_FILL, DENSE_FILL, LOWRANK_FILL]; categorical=true),
            clims=(-0.5, 2.5),
            colorbar_ticks=([0, 1, 2], ["empty", "dense", "low-rank"]),
            kwargs...)
end

end
