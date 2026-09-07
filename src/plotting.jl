# Block-pattern plotting through RecipesBase: any recipe-aware plotting
# backend (e.g. Plots.jl) renders an HMatrix directly with `plot(H)` —
# no package extension needed, and loading HMatrixGPU pulls no plotting code.

# fill colors: teal = dense (near field), amber = low-rank (far field)
const DENSE_FILL = "#2a9d8f"
const LOWRANK_FILL = "#f4a261"

# `plot(H)` renders the block structure: teal = dense (near-field) blocks,
# amber = low-rank (far-field) blocks, origin (0, 0) at the bottom left
# (matrix entry (1, 1) sits there); no title/axis labels so a caption can be
# provided externally, `kwargs...` go to the backend (size, dpi, ...).
@recipe function f(H::HMatrix)
    m, n = size(H)
    dense, lowrank = hmatrix_blocks(H)
    legend --> false
    grid --> false
    aspect_ratio --> :equal
    xlims --> (0, n)
    ylims --> (0, m)
    seriestype := :shape
    linecolor --> :black
    linewidth --> 0.5
    for (rects, fillc, lab) in ((dense, DENSE_FILL, "dense (near field)"),
                                (lowrank, LOWRANK_FILL, "low-rank (far field)"))
        for (i, rect) in enumerate(rects)
            r, c = rect[1], rect[2]
            @series begin
                fillcolor --> fillc
                label --> i == 1 ? lab : ""
                x := [first(c) - 1, last(c), last(c), first(c) - 1]
                y := [first(r) - 1, first(r) - 1, last(r), last(r)]
                ()
            end
        end
    end
end
