using HMatrixGPU
using Documenter

DocMeta.setdocmeta!(HMatrixGPU, :DocTestSetup, :(using HMatrixGPU); recursive=true)

makedocs(;
    modules=[HMatrixGPU],
    authors="Weiwei Wang",
    repo="https://github.com/ww1g11/HMatrixGPU.jl/blob/{commit}{path}#{line}",
    sitename="HMatrixGPU.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
