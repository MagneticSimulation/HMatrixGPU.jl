using HMatrixGPU
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(HMatrixGPU, :DocTestSetup, :(using HMatrixGPU); recursive=true)

makedocs(;
    modules=[HMatrixGPU],
    authors="Weiwei Wang",
    repo=Remotes.GitHub("MagneticSimulation", "HMatrixGPU.jl"),
    sitename="HMatrixGPU.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MagneticSimulation/HMatrixGPU.jl.git",
    devbranch="main",
)
