using HMatrixGPU
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(HMatrixGPU, :DocTestSetup, :(using HMatrixGPU); recursive=true)

makedocs(;
    modules=[HMatrixGPU],
    authors="Weiwei Wang",
    repo=Remotes.GitHub("MagneticSimulation", "HMatrixGPU.jl"),
    sitename="HMatrixGPU.jl",
    # Phase 1 drafts the v1.0 pages against the pre-v1.0 code: docstrings of
    # symbols that v1.0 removes (e.g. the global backend API) or of internal
    # helpers outside the v1.0 API page would fail the exported-surface check.
    # Restore the default checkdocs once the v1.0 code and its docstrings land.
    checkdocs=:none,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Manual" => "manual.md",
        "Examples" => "examples.md",
        "Internals" => "internals.md",
        "API reference" => "api.md",
        "Known issues" => "known-issues.md",
    ],
)

deploydocs(;
    repo="github.com/MagneticSimulation/HMatrixGPU.jl.git",
    devbranch="main",
)
