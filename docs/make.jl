using FiniteElementContainers
using Documenter

DocMeta.setdocmeta!(FiniteElementContainers, :DocTestSetup, :(using FiniteElementContainers); recursive=true)

makedocs(;
    modules=[FiniteElementContainers],
    authors="Craig M. Hamel <cmhamel32@gmail.com> and contributors",
    repo="https://github.com/cmhamel/FiniteElementContainers.jl/blob/{commit}{path}#{line}",
    sitename="FiniteElementContainers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cmhamel.github.io/FiniteElementContainers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cmhamel/FiniteElementContainers.jl",
    devbranch="main",
)
