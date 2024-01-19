# push!(LOAD_PATH, "../src/")
using Documenter
using Exodus
using FiniteElementContainers

# DocMeta.setdocmeta!(FiniteElementContainers, :DocTestSetup, :(using FiniteElementContainers); recursive=true)

makedocs(;
    modules=[
        FiniteElementContainers,
        Base.get_extension(FiniteElementContainers, :FiniteElementContainersExodusExt)
    ],
    authors="Craig M. Hamel <cmhamel32@gmail.com> and contributors",
    repo="https://github.com/Cthonios/FiniteElementContainers.jl/blob/{commit}{path}#{line}",
    source="src",
    sitename="FiniteElementContainers.jl",
    format=Documenter.HTML(;
        repolink="https://github.com/Cthonios/FiniteElementContainers.jl",
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cthonios.github.io/FiniteElementContainers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"            => "index.md",
        "Assemblers"      => "assemblers.md",
        "Connectivities"  => "connectivities.md",
        "DofManager"      => "dof_manager.md",
        "Fields"          => "fields.md",
        "Formulations"    => "formulations.md",
        "Function spaces" => "function_spaces.md",
        "Meshes"          => "meshes.md"
    ],
    # delete later
    checkdocs=:none
)

deploydocs(;
    repo="github.com/Cthonios/FiniteElementContainers.jl",
    devbranch="main",
)
