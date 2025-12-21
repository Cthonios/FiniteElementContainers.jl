# push!(LOAD_PATH, "../src/")
# using Adapt
using FiniteElementContainers
using Documenter

# DocMeta.setdocmeta!(Adapt, :DocTestSetup, :(using Adapt); recursive=true)
DocMeta.setdocmeta!(FiniteElementContainers, :DocTestSetup, :(using FiniteElementContainers); recursive=true)

makedocs(;
    # modules=[Adapt, FiniteElementContainers],
    # modules=[
    #     # Adapt,
    #     FiniteElementContainers,
    #     isdefined(Base, :get_extension) ? Base.get_extension(FiniteElementContainers, :FiniteElementContainersAdaptExt) :
    #     FiniteElementContainers.FiniteElementContainersAdaptExt
    # ],
    modules=[FiniteElementContainers],
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
        "Home"                => "index.md",
        "Tutorial"            => [
            "1 Poisson Equation"             => "tutorials/1_poisson_equation.md",
            "2 Advection-Diffusion Equation" => "tutorials/2_advection_diffusion_equation.md",
            "3 Coupled Problem"              => "tutorials/3_coupled_problem.md",
            "4 Transient Problem"            => "tutorials/4_transient_problem.md",
            "5 Solid Mechanics"              => "tutorials/5_solid_mechanics.md"
        ],
        "Assemblers"          => "assemblers.md",
        "Boundary Conditions" => "boundary_conditions.md",
        "DofManager"          => "dof_manager.md",
        "Fields"              => "fields.md",
        "Formulations"        => "formulations.md",
        "Function spaces"     => "function_spaces.md",
        "Functions"           => "functions.md",
        "Meshes"              => "meshes.md",
        "Parameters"          => "parameters.md",
        "Physics"             => "physics.md"
    ],
)

deploydocs(;
    repo="github.com/Cthonios/FiniteElementContainers.jl",
    devbranch="main",
)
