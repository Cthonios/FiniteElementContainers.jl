import FiniteElementContainers as FEC
import FiniteElementContainers.AppTools as AT
using Exodus
using FiniteElementContainers

# include files
include("Physics.jl")

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_zero_func(_, _) = 0.0

function app_main(ARGS::Vector{String})
    app = AT.App("MyApp")
    AT.add_cli_arg!(app, "--backend"; default = "cpu")
    sim = AT.setup(app, ARGS)

    #####################################
    # setup function space
    #####################################
    V = FunctionSpace{true}(sim.mesh, H1Field, Lagrange)
    u = ScalarFunction(V, "u")
    dof = DofManager{false}(u)
    sp_type = FiniteElementContainers.CSCMatrix
    asm = SparseMatrixAssembler{sp_type, false, false}(dof)

    physics = Poisson(f)
    props = create_properties(physics)

    # U = create_unknowns(asm)
    U = create_field(asm)

    # trying to set up ics
    FT = AT.ExpressionFunction{Float64}
    ics = InitialConditions{FT}(sim.mesh, dof, sim.ics)
    update_ic_values!(ics, sim.mesh.nodal_coords)
    update_field_ics!(U, ics)

    # # p = create_parameters(mesh, asm, physics, props)
    println(sim.log_file.io, "Setup complete")
end

function @main(ARGS::Vector{String})
    app_main(ARGS)
    return 0
end