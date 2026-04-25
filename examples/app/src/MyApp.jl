import FiniteElementContainers as FEC
import FiniteElementContainers.AppTools as AT
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using TimerOutputs

# include files
include("Physics.jl")

f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])

function app_main(ARGS::Vector{String})
    app = AT.App("MyApp")
    AT.add_cli_arg!(app, "--backend"; default = "cpu")
    sim = AT.setup(app, ARGS)

    #####################################
    # need to define some types
    #####################################
    FT = AT.ExpressionFunction{Float64}
    SPT = FEC.CSCMatrix()

    #####################################
    # setup function space
    #####################################
    # block_ids = FEC._setup_juliac_safe_block_to_ref_fe_id(sim.mesh)
    # for n in 1:16
    #     println(Core.stdout, "Block $n element type $(block_ids[n])")
    # end
    V = FunctionSpace{true}(sim.mesh, H1Field, Lagrange)
    u = ScalarFunction(V, "u")
    dof = DofManager{false}(u)
    asm = SparseMatrixAssembler{SPT, false, false}(dof)

    # setup physics and properties
    physics = Poisson{typeof(f)}[]
    props = Vector{Float64}[]
    for _ in 1:length(sim.mesh.element_conns)
        temp_physics = Poisson(f)
        push!(physics, temp_physics)
        push!(props, create_properties(temp_physics))
    end

    times = TimeStepper(0.0, 0.0, 1)
    p = FEC.TypeStableParameters{FT}(sim.mesh, asm, physics, props, sim.ics, sim.dbcs, times)
    # U = create_field(asm)
    # Uu = create_unknowns(asm)

    # # setting up ics
    # # ics = InitialConditions{FT}(sim.mesh, dof, sim.ics)
    # X = sim.mesh.nodal_coords
    # # println(Core.stdout, "X size 1 = $(size(X, 1))")
    # # println(Core.stdout, "X size 2 = $(size(X, 2))")
    # # for block_id in 1:1
    # #     ref_fe = FEC.block_reference_element(V, block_id)
    # #     println(Core.stdout, ref_fe)
    # # end

    FEC.initialize!(p)
    # FEC._update_for_assembly!(p, dof, Uu)

    # # lsolver = DirectLinearSolver(asm)
    # # lsolver = x -> IterativeLinearSolver(x, :cg)
    preconditioner = I
    timer = TimerOutput()
    workspace = CgWorkspace(stiffness(asm), residual(asm))
    ΔUu = create_unknowns(asm)
    lsolver = IterativeLinearSolver(asm, preconditioner, workspace, timer, ΔUu)
    nlsolver = NewtonSolver(lsolver)

    # println(Core.stdout, "Number of blocks = $(length(sim.mesh.element_types))")
    # # N = length(sim.mesh.element_types)
    # # # N = Val(N)
    # # N = Val{N}()
    # # ref_fe = _get_ref_fe(V, N)
    # # indices = ntuple(x -> x, N)
    # # indices = ntuple(x -> x, Val(N))
    # # s = MyStruct{N}(N)::MyStruct{N}
    integrator = QuasiStaticIntegrator(nlsolver)
    println(sim.log_file.io, "Setup complete")
    println(sim.log_file.io, "Solving...")
    evolve!(integrator, p)
    println(sim.log_file.io, "Solve complete")
    println(Core.stdout, maximum(p.field.data))
    println(Core.stdout, minimum(p.field.data))

    # pp = PostProcessor(sim.mesh, "juliac_output.e", u)

end

function @main(ARGS::Vector{String})
    app_main(ARGS)
    return 0
end