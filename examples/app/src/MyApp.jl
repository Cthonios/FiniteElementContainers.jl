import FiniteElementContainers as FEC
import FiniteElementContainers.AppTools as AT
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using TimerOutputs

# include files
include("Physics.jl")

# f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])

const N = 1

function app_main(ARGS::Vector{String})
    #####################################
    # need to define some types
    #####################################
    # N = 1 # number of fields to solve for in app
    ET = ExodusDatabase{Int32, Int32, Int32, Float64}
    SFT = AT.ScalarExpressionFunction{Float64}
    VFT = AT.VectorExpressionFunction{N, Float64}
    SPT = FEC.CSCMatrix()

    ##################################################
    # Setup app
    ##################################################
    app = AT.App{N}("MyApp")
    sim = AT.setup(app, ARGS)

    #####################################
    # setup function space
    #####################################
    V = FunctionSpace{true}(sim.mesh, H1Field, Lagrange)
    u = ScalarFunction(V, "u")
    dof = DofManager{false}(u)
    asm = SparseMatrixAssembler{SPT, false, false}(dof)

    # setup physics and properties
    # physics = Poisson{typeof(f)}[]
    physics = Laplace[]
    props = Vector{Float64}[]
    for _ in 1:length(sim.mesh.element_conns)
        # temp_physics = Poisson(f)
        temp_physics = Laplace()
        push!(physics, temp_physics)
        push!(props, create_properties(temp_physics))
    end

    times = TimeStepper(0.0, 0.0, 1)
    p = FEC.TypeStableParameters{SFT, VFT}(
        sim.mesh, asm,
        physics, props,
        sim.ics, sim.dbcs, sim.nbcs, sim.pbcs, sim.srcs, times
    )
    FEC.initialize!(p)

    preconditioner = I
    timer = TimerOutput()
    workspace = CgWorkspace(stiffness(asm), residual(asm))
    ΔUu = create_unknowns(asm)
    lsolver_settings = FEC.IterativeLinearSolverSettings()
    lsolver = IterativeLinearSolver(asm, preconditioner, lsolver_settings, timer, ΔUu, workspace)
    nlsolver = NewtonSolver(lsolver)
    integrator = QuasiStaticIntegrator(nlsolver)

    println(sim.log_file.io, "Setup complete")
    println(sim.log_file.io, "Solving...")
    evolve!(integrator, p)
    println(sim.log_file.io, "Solve complete")
    println(Core.stdout, maximum(p.field.data))
    println(Core.stdout, minimum(p.field.data))

    pp = PostProcessor{ET}(FEC.ExodusMesh, sim.mesh, "juliac_output_2.e")
    FEC.add_function!(pp, u)
    FEC.finalize_setup!(pp)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ("u",), p.field)
    close(pp)
end

function @main(ARGS::Vector{String})
    app_main(ARGS)
    return 0
end
