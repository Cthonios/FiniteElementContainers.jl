@testitem "BCs - test_point_sources" setup=[BCHelper] begin
    if "--test-amdgpu" in ARGS @eval using AMDGPU end
    if "--test-cuda" in ARGS @eval using CUDA end
    using FiniteElementContainers
    using Tensors
    using Test
    include("../mechanics/TestMechanicsCommon.jl")
    # backends = _get_backends()

    mesh_file = Base.source_dir() * "/cube.g"
    output_file = Base.source_dir() * "/cube.e"

    fixed(_, _) = 0.
    load(_, t) = 0.25e9 * t

    mesh = UnstructuredMesh(mesh_file)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Mechanics(ThreeDimensional())
    props = create_properties(physics)
    u = VectorFunction(V, "displ")
    times = TimeStepper(0., 1., 11)
    # lsolver = x -> IterativeLinearSolver(x, :cg)
    lsolver = DirectLinearSolver
    asm = SparseMatrixAssembler(
        u;
        use_inplace_methods = true
    )
    dbcs = [
        DirichletBC("displ_x", fixed; sideset_name = "ssx-")
        DirichletBC("displ_y", fixed; sideset_name = "ssy-")
        DirichletBC("displ_z", fixed; sideset_name = "ssz-")
    ]

    point_source_bcs = [
        PointSourceBC("displ_z", load; sideset_name = "ssz+")
    ]

    # pre-setup some scratch arrays
    p = create_parameters(
        mesh, asm, physics, props;
        dirichlet_bcs = dbcs,
        point_source_bcs = point_source_bcs,
        times = times
    )

    # if dev != cpu
    #   p = p |> dev
    #   asm = asm |> dev 
    # end

    solver = NewtonSolver(lsolver(asm))
    integrator = QuasiStaticIntegrator(solver)
    # pp = PostProcessor(mesh, output_file, u)
    U = p.field
    # write_times(pp, 1, 0.0)
    for n in 2:12
        evolve!(integrator, p)
    #     write_times(pp, n, p.times.time_current)
    #     write_field(pp, n, ("displ_x", "displ_y", "displ_z"), U)
    end

    # close(pp)
    # @test mean(U[1, :]) ≈ -0.125 rtol=1e-6
    # @test mean(U[2, :]) ≈ -0.125 rtol=1e-6
    # @test mean(U[3, :]) ≈ 0.5 rtol=1e-6
end
