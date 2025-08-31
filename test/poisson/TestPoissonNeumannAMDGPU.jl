import KernelAbstractions as KA
using Adapt
using AMDGPU
using Exodus
using FiniteElementContainers
using Krylov
using StaticArrays
using Test

mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/poisson.e"

f(X, _) = 1.
bc_func_1(_, _) = 2.
bc_func_2(_, _) = SVector{1, Float64}(1.)

include("TestPoissonCommon.jl")

function test_poisson_neumann()
    mesh = UnstructuredMesh(mesh_file)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Poisson()
    props = create_properties(physics)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(H1Field, u)

    dbcs = DirichletBC[
        DirichletBC(:u, :sset_1, bc_func_1)
        DirichletBC(:u, :sset_2, bc_func_1)
    ]
    nbcs = NeumannBC[
        NeumannBC(:u, :sset_3, bc_func_2)
        NeumannBC(:u, :sset_4, bc_func_2)
    ]

    p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs, neumann_bcs=nbcs)

    # device movement
    p_gpu = p |> rocm
    asm_gpu = asm |> rocm

    solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
    integrator = QuasiStaticIntegrator(solver)
    evolve!(integrator, p_gpu)

    p = p_gpu |> cpu
    U = p.h1_field

    pp = PostProcessor(mesh, output_file, u)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ("u",), p.h1_field)
    close(pp)
end

@time test_poisson_neumann()
@time test_poisson_neumann()

