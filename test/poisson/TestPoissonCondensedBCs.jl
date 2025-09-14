import KernelAbstractions as KA
using Exodus
using FiniteElementContainers
using LinearAlgebra
using StaticArrays
using Test

# mesh file
gold_file = Base.source_dir() * "/poisson.gold"
mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
# f(_, _) = 1.
bc_func(_, _) = 0.

include("TestPoissonCommon.jl")

function test_poisson_condensed_bcs()
    mesh = UnstructuredMesh(mesh_file)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Poisson()
    props = create_properties(physics)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u; use_condensed=true)

    # setup and update bcs
    dbcs = DirichletBC[
        DirichletBC(:u, :sset_1, bc_func),
        DirichletBC(:u, :sset_2, bc_func),
        DirichletBC(:u, :sset_3, bc_func),
        DirichletBC(:u, :sset_4, bc_func),
    ]

    p = create_parameters(
        mesh, asm, physics, props; 
        dirichlet_bcs=dbcs
    )
    # Uu = create_unknowns(asm)

    # FiniteElementContainers.update_bc_values!(p)
    # for bc in values(p.dirichlet_bcs)
    #     FiniteElementContainers._update_field_dirichlet_bcs!(Uu, bc, KA.get_backend(bc))
    # end

    # for n in 1:3
    #     assemble_stiffness!(asm, stiffness, Uu, p)
    #     assemble_vector!(asm, residual, Uu, p)
    #     K = stiffness(asm)
    #     R = residual(asm)

    #     dUu = -K \ R
    #     Uu = Uu + dUu
    # end
    # Uu

    solver = NewtonSolver(DirectLinearSolver(asm))
    integrator = QuasiStaticIntegrator(solver)
    evolve!(integrator, p)

    pp = PostProcessor(mesh, output_file, u)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ("u",), p.h1_field)
    close(pp)

    if !Sys.iswindows()
        @test exodiff(output_file, gold_file)
    end
    rm(output_file; force=true)
    display(solver.timer)
end

test_poisson_condensed_bcs()
test_poisson_condensed_bcs()
