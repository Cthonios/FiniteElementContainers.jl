using FiniteElementContainers
using StaticArrays

include("../../test/poisson/TestPoissonCommon.jl")

mesh_file = dirname(dirname(Base.source_dir())) * "/test/poisson/poisson.g"
output_file = "output-test.e"

u_exact(X) = sin(π * X[1]) * sin(π * X[2]) + X[1] + X[2]

f(X, _) = 2π^2 * sin(π * X[1]) * sin(π * X[2])

bc_func_1(x, t, u::SVector{NF, T}) where {NF, T} = SVector{1, T}(
    x[1] + 2 - π * sin(π * x[1]) - u[1]
) # y = 1
bc_func_2(x, t, u::SVector{NF, T}) where {NF, T} = SVector{1, T}(
    x[2] - π * sin(π * x[2]) - 1 - u[1]
) # x = 0
bc_func_3(x, t,u::SVector{NF, T}) where {NF, T} = SVector{1, T}(
    x[1] - π * sin(π * x[1]) - 1 - u[1]
) # y = 0
bc_func_4(x, t, u::SVector{NF, T}) where {NF, T} = SVector{1, T}(
    x[2] + 2 - π * sin(π * x[2]) - u[1]
) # x = 1

mesh = UnstructuredMesh(mesh_file)

rbcs = RobinBC[
    RobinBC("u", bc_func_1, "sset_1"),
    RobinBC("u", bc_func_2, "sset_2"),
    RobinBC("u", bc_func_3, "sset_3"),
    RobinBC("u", bc_func_4, "sset_4"),
]

V = FunctionSpace(mesh, H1Field, Lagrange)

physics = Poisson(f)
props = create_properties(physics)

u = ScalarFunction(V, "u")
asm = SparseMatrixAssembler(u)

p = create_parameters(
    mesh,
    asm,
    physics,
    props;
    robin_bcs = rbcs,
)

solver = NewtonSolver(DirectLinearSolver(asm))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

X = p.coords
@show u_analytic = u_exact.(eachcol(X))
@show p.field