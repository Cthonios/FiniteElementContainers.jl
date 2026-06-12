using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using SparseArrays

include("../../test/poisson/TestPoissonCommon.jl")

# function simulation()
mesh_file = dirname(dirname(Base.source_dir())) * "/test/poisson/poisson.g"
output_file = "output-poisson-periodic.e"

u_analytic(x) = cos(2π*x[1]) + 0.5*cos(4π*x[2]) + 0.25*sin(2π*x[1])*sin(4π*x[2])

f(X, _) = begin
    x, y = X
    (2π)^2 * cos(2π * x) +
    0.5 * (4π)^2 * cos(4π * y) +
    0.25 * ((2π)^2 + (4π)^2) * sin(2π * x) * sin(4π * y)
end
zero_func(_, _) = 0.

mesh = UnstructuredMesh(mesh_file)
V = FunctionSpace(mesh, H1Field, Lagrange) 
physics = Poisson(f)
props = create_properties(physics)
u = ScalarFunction(V, "u")
dof = DofManager(u)
asm = SparseMatrixAssembler(dof; use_inplace_methods = false, use_sparse_vector = false)

pbcs = PeriodicBC[
    PeriodicBC("u", "x", zero_func, "sset_1", "sset_3")
    PeriodicBC("u", "y", zero_func, "sset_4", "sset_2")
]
p = create_parameters(mesh, asm, physics, props; periodic_bcs = pbcs)
# U = create_unknowns(dof)
solver = NewtonSolver(IterativeLinearSolver(asm, :cg))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

# display(Uu)
U = p.field
# U_an = similar(U)
# X = p.coords
u_an = H1Field{Float64, Vector{Float64}, 1}(u_analytic.(eachcol(mesh.nodal_coords)))

for n in axes(mesh.nodal_coords, 2)
    @assert isapprox(U[n], u_an[n], atol=5e-4)
end


# pp = PostProcessor(mesh, output_file, u; copy_mesh_file = false)
pp = PostProcessor(mesh, output_file, u)
write_times(pp, 1, 0.0)
write_field(pp, 1, ("u",), U)
close(pp)
