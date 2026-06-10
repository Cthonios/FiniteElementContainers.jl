using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using SparseArrays


# THINGS TO DO
# 1. Need to propogate dof_to_unknown map to assemblers
#    so we can use the new dof_to_unknown_index method
#    or at least that behavior so that PBC assembly works correctly

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
asm = SparseMatrixAssembler(dof)

pbcs = PeriodicBC[
    PeriodicBC("u", "x", zero_func, "sset_1", "sset_3")
    PeriodicBC("u", "y", zero_func, "sset_4", "sset_2")
]
p = create_parameters(mesh, asm, physics, props; periodic_bcs = pbcs)
# U = create_unknowns(dof)
solver = NewtonSolver(DirectLinearSolver(asm))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

# display(Uu)
U = p.field
# U_an = similar(U)
# X = p.coords
u_an = u_analytic.(eachcol(mesh.nodal_coords))

# for n in axes(mesh.nodal_coords, 2)
#     @assert isapprox(U[n], u_an[n], atol=5e-2)
# end


# pp = PostProcessor(mesh, output_file, u; copy_mesh_file = false)
pp = PostProcessor(mesh, output_file, u)
write_times(pp, 1, 0.0)
write_field(pp, 1, ("u",), U)
close(pp)
