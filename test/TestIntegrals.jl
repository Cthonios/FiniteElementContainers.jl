function test_scalar_integral()
    mesh_file = "./poisson/poisson.g"
    mesh = UnstructuredMesh(mesh_file)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u)
    f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
    physics = Poisson(f)
    integral = FiniteElementContainers.ScalarCellIntegral(asm, energy)
    grad_integral = FiniteElementContainers.gradient(integral)
end

@testset "Integrals" begin
    test_scalar_integral()
end
