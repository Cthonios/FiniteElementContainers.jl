# TODO adapt test to work on GPU
mesh_file = Base.source_dir() * "/poisson.g"

function test_poisson_pbcs()

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
    u = ScalarFunction(V, :u)
    dof = DofManager(u; use_condensed=true)
    asm = SparseMatrixAssembler(dof)
    p = create_parameters(mesh, asm, physics, props)

    pbcs = PeriodicBCs(mesh, dof, PeriodicBC[
        PeriodicBC(:u, :x, :sset_1, :sset_3, zero_func)
        PeriodicBC(:u, :y, :sset_2, :sset_4, zero_func)
    ])

    U = create_unknowns(dof)
    λ = FiniteElementContainers._create_constraint_field(dof, pbcs)
    sol = vcat(U, λ)
    assemble_stiffness!(asm, stiffness, U, p)

    K = stiffness(asm)
    C = FiniteElementContainers._constraint_matrix(dof, pbcs)

    nu = size(K, 1)
    nc = size(C, 1)

    A = vcat(
        hcat(K, C'),
        hcat(C, spzeros(nc, nc))
    )

    for n in 1:10
        U = sol[1:nu]
        λ = sol[nu+1:end]
        assemble_vector!(asm.residual_storage, asm.dof, residual, U, p)
        R_u = residual(asm) + C' * λ
        R_c = C * U
        R = vcat(R_u, R_c)

        # trying a singel newton step
        solver = Krylov.MinresSolver(A, -R)
        Krylov.solve!(solver, A, -R)
        Δsol = Krylov.solution(solver)
        sol = sol + Δsol

        # @show norm(R) norm(Δsol)
        # @show norm(R_u) norm(R_c)
        # display(λ)
    end

    u_an = u_analytic.(eachcol(mesh.nodal_coords))

    for n in axes(mesh.nodal_coords, 2)
        @test isapprox(sol[n], u_an[n], atol=5e-2)
    end
end
