dummy_ic_func_1(x) = 3.

function test_ic_input()
    ic = InitialCondition(:my_var, dummy_ic_func_1, :my_block)
    @test ic.block_name == :my_block
    @test typeof(ic.func) == typeof(dummy_ic_func_1)
    @test ic.var_name == :my_var
end

function ic_container_init()
    mesh = UnstructuredMesh("poisson/poisson.g")
    fspace = FunctionSpace(mesh, H1Field, Lagrange)
    u = VectorFunction(fspace, :displ)
    dof = DofManager(u)
    ic_in = InitialCondition(:displ_x, dummy_ic_func_1, :block_1)
    ics = InitialConditions(mesh, dof, [ic_in])
    U = create_field(dof)
    return ics, mesh.nodal_coords, U
end

function test_ic_update_values(ics, X)
    FiniteElementContainers.update_ic_values!(ics, X)
    @test all(values(ics.ic_caches)[1].vals .== 3.)
end

function test_ic_update_field_ics(ics, U)
    update_field_ics!(U, ics)
    @test all(U[1, :] .== 3.)
    @test all(U[2, :] .== 0.)
end

@testset "InitialConditions" begin
    test_ic_input()
    ics, X, U = ic_container_init()
    test_ic_update_values(ics, X)
    test_ic_update_field_ics(ics, U)
end
