dummy_ic_func_1(x) = 3.

function test_ic_input()
    ic = InitialCondition(:my_var, :my_block, dummy_ic_func_1)
    @test ic.block_name == :my_block
    @test typeof(ic.func) == typeof(dummy_ic_func_1)
    @test ic.var_name == :my_var
end

function ic_container_init()
    mesh = UnstructuredMesh("poisson/poisson.g")
    fspace = FunctionSpace(mesh, H1Field, Lagrange)
    u = VectorFunction(fspace, :displ)
    dof = DofManager(u)
    ic_in = InitialCondition(:displ_x, :block_1, dummy_ic_func_1)
    # ic_cont = FiniteElementContainers.InitialConditionContainer(mesh, dof, ic_in)
    ic_conts, ic_funcs = create_ics(mesh, dof, [ic_in])
    U = create_field(dof)
    return ic_conts, ic_funcs, mesh.nodal_coords, U
end

function test_ic_update_values(ic_conts, ic_funcs, X)
    FiniteElementContainers.update_ic_values!(ic_conts, ic_funcs, X)
    @test all(values(ic_conts)[1].vals .== 3.)
end

function test_ic_update_field_ics(ic_conts, U)
    update_field_ics!(U, ic_conts)
    @test all(U[1, :] .== 3.)
    @test all(U[2, :] .== 0.)
end

@testset ExtendedTestSet "InitialConditions" begin
    test_ic_input()
    ic_conts, ic_funcs, X, U = ic_container_init()
    test_ic_update_values(ic_conts, ic_funcs, X)
    test_ic_update_field_ics(ic_conts, U)
end
