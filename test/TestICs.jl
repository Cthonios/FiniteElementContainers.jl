@testsnippet ICHelper begin
    dummy_ic_func_1(x) = 3.
    mesh = UnstructuredMesh("poisson/poisson.g")
    fspace = FunctionSpace(mesh, H1Field, Lagrange)
    u = VectorFunction(fspace, "displ")
    dof = DofManager(u)
end

@testitem "ICs - test_ic_input" setup=[ICHelper] begin
    ic = InitialCondition("my_var", dummy_ic_func_1; block_name = "my_block")
    @test ic.block_name == "my_block"
    @test ic.nset_name === nothing
    @test ic.sset_name === nothing
    @test typeof(ic.func) == typeof(dummy_ic_func_1)
    @test ic.var_name == "my_var"
end

@testitem "ICs - ic_container_init_and_update - block" setup=[ICHelper] begin
    ic_in = InitialCondition("displ_x", dummy_ic_func_1; block_name = "block_1")
    ics = InitialConditions(mesh, dof, [ic_in])
    @show ics
    U = create_field(dof)
    X = mesh.nodal_coords
    FiniteElementContainers.update_ic_values!(ics, X)
    @test all(values(ics.ic_caches)[1].vals .== 3.)
    update_field_ics!(U, ics)
    @test all(U[1, :] .== 3.)
    @test all(U[2, :] .== 0.)
end

@testitem "ICs - ic_container_init_and_update - nodeset" setup=[ICHelper] begin
    nodes = mesh.nodeset_nodes["nset_1"]
    ic_in = InitialCondition("displ_x", dummy_ic_func_1; nodeset_name = "nset_1")
    ics = InitialConditions(mesh, dof, [ic_in])
    @show ics
    U = create_field(dof)
    X = mesh.nodal_coords
    FiniteElementContainers.update_ic_values!(ics, X)
    @test all(values(ics.ic_caches)[1].vals .== 3.)
    update_field_ics!(U, ics)
    @test all(U[1, nodes] .== 3.)
    @test all(U[2, nodes] .== 0.)
end

@testitem "ICs - ic_container_init_and_update - sideset" setup=[ICHelper] begin
    nodes = mesh.sideset_nodes["sset_1"]
    ic_in = InitialCondition("displ_x", dummy_ic_func_1; sideset_name = "sset_1")
    ics = InitialConditions(mesh, dof, [ic_in])
    @show ics
    U = create_field(dof)
    X = mesh.nodal_coords
    FiniteElementContainers.update_ic_values!(ics, X)
    @test all(values(ics.ic_caches)[1].vals .== 3.)
    update_field_ics!(U, ics)
    @test all(U[1, nodes] .== 3.)
    @test all(U[2, nodes] .== 0.)
end

@testitem "ICs - ic_container_init_and_update_juliac_safe" setup=[ICHelper] begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction
    expr_func = ScalarExpressionFunction{Float64}("3.0", ["x", "y"])
    ic_in = InitialCondition{ScalarExpressionFunction{Float64}}("displ_x", expr_func, "block_1", nothing, nothing)
    ics = InitialConditions(mesh, dof, [ic_in])
    @show ics
    U = create_field(dof)
    X = mesh.nodal_coords
    FiniteElementContainers.update_ic_values!(ics, X)
    @test all(values(ics.ic_caches)[1].vals .== 3.)
    update_field_ics!(U, ics)
    @test all(U[1, :] .== 3.)
    @test all(U[2, :] .== 0.)
end

# TODO add adapt test methods and gpu methods?
