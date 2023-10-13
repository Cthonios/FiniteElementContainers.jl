@testset ExtendedTestSet "Poisson Multiple Fields" begin
  f_u(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
  f_v(X, _) = 2. * π^2 * sin(π * X[1]) * cos(π * X[2])
  f_w(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

  function residual(cell, u_el)
    @unpack X, N, ∇N_X, JxW = cell
    ∇u_q = ∇N_X' * u_el[1, :]
    ∇v_q = ∇N_X' * u_el[2, :]
    ∇w_q = ∇N_X' * u_el[3, :]
    R_u_q = (∇N_X * ∇u_q)' .- N' * f_u(X, 0.0)
    R_v_q = (∇N_X * ∇v_q)' .- N' * f_v(X, 0.0)
    R_w_q = (∇N_X * ∇w_q)' .- N' * f_w(X, 0.0)
    R_q = JxW * vcat(R_u_q, R_v_q, R_w_q)[:]
    return R_q
  end

  tangent(cell, u_el) = ForwardDiff.jacobian(z -> residual(cell, z), u_el)


  q_degree = 2
  mesh = read_mesh("./poisson_multi_field/poisson_multi_field.g", [1, 2, 3, 4])
  bcs = EssentialBC[
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
    EssentialBC(mesh, 1, 2)
    EssentialBC(mesh, 2, 2)
    EssentialBC(mesh, 3, 2)
    EssentialBC(mesh, 4, 2)
    EssentialBC(mesh, 1, 3)
    EssentialBC(mesh, 2, 3)
    EssentialBC(mesh, 3, 3)
    EssentialBC(mesh, 4, 3)
  ]

  fspaces, dof, asm = container_setup(mesh, [1], q_degree, 3)
  U = simple_solver(mesh, fspaces, dof, asm, bcs, residual, tangent)
  simple_post_processor("./poisson_multi_field/poisson_multi_field.g", U, ["u", "v", "w"])
  
  @test exodiff("./poisson_multi_field/poisson_multi_field.e", "./poisson_multi_field/poisson_multi_field.gold";
                command_file="./poisson_multi_field/poisson_multi_field.cmd")
  rm("./poisson_multi_field/poisson_multi_field.e")
end