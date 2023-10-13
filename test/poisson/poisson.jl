@testset ExtendedTestSet "Poisson" begin
  f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

  function residual(cell, u_el)
    @unpack X, N, ∇N_X, JxW = cell
    ∇u_q = ∇N_X' * u_el[1, :]
    R_q = (∇N_X * ∇u_q)' .- N' * f(X, 0.0)
    return JxW * R_q[:]
  end

  function tangent(cell, _)
    @unpack X, N, ∇N_X, JxW = cell
    K_q = ∇N_X * ∇N_X'
    return JxW * K_q
  end

  q_degree = 2
  mesh = read_mesh("./poisson/poisson.g", [1, 2, 3, 4])
  bcs = EssentialBC[
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
  ]

  fspaces, dof, asm = container_setup(mesh, [1], q_degree, 1)
  U = simple_solver(mesh, fspaces, dof, asm, bcs, residual, tangent)
  simple_post_processor("./poisson/poisson.g", U, ["u"])
  
  @test exodiff("./poisson/poisson.e", "./poisson/poisson.gold")
  rm("./poisson/poisson.e")
end