@testset ExtendedTestSet "Poisson Energy" begin

  f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

  function energy(cell, u_el)
    @unpack X, N, ∇N_X, JxW = cell
    u_q = dot(N, u_el[1, :])
    ∇u_q = ∇N_X' * u_el[1, :]
    Π = 0.5 * dot(∇u_q, ∇u_q) - f(X, 0.0) * u_q
  end

  residual(cell, u_el) = ForwardDiff.gradient(z -> energy(cell, z), u_el)[:]
  tangent(cell, u_el)  = ForwardDiff.hessian(z -> energy(cell, z), u_el)

  q_degree = 2
  mesh = read_mesh("./poisson_energy/poisson_energy.g", [1, 2, 3, 4])
  bcs = EssentialBC[
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
  ]

  fspaces, dof, asm = container_setup(mesh, [1], q_degree, 1)
  U = simple_solver(mesh, fspaces, dof, asm, bcs, residual, tangent)
  simple_post_processor("./poisson_energy/poisson_energy.g", U, ["u"])
  
  @test exodiff("./poisson_energy/poisson_energy.e", "./poisson_energy/poisson_energy.gold")
  rm("./poisson_energy/poisson_energy.e")
end