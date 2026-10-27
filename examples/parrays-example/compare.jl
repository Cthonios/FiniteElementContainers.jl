using Exodus
exo = ExodusDatabase("output.e", "r")
x = read_coordinates(exo)
u = read_values(exo, NodalVariable, 1, "u")
u_an = 0.25 * sin.(2π * x[1, :]) .* sin.(2π * x[2, :])
@show maximum(abs.(u .- u_an))
