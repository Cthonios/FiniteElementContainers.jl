# using Exodus
# using FiniteElementContainers

# function test_some_stuff()
#     a = Vector{Number}(undef, 0)
#     push!(a, 1)
#     push!(a, 2.0)
#     return a
# end

# function parse_args(ARGS)
#     args = Dict{String,String}()
#     i = 1
#     while i <= length(ARGS)
#         arg = ARGS[i]
#         if startswith(arg, "-")
#             if i < length(ARGS) && !startswith(ARGS[i + 1], "-")
#                 args[arg] = ARGS[i + 1]
#                 i += 2
#             else
#                 args[arg] = "true"  # flag
#                 i += 1
#             end
#         else
#             push!(args, arg => "true")  # positional?
#             i += 1
#         end
#     end
#     return args
# end


# function @main(ARGS)
#     println(Core.stdout, "MyApp:")
#     args = parse_args(ARGS)
#     for (k, v) in pairs(args)
#         println(Core.stdout, "$k = $v")
#     end

#     mesh_file = args["--mesh-file"]

#     exo = ExodusDatabase{Int32, Int32, Int32, Float64}(mesh_file, "r")
#     fm = FileMesh(ExodusMesh, exo)
#     mesh = UnstructuredMesh(fm, false, false)
#     # # display(exo)
#     # # println(Core.stdout, exo)
#     # # @show exo
#     # coords = read_coordinates(exo)
#     # println(Core.stdout, "My coords $(sum(coords))")
#     # close(exo)

#     # mesh = UnstructuredMesh(mesh_file)
#     # V = FunctionSpace(mesh, H1Field, Lagrange)
#     # u = ScalarFunction(V, :u)

#     # vals = test_some_stuff()
#     # # println(Core.stdout, vals)
#     # for val in vals
#     #     # println(Core.stdout, val)
#     #     println(val)
#     # end
#     close(exo)
#     return 0
# end


import FiniteElementContainers as FEC
import FiniteElementContainers: ExodusMesh, FileMesh
import FiniteElementContainers: nodal_coordinates_and_ids
import FiniteElementContainers.AppTools as AT
using Exodus
using FiniteElementContainers
using ReferenceFiniteElements

struct Poisson{F <: Function} <: AbstractPhysics{1, 0, 0}
    func::F
end

@inline function FiniteElementContainers.residual(
    physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
  )
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    R_q = ∇u_q * ∇N_X' - N' * physics.func(X_q, 0.0)
    return JxW * R_q[:]
end

@inline function FiniteElementContainers.stiffness(
    physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
  )
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    K_q = ∇N_X * ∇N_X'
    return JxW * K_q
end

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_zero_func(_, _) = 0.0

function @main(ARGS)
    cli_args = AT.parse_cli_args()
    input_settings = AT.parse_input_file(cli_args)

    #####################################
    # setup mesh
    #####################################
    mesh_settings = input_settings.mesh
    mesh_path = mesh_settings.file_path
    exo = ExodusDatabase{Int32, Int32, Int32, Float64}(mesh_path, "r")
    fm = FileMesh{
        ExodusDatabase{Int32, Int32, Int32, Float64},
        ExodusMesh
    }(mesh_path, exo)
    # read nodes
    coords_type = H1Field{Float64, Vector{Float64}, 2}
    nodal_coords, n_id_map = nodal_coordinates_and_ids(coords_type, fm)
    # read element block types, conn, etc.
    el_id_map = element_ids(fm)
    el_conns, el_id_maps, el_block_names, el_block_names_map, el_types = element_blocks(fm)
    # read nodesets
    nset_names, nset_nodes = nodesets(fm)

    # read sidesets 
    sset_elems, sset_names, sset_nodes, sset_sides, sset_side_nodes = sidesets(fm)

    mesh = UnstructuredMesh{
        FileMesh{
            ExodusDatabase{Int32, Int32, Int32, Float64},
            ExodusMesh
        },
        2, Float64, Int, Nothing, Nothing
    }(
        fm,
        nodal_coords, 
        el_block_names, el_block_names_map, el_types, el_conns, 
        el_id_map, el_id_maps, 
        n_id_map,
        nset_names, nset_nodes,
        sset_names, sset_elems, sset_nodes, 
        sset_sides, sset_side_nodes,
        nothing, nothing
    )

    #####################################
    # setup function space
    #####################################
    V = FunctionSpace{true}(mesh, H1Field, Lagrange)
    physics = Poisson(f)
    props = create_properties(physics)
    u = ScalarFunction(V, "u")
    dof = DofManager{false}(u)
    sp_type = FiniteElementContainers.CSCMatrix
    asm = SparseMatrixAssembler{sp_type, false, false}(dof)

    U = create_unknowns(asm)
    # p = create_parameters(mesh, asm, physics, props)
    return 0
end
