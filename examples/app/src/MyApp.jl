using Exodus
using FiniteElementContainers

function test_some_stuff()
    a = Vector{Number}(undef, 0)
    push!(a, 1)
    push!(a, 2.0)
    return a
end

function parse_args(ARGS)
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "-")
            if i < length(ARGS) && !startswith(ARGS[i + 1], "-")
                args[arg] = ARGS[i + 1]
                i += 2
            else
                args[arg] = "true"  # flag
                i += 1
            end
        else
            push!(args, arg => "true")  # positional?
            i += 1
        end
    end
    return args
end

function @main(ARGS)
    println(Core.stdout, "MyApp:")
    args = parse_args(ARGS)
    for (k, v) in pairs(args)
        println(Core.stdout, "$k = $v")
    end

    mesh_file = args["--mesh-file"]

    exo = ExodusDatabase{Int32, Int32, Int32, Float64}(mesh_file, "r")
    fm = FileMesh(ExodusMesh, exo)
    mesh = UnstructuredMesh(fm, false, false)
    # # display(exo)
    # # println(Core.stdout, exo)
    # # @show exo
    # coords = read_coordinates(exo)
    # println(Core.stdout, "My coords $(sum(coords))")
    # close(exo)

    # mesh = UnstructuredMesh(mesh_file)
    # V = FunctionSpace(mesh, H1Field, Lagrange)
    # u = ScalarFunction(V, :u)

    # vals = test_some_stuff()
    # # println(Core.stdout, vals)
    # for val in vals
    #     # println(Core.stdout, val)
    #     println(val)
    # end
    close(exo)
    return 0
end
