"""
All of these tools are meant to be type safe
from the perspective of working with JuliaC
trim mode for small binaries
"""
module AppTools

import ..DirichletBC
import ..Expressions.ExpressionFunction
import ..ExodusMesh
import ..FileMesh
import ..FunctionSpace
import ..H1Field
import ..InitialCondition
import ..UnstructuredMesh
import ..element_blocks
import ..element_ids
import ..nodal_coordinates_and_ids
import ..nodesets
import ..sidesets
using Exodus
using ReferenceFiniteElements
using TOML

function _print_error_message(io::IO, msg::String)
    println(io, msg)
end

function error_message(io::IO, msg::String)
    @assert false _print_error_message(io, msg)
end

#######################################################
# CLI IO
#######################################################
struct CLIArg
    has_default::Bool
    has_input::Bool
    has_short_name::Bool
    is_required::Bool
    default::String
    help_message::String
    name::String
    short_name::String

    function CLIArg(
        name::String;
        default::String = "",
        has_input::Bool = true,
        help_message::String = "",
        is_required::Bool = true,
        short_name::String = ""
    )
        if default == ""
            has_default = false
        else
            has_default = true
        end

        if short_name == ""
            has_short_name = false
        else
            has_short_name = true
        end

        new(
            has_default, has_input, has_short_name, is_required,
            default, help_message, name, short_name
        )
    end
end

function argument_not_found_message(arg::CLIArg)
    if arg.has_short_name
        return "Argument $(arg.name) or $(arg.short_name) not found in input.\n"
    else
        return "Argument $(arg.name) not found in input.\n"
    end
end

mutable struct CLIArgParser
    cli_args::Vector{CLIArg}
    num_required_args::Int
    parsed_args::Dict{String, String}

    function CLIArgParser()
        cli_args = CLIArg[
            CLIArg("--help"; has_input = false, is_required = false, short_name = "-h")
            CLIArg("--input-file"; short_name = "-i")
            CLIArg("--log-file"; short_name = "-l")
        ]
        new(cli_args, 2, Dict{String, String}())
    end
end

function Base.show(io::IO, p::CLIArgParser)
    print_dict(io, p.parsed_args)
end

# TODO need to add check to make sure you don't
# add a CLI arg that is already there
function add_cli_arg!(p::CLIArgParser, name::String; kwargs...)
    cli_arg = CLIArg(name; kwargs...)
    push!(p.cli_args, cli_arg)
    if cli_arg.is_required
        p.num_required_args += 1
    end
    return nothing
end

function get_cli_arg(p::CLIArgParser, name::String)
    return p.parsed_args[name]
end

function help_message(p::CLIArgParser, other_msg::String)
    usage = """


    $(other_msg)


    How to use this executable

    <name of executable> 
    """
    for arg in p.cli_args
        usage *= arg.name * " "
        if arg.has_input
            usage *= "<some input> "
        end
        usage *= arg.help_message
        usage *= "\n"
    end
    usage *= "\n\n"
    for arg in p.cli_args
        usage *= arg.name
        if arg.has_short_name
            usage *= ", $(arg.short_name)"
        end
        usage *= "\n"
    end
    println(Core.stdout, usage)
end

function parse!(p::CLIArgParser, args::Vector{String})
    error_flag = 0
    error_msg = ""

    # parsed_args = Dict{String, String}()
    for arg in p.cli_args
        if arg.name in args || arg.short_name in args
            if arg.has_input
                index = findfirst(x -> x == arg.name, args)
                if index === nothing
                    if arg.has_short_name
                        index = findfirst(x -> x == arg.short_name, args)
                    else
                        error_flag += 1
                        error_msg *= argument_not_found_message(arg)
                    end
                end

                # TODO currently if we don't have an input
                # after a parameter that requires one
                # it will just return whatever it sees next
                # which could be another option name or out of bounds
                # this one we need to handle carefully
                p.parsed_args[arg.name] = args[index + 1]
            else
                p.parsed_args[arg.name] = "true"
            end
        end
    end

    # if we see help just abort early
    if haskey(p.parsed_args, "--help")
        @assert false help_message(p, "")
    end

    # check for required args
    for arg in p.cli_args
        if arg.is_required
            if !haskey(p.parsed_args, arg.name)
                error_flag += 1
                error_msg *= argument_not_found_message(arg)
            end
        end

        if arg.has_default && !haskey(p.parsed_args, arg.name)
            p.parsed_args[arg.name] = arg.default
        end
    end

    if error_flag > 0
        @assert false help_message(p, error_msg)
    end

    # return parsed_args
end

#######################################################
# Log file
#######################################################
struct LogFile{IO}
    io::IO

    function LogFile(file_name::String)
        io = open(file_name, "w")
        new{typeof(io)}(io)
    end
end

function Base.close(l::LogFile)
    close(l.io)
end

function Base.show(l::LogFile, msg)
    show(l.io, msg)
end

const MAX_COLUMNS = 96

function print_banner(io::IO, msg::String)
    string = repeat("=", MAX_COLUMNS)
    string *= "\n= $(msg)\n"
    string *= repeat("=", MAX_COLUMNS)
    println(io, string)
end

print_banner(l::LogFile, msg::String) = print_banner(l.io, msg)

function print_dict(io::IO, d::Dict{String, String})
    isempty(d) && return

    # Compute max key length (type-stable)
    maxlen = 0
    for k in Base.keys(d)
        len = ncodeunits(k)  # safer for strings than length in some contexts
        if len > maxlen
            maxlen = len
        end
    end

    # Optional: deterministic order without Pair allocations
    ks = collect(Base.keys(d))  # Vector{String}, concrete
    sort!(ks)

    for k in ks
        v = d[k]

        print(io, '"')
        print(io, k)
        print(io, '"')

        # manual padding (avoids rpad allocation)
        pad = maxlen - ncodeunits(k)
        for _ in 1:pad
            print(io, ' ')
        end

        print(io, " => ")

        print(io, '"')
        print(io, v)
        println(io, '"')
    end
end

print_dict(l::LogFile, d::Dict{String, String}) = print_dict(l, d)

#######################################################
# Input file
#######################################################
struct BCSettings{T <: Number}
    dirichlet::Vector{DirichletBC{ExpressionFunction{T}}}
end

struct FunctionSettings{T <: Number}
    expr_funcs::Dict{String, ExpressionFunction{T}}
end

struct FunctionSpaceSettings
    field_type::String
    polynomial_type::String
end

struct MeshSettings
    file_path::String
    file_type::String
end

struct VariableSettings
    function_space::String
    name::String
    variable_type::String
end

struct InputSettings{T <: Number}
    # function_spaces::Dict{String, FunctionSpaceSettings}
    bcs::BCSettings{T}
    functions::FunctionSettings{T}
    # ics::Vector{InitialConditionSettings}
    ics::Vector{InitialCondition{ExpressionFunction{T}}}
    mesh::MeshSettings
    # variables::Dict{String, VariableSettings}
end

function _parse_boundary_condition_settings(log_file, data, functions::FunctionSettings{T}) where T <: Number
    print_banner(log_file, "Boundary conditions")
    bc_settings = data["boundary_conditions"]::Dict{String, Any}
    dbc_settings = bc_settings["dirichlet"]::Vector{Any}
    dbcs = DirichletBC{ExpressionFunction{T}}[]
    for bc in dbc_settings
        temp = bc::Dict{String, Any}
        func = temp["function"]::String
        func = functions.expr_funcs[func]
        var = temp["variable"]::String
        if haskey(temp, "block")
            block = temp["block"]::String
            push!(dbcs, DirichletBC(var, func; block_name = block))
        elseif haskey(temp, "node_set")
            node_set = temp["node_set"]::String
            push!(dbcs, DirichletBC(var, func; nodeset_name = node_set))
        elseif haskey(temp, "side_set")
            side_set = temp["side_set"]::String
            push!(dbcs, DirichletBC(var, func; sideset_name = side_set))
        else
            error_message(log_file, "Requires block, node_set, or side_set")
        end
    end
    return BCSettings(dbcs)
end

function _parse_function_space_settings(log_file, data)
    settings = data["function_spaces"]::Dict{String, Any}
    fspaces = Dict{String, FunctionSpaceSettings}()
    for (k, v) in settings
        name = k::String
        temp = v::Dict{String, Any}
        field_type = temp["field_type"]::String
        polynomial_type = temp["polynomial_type"]::String
        fspaces[name] = FunctionSpaceSettings(field_type, polynomial_type)
    end
    return fspaces
end

function _parse_function_settings(log_file, data, ::Type{T}) where T <: Number
    print_banner(log_file, "Functions")
    functions = Dict{String, ExpressionFunction{T}}()
    func_settings = data["functions"]::Dict{String, Any}
    for (k, v) in pairs(func_settings)
        name = k::String
        temp = v::Dict{String, Any}
        type = temp["type"]::String
        # TODO constant and anaytic are really the same
        # should we fuse these into "expression" or something like that?
        if type == "analytic"
            expr = temp["expression"]::String
            vars = temp["variables"]::Vector{String}
            println(log_file.io, "Parsing analytic function with expression = $expr")
            functions[name] = ExpressionFunction{T}(expr, vars)
        elseif type == "constant"
            expr = temp["expression"]::String
            vars = temp["variables"]::Vector{String}
            println(log_file.io, "Parsing constant function with expression = $expr")
            functions[name] = ExpressionFunction{T}(expr, vars)
        else
            @assert false "Unsupported function type $type"
        end
    end
    return FunctionSettings{T}(functions)
end

function _parse_initial_condition_settings(log_file, data, functions)
    print_banner(log_file, "Initial conditions")
    ic_settings = data["initial_conditions"]::Vector{Any}
    ics = InitialCondition{ExpressionFunction{Float64}}[]
    for ic in ic_settings
        temp = ic::Dict{String, Any}
        block = temp["block"]::String
        func = temp["function"]::String
        var = temp["variable"]::String
        func = functions.expr_funcs[func]
        push!(ics, InitialCondition(var, func, block))
    end
    return ics
end

function _parse_mesh_settings(log_file, data)
    mesh_settings = data["mesh"]::Dict{String, Any}
    file_path = mesh_settings["file_path"]::String
    file_type = lowercase(mesh_settings["file_type"]::String)
    return MeshSettings(file_path, file_type)
end

function _parse_variable_settings(log_file, data)
    var_settings = data["variables"]::Dict{String, Any}
    vars = Dict{String, VariableSettings}()
    for (k, v) in var_settings
        name = k::String
        temp = v::Dict{String, Any}
        fspace = temp["function_space"]::String
        type = temp["type"]::String
        vars[name] = VariableSettings(fspace, name, type)
    end
    return vars
end

function parse_input_file(cli_args::CLIArgParser, log_file::LogFile, ::Type{T} = Float64) where T <: Number
    input_file = cli_args.parsed_args["--input-file"]
    println(log_file.io, "Parsing input file from $(input_file)")
    println(log_file.io, "Input file contents...")
    print_banner(log_file, "Parsed Input File")
    io = open(input_file, "r")
    for line in eachline(io)
        println(log_file.io, line)
    end
    close(io)
    data = TOML.parsefile(input_file)
    functions = _parse_function_settings(log_file, data, T)
    bcs = _parse_boundary_condition_settings(log_file, data, functions)
    ics = _parse_initial_condition_settings(log_file, data, functions)
    println(log_file.io, "HERER")
    mesh = _parse_mesh_settings(log_file, data)
    # fspaces = _parse_function_space_settings(log_file, data)
    # vars = _parse_variable_settings(log_file, data)
    # return InputSettings{T}(fspaces, functions, mesh, vars)
    return InputSettings{T}(bcs, functions, ics, mesh)
end

#######################################################
# MeshIO strongly typed helpers
#######################################################
function read_exodus_mesh(mesh_settings::MeshSettings)
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

    # finally setup mesh
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
    return mesh
end

function _setup_function_space(log_file::LogFile, settings::InputSettings, mesh, name)
    # fspaces = Dict{String, FunctionSpace}()
    # println("Num fspaces = $(length(settings.function_spaces))")
    # for (k, v) in settings.function_spaces
    #     name = k::String
    #     temp = v
    # println(log_file.io, "Fspace name = $name")
    temp = settings.function_spaces[name]
    if lowercase(temp.field_type) == "h1"
        if lowercase(temp.polynomial_type) == "lagrange"
            return FunctionSpace{true}(mesh, H1Field, Lagrange)
        else
            error_message(log_file.io, "Unsupported polynomial_type $(temp.polynomial_type)")
        end
    else
        error_message(log_file.io, "Unsupported field_type $(temp.field_type)")
    end
    # end
    # return fspaces
end

function _setup_initial_conditions(log_file::LogFile, settings::InputSettings)
    for ic in settings.ics

    end
end

function _setup_mesh(log_file::LogFile, settings::InputSettings)
    if lowercase(settings.mesh.file_type) == "exodus"
        return read_exodus_mesh(settings.mesh)
    else
        error_message(log_file.io, "Unsupported mesh type $(settings.mesh.file_type)")
    end
end

function _setup_variables(log_file::LogFile, settings::InputSettings, mesh)
    vars = Dict{String, Any}()
    for (k, v) in settings.variables
        var_name = k::String
        fspace_name = v.function_space
        println(log_file.io, "Setting up variable $(var_name) as function from space $(fspace_name)")
        fspace_settings = settings.function_spaces[fspace_name]
        # fspace = _setup_function_space(log_file, settings, mesh, fspace_name)
        # println(log_file.io, fspace)

        
        # if v.variable_type == "scalar"
        #     var = ScalarFunction(fspace, var_name)
        #     # println(log_file.io, var)
        # else
        #     error_message(log_file, "Unsupported variable type $(v.variable_type)")
        # end

        if lowercase(fspace_settings.field_type) == "h1"
            if lowercase(fspace_settings.polynomial_type) == "lagrange"
                if lowercase(v.variable_type) == "scalar"
                    fspace = FunctionSpace{true}(mesh, H1Field, Lagrange)
                    var = ScalarFunction{typeof(fspace)}(fspace, var_name)
                else
                    error_message(log_file, "Unsupported variable type $(v.variable_type)")
                end
            else
                error_message(log_file, "Unsupported polynomial type $(v.polynomial_type)")
            end
        else
            error_message(log_file, "Unsupported field type $(v.field_type)")
        end
    end
end

#########################################################################
# main app type
#########################################################################
struct App
    cli_arg_parser::CLIArgParser
    name::String

    function App(name::String)
        cli_arg_parser = CLIArgParser()
        new(cli_arg_parser, name)
    end
end

function add_cli_arg!(app::App, name::String; kwargs...)
    add_cli_arg!(app.cli_arg_parser, name; kwargs...)
    return nothing
end

function get_cli_arg(app::App, name::String)
    return get_cli_arg(app.cli_arg_parser, name)
end

struct Simulation{T <: Number, IO, Mesh}
    ics::Vector{InitialCondition{ExpressionFunction{T}}}
    log_file::LogFile{IO}
    mesh::Mesh

    function Simulation(settings::InputSettings, log_file::LogFile{IO}) where IO
        print_banner(log_file, "Mesh")
        mesh = _setup_mesh(log_file, settings)
        println(log_file.io, mesh)
        # print_banner(log_file, "Variables")
        # _setup_variables(log_file, settings, mesh)
        # print_banner(log_file, "Function spaces")
        # fspaces = _setup_function_spaces(log_file, settings, mesh)
        # for (k, fspace) in fspaces
        #     # println(log_file.io, "$k:")
        #     println(log_file.io, fspace::FunctionSpace)
        # end
        if length(settings.ics) > 1
            T = eltype(settings.ics[1])
        else
            T = Float64
        end
        new{T, IO, typeof(mesh)}(settings.ics, log_file, mesh)
    end
end

function setup(app::App, args::Vector{String})
    parse!(app.cli_arg_parser, args)
    log_file = LogFile(get_cli_arg(app, "--log-file"))
    try
        print_banner(log_file, "CLI Arguments")
        print_dict(log_file.io, app.cli_arg_parser.parsed_args)
        input_settings = parse_input_file(app.cli_arg_parser, log_file)
        return Simulation(input_settings, log_file)
    catch e
        close(log_file)
        throw(e)
    end
end

#########################################################################
# tools to create new projects
#########################################################################
function generate_app(
    name::String;
    backends::Vector{String} = ["cpu"],
    directory::String = pwd(),
    trim::Bool = false
)
    path = joinpath(directory, name)
    @info "Creating new FiniteElementContainers app at $path"

    # create directory
    mkdir(path)
    mkdir(joinpath(path, "src"))

    compats = Pair{String, String}[
        "Exodus" => "0.14",
        "FiniteElementContainers" => "0.13"
    ]
    deps = Pair{String, String}[
        "Exodus" => "f57ae99e-f805-4780-bdca-96e224be1e5a",
        "FiniteElementContainers" => "d08262e4-672f-4e7f-a976-f2cea5767631"
    ]

    if "cuda" in backends
        push!(compats, "CUDA" => "6")
        push!(deps, "CUDA" => "052768ef-5323-5732-b1bb-66c8b64840ba")
    end

    if "mpi" in backends
        push!("deps", "PartitionedArrays" => "5a9dfac6-5c52-46f7-8278-5e2210713be9")
    end

    if "rocm" in backends
        push!(compats, "AMDGPU" => "2")
        push!(deps, "AMDGPU" => "21141c5a-9bdb-4563-92ae-f87d6854732e")
    end

    sort!(compats; by = x -> x[1])
    compats_string = ""
    for compat in compats
        compats_string *= compat[1] * " = \"" * compat[2] * "\"\n"
    end

    sort!(deps; by = x -> x[1])
    deps_string = ""
    for dep in deps
        deps_string *= dep[1] * " = \"" * dep[2] * "\"\n"
    end

    toml = """
    name = "$name"

    [deps]
    $(deps_string)
    [compat]
    $(compats_string)
    """

    open(joinpath(path, "Project.toml"), "w") do io
        print(io, toml)
    end

    # create basic src file
    open(joinpath(path, "src", "$(name).jl"), "w") do io
        src = """
        import FiniteElementContainers.AppTools as AT

        function @main(ARGS::Vector{String})
            return 0
        end
        """
        print(io, src)
    end

    # create basic juliac build file
    if trim
        trim_str = "safe"
    else
        trim_str = "no"
    end
    open(joinpath(path, "build.jl"), "w") do io
        src = """
        using JuliaC
        build_path = joinpath(@__DIR__, "build")
        src_path = joinpath(@__DIR__)
        @show build_path
        @show src_path
        rm(build_path; force=true, recursive=true)

        img = ImageRecipe(
            output_type    = "--output-exe",
            file           = "\$src_path/src/$name.jl",
            trim_mode      = "$trim_str",
            add_ccallables = false,
            verbose        = true,
        )

        link = LinkRecipe(
            image_recipe = img,
            outname      = "\$build_path/$(lowercase(name))"
        )

        bun = BundleRecipe(
            link_recipe = link,
            output_dir  = build_path # or `nothing` to skip bundling
        )

        compile_products(img)
        link_products(link)
        bundle_products(bun)
        """
        print(io, src)
    end
end

end # module AppTools
