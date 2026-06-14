module AppTools

export App

import ..DirichletBC
import ..Expressions.ScalarExpressionFunction
import ..Expressions.VectorExpressionFunction
import ..ExodusMesh
import ..FileMesh
import ..FunctionSpace
import ..H1Field
import ..InitialCondition
import ..InputFileParser
import ..NeumannBC
import ..PeriodicBC
import ..RobinBC
import ..Source
import ..TimeStepper
import ..UnstructuredMesh
import ..element_blocks
import ..element_ids
import ..nodal_coordinates_and_ids
import ..nodesets
import ..sidesets
using Exodus
using ..InputFileParser
using ReferenceFiniteElements
using TOML

function _print_error_message(io::IO, msg::String)
    println(io, msg)
end

function error_message(io::IO, msg::String)
    @assert false _print_error_message(io, msg)
end

# this one isn't quite working yet for some reason
function has_key_check(io::IO, d::Dict{String, Any}, k::String, msg::String)
    if !haskey(d, k)
        error_message(io, msg)
    end
    return nothing
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
        help_message(p, "")
        exit()
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
struct FunctionSettings{N, T <: Number}
    dict::Dict{String, Any}
    scalar_expr_funcs::Dict{String, ScalarExpressionFunction{T}}
    vector_expr_funcs::Dict{String, VectorExpressionFunction{N, T}}

    function FunctionSettings{N, T}(log_file, parser) where {N, T <: Number}
        print_banner(log_file, "Functions")
        scalar_functions = Dict{String, ScalarExpressionFunction{T}}()
        vector_functions = Dict{String, VectorExpressionFunction{N, T}}()
        if haskey(parser, "functions")
            func_settings = parser["functions"]::Dict{String, Any}
            for (k, v) in pairs(func_settings)
                name = k::String
                temp = v::Dict{String, Any}
                type = temp["type"]::String
                vars = InputFileParser.get_string_array(temp, "variables", parser.input_style)

                # TODO constant and anaytic are really the same
                # should we fuse these into "expression" or something like that?
                if type == "scalar expression"
                    expr = temp["expression"]::String
                    expr = String(strip(expr, '"')) # for yaml
                    println(log_file.io, "Parsing analytic function with expression = $expr")
                    scalar_functions[name] = ScalarExpressionFunction{T}(expr, vars)
                elseif type == "vector expression"
                    exprs = InputFileParser.get_string_array(temp, "expressions", parser.input_style)
                    for (n, expr) in enumerate(exprs)
                        exprs[n] = String(strip(expr, '"'))
                    end
                    println(log_file.io, "Parsing expression function expressions")
                    for expr in exprs
                        println(log_file.io, expr)
                    end
                    vector_functions[name] = VectorExpressionFunction{N, T}(exprs, vars)
                else
                    @assert false "Unsupported function type $type"
                end
            end
        else
            func_settings = Dict{String, Any}()
        end
        return new{N, T}(func_settings, scalar_functions, vector_functions)
    end
end

struct BCSettings{N, T <: Number}
    dict::Dict{String, Any}
    dirichlet::Vector{DirichletBC{ScalarExpressionFunction{T}}}
    neumann::Vector{NeumannBC{VectorExpressionFunction{N, T}}}
    periodic::Vector{PeriodicBC{ScalarExpressionFunction{T}}}
    robin::Vector{RobinBC{VectorExpressionFunction{N, T}}}
    source::Vector{Source{VectorExpressionFunction{N, T}}}

    function BCSettings{N, T}(log_file, parser, functions::FunctionSettings{N, T}) where {N, T <: Number}
        print_banner(log_file, "Boundary conditions")
        if haskey(parser, "boundary conditions")
            bc_settings = InputFileParser.get_nested_block(parser, "boundary conditions")
        else
            bc_settings = Dict{String, Any}()
        end
        dbcs = DirichletBC{ScalarExpressionFunction{T}}[]
        nbcs = NeumannBC{VectorExpressionFunction{N, T}}[]
        pbcs = PeriodicBC{ScalarExpressionFunction{T}}[]
        rbcs = RobinBC{VectorExpressionFunction{N, T}}[]
        srcs = Source{VectorExpressionFunction{N, T}}[]
        if haskey(bc_settings, "dirichlet")
            dbc_settings = bc_settings["dirichlet"]::Vector{Any}
            for bc in dbc_settings
                # has_key_check(log_file, bc, "function", "key \"function\" not found for DirichletBC")
                # has_key_check(log_file, bc, "variable", "key \"variable\" not found for DirichletBC")
                temp = bc::Dict{String, Any}
                func = temp["function"]::String
                func = functions.scalar_expr_funcs[func]
                vars = InputFileParser.get_string_array(temp, "variables", parser.input_style)
                if haskey(temp, "blocks")
                    blocks = InputFileParser.get_string_array(temp, "blocks", parser.input_style)
                    for block in blocks
                        for var in vars
                            push!(dbcs, DirichletBC(var, func; block_name = block))
                        end
                    end
                elseif haskey(temp, "node sets")
                    node_sets = InputFileParser.get_string_array(temp, "node sets", parser.input_style)
                    for node_set in node_sets
                        for var in vars
                            push!(dbcs, DirichletBC(var, func; nodeset_name = node_set))
                        end
                    end
                elseif haskey(temp, "side sets")
                    side_sets = InputFileParser.get_string_array(temp, "side sets", parser.input_style)
                    for side_set in side_sets
                        for var in vars
                            push!(dbcs, DirichletBC(var, func; sideset_name = side_set))
                        end
                    end
                else
                    error_message(log_file, "Requires block, node_set, or side_set")
                end
            end
        end
    
        if haskey(bc_settings, "neumann")
            nbc_settings = bc_settings["neumann"]::Vector{Any}
            for bc in nbc_settings
                temp = bc::Dict{String, Any}
                func = temp["function"]::String
                func = functions.vector_expr_funcs[func]
                sidesets = temp["side sets"]::Vector{String}
                vars = temp["variables"]::Vector{String}
                for side_set in sidesets
                    for var in vars
                        push!(nbcs, NeumannBC(var, func, side_set))
                    end
                end
            end
        end
    
        if haskey(bc_settings, "periodic")
            # TODO
            @assert false
        end

        if haskey(bc_settings, "robin")
            rbc_settings = bc_settings["robin"]::Vector{Any}
            for bc in rbc_settings
                temp = bc::Dict{String, Any}
                func = temp["function"]::String
                func = functions.vector_expr_funcs[func]
                sidesets = InputFileParser.get_string_array(temp, "side sets", parser.input_style)
                vars = InputFileParser.get_string_array(temp, "variables", parser.input_style)
                for side_set in sidesets
                    for var in vars
                        push!(rbcs, RobinBC(var, func, side_set))
                    end
                end
            end
        end
    
        if haskey(bc_settings, "source")
            src_settings = bc_settings["source"]::Vector{Any}
            for src in src_settings
                temp = src::Dict{String, Any}
                blocks = InputFileParser.get_string_array(temp, "blocks", parser.input_style)
                func = temp["function"]::String
                func = functions.vector_expr_funcs[func]
                sidesets = InputFileParser.get_string_array(temp, "blocks", parser.input_style)
                vars = InputFileParser.get_string_array(temp, "variables", parser.input_style)
                for block in blocks
                    for var in vars
                        push!(srcs, Source(var, func, block))
                    end
                end
            end
        end
    
        new{N, T}(bc_settings, dbcs, nbcs, pbcs, rbcs, srcs)
    end
end

struct ICSettings{T <: Number}
    ics::Vector{InitialCondition{ScalarExpressionFunction{T}}}

    function ICSettings{T}(log_file, parser, functions::FunctionSettings{N, T}) where {N, T}
        print_banner(log_file, "Initial conditions")
        ics = InitialCondition{ScalarExpressionFunction{Float64}}[]
        if haskey(parser, "initial conditions")
            ic_settings = parser["initial conditions"]::Vector{Any}
            for ic in ic_settings
                temp = ic::Dict{String, Any}
                func = temp["function"]::String
                func = functions.scalar_expr_funcs[func]
                vars = InputFileParser.get_string_array(temp, "variables", parser.input_style)
                if haskey(temp, "blocks")
                    blocks = InputFileParser.get_string_array(temp, "blocks", parser.input_style)
                    for block in blocks
                        for var in vars
                            push!(ics, InitialCondition(var, func; block_name = block))
                        end
                    end
                elseif haskey(temp, "node sets")
                    nodesets = InputFileParser.get_string_array(temp, "node sets", parser.input_file)
                    for nodeset in nodesets
                        for var in vars
                            push!(ics, InitialCondition(var, func; nodeset_name = nodeset))
                        end
                    end
                elseif haskey(temp, "side sets")
                    sidesets = InputFileParser.get_string_array(temp, "side sets", parser.input_style)
                    for sideset in sidesets
                        for var in vars
                            push!(ics, InitialCondition(var, func; sideset_name = sideset))
                        end
                    end
                else
                    @assert false "Couldn't find blocks, nodesets, or sidesets."
                end
            end
        end
        new{T}(ics)
    end
end

struct MaterialSettings
    materials::Dict{String, Any}

    function MaterialSettings(log_file, parser)
        if haskey(parser, "materials")
            materials = InputFileParser.get_nested_block(parser, "materials")
        else
            materials = Dict{String, Any}()
        end
        new(materials)
    end
end

struct MeshSettings
    dimension::Int
    file_path::String
    file_type::String

    function MeshSettings(log_file, parser)
        mesh_settings = InputFileParser.get_nested_block(parser, "mesh")
        dimension = mesh_settings["dimension"]::Int
        file_path = mesh_settings["file path"]::String
        file_type = lowercase(mesh_settings["file type"]::String)
        new(dimension, file_path, file_type)
    end
end

struct TimeSettings{T <: Number}
    dict::Dict{String, Any}
    time::Union{Nothing, TimeStepper{T}}

    function TimeSettings(log_file, parser)
        if haskey(parser, "time")
            time_settings = InputFileParser.get_nested_block(parser, "time")
            end_time = time_settings["end time"]::Float64
            start_time = time_settings["start time"]::Float64
            if haskey(time_settings, "number of time steps")
                nt = time_settings["number of time steps"]::Int
                Δt = (end_time - start_time) / nt
            elseif haskey(time_settings, "time step")
                Δt = time_settings["time step"]::Float64
            else
                @assert false "Requires either one of \"number of time steps\" or \"time step\" commands."
            end
            new{Float64}(time_settings, TimeStepper(start_time, end_time, start_time, Δt))
        else
            new{Float64}(Dict{String, Any}(), nothing)
        end
    end
end

struct InputSettings{N, T <: Number}
    bcs::BCSettings{N, T}
    functions::FunctionSettings{N, T}
    ics::ICSettings{T}
    materials::MaterialSettings
    mesh::MeshSettings
    parser::InputFileParser.Parser
    time::TimeSettings{T}
end

function InputSettings{N}(cli_args::CLIArgParser, log_file::LogFile, ::Type{T} = Float64) where {N, T <: Number}
    input_file = cli_args.parsed_args["--input-file"]
    println(log_file.io, "Parsing input file from $(input_file)")
    println(log_file.io, "Input file contents...")
    print_banner(log_file, "Parsed Input File")
    io = open(input_file, "r")
    for line in eachline(io)
        println(log_file.io, line)
    end
    close(io)
    parser = InputFileParser.Parser(input_file)

    functions = FunctionSettings{N, T}(log_file, parser)
    bcs = BCSettings{N, T}(log_file, parser, functions)
    ics = ICSettings{T}(log_file, parser, functions)
    materials = MaterialSettings(log_file, parser)
    mesh = MeshSettings(log_file, parser)
    time = TimeSettings(log_file, parser)
    return InputSettings{N, T}(bcs, functions, ics, materials, mesh, parser, time)
end

#######################################################
# MeshIO strongly typed helpers
#######################################################
# TODO this needs to have dimension as compile time constant...
function read_exodus_mesh(mesh_settings::MeshSettings, ::Val{D}) where D
    mesh_path = joinpath(pwd(), mesh_settings.file_path)
    exo = ExodusDatabase{Int32, Int32, Int32, Float64}(mesh_path, "r")
    fm = FileMesh{
        ExodusDatabase{Int32, Int32, Int32, Float64},
        ExodusMesh
    }(mesh_path, exo)
    # read nodes
    # if mesh_settings.dimension == 1
    coords_type = H1Field{Float64, Vector{Float64}, D}

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
        D, Float64, Int, Nothing, Nothing
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

function _setup_mesh(log_file::LogFile, settings::InputSettings, ::Val{D}) where D
    @assert settings.mesh.dimension == D
    if lowercase(settings.mesh.file_type) == "exodus"
        return read_exodus_mesh(settings.mesh, Val{D}())
    else
        error_message(log_file.io, "Unsupported mesh type $(settings.mesh.file_type)")
    end
end

#########################################################################
# main app type
#########################################################################
struct App{D, N}
    cli_arg_parser::CLIArgParser
    name::String

    function App{D, N}(name::String) where {D, N}
        cli_arg_parser = CLIArgParser()
        new{D, N}(cli_arg_parser, name)
    end
end

function add_cli_arg!(app::App, name::String; kwargs...)
    add_cli_arg!(app.cli_arg_parser, name; kwargs...)
    return nothing
end

function get_cli_arg(app::App, name::String)
    return get_cli_arg(app.cli_arg_parser, name)
end

function setup(app::App{D, N}, args::Vector{String}) where {D, N}
    parse!(app.cli_arg_parser, args)
    log_file = LogFile(get_cli_arg(app, "--log-file"))
    try
        print_banner(log_file, "CLI Arguments")
        print_dict(log_file.io, app.cli_arg_parser.parsed_args)
        input_settings = InputSettings{N}(app.cli_arg_parser, log_file)
        return Simulation{D, N}(input_settings, log_file)
    catch e
        close(log_file)
        throw(e)
    end
end

struct Simulation{D, N, T <: Number, IO, Mesh}
    dbcs::Vector{DirichletBC{ScalarExpressionFunction{T}}}
    ics::Vector{InitialCondition{ScalarExpressionFunction{T}}}
    input_settings::InputSettings{N, T}
    log_file::LogFile{IO}
    mesh::Mesh
    nbcs::Vector{NeumannBC{VectorExpressionFunction{N, T}}}
    pbcs::Vector{PeriodicBC{ScalarExpressionFunction{T}}}
    rbcs::Vector{RobinBC{VectorExpressionFunction{N, T}}}
    srcs::Vector{Source{VectorExpressionFunction{N, T}}}

    function Simulation{D, N}(settings::InputSettings, log_file::LogFile{IO}) where {D, N, IO}
        print_banner(log_file, "Mesh")
        mesh = _setup_mesh(log_file, settings, Val{D}())
        println(log_file.io, mesh)
        # print_banner(log_file, "Variables")
        # _setup_variables(log_file, settings, mesh)
        # print_banner(log_file, "Function spaces")
        # fspaces = _setup_function_spaces(log_file, settings, mesh)
        # for (k, fspace) in fspaces
        #     # println(log_file.io, "$k:")
        #     println(log_file.io, fspace::FunctionSpace)
        # end
        if length(settings.ics.ics) > 1
            T = eltype(settings.ics.ics[1])
        else
            T = Float64
        end
        new{D, N, T, IO, typeof(mesh)}(
            settings.bcs.dirichlet, settings.ics.ics, 
            settings, log_file, mesh,
            settings.bcs.neumann, settings.bcs.periodic,
            settings.bcs.robin, settings.bcs.source
        )
    end
end

#########################################################################
# tools to create new projects
#########################################################################
function build_app(; path::String = pwd())
    run(`julia --project=$path $(joinpath(path, "build.jl"))`)
end

function generate_app(
    name::String;
    backends::Vector{String} = ["cpu"],
    directory::String = pwd(),
    trim::Bool = true
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
        push!(deps, "PartitionedArrays" => "5a9dfac6-5c52-46f7-8278-5e2210713be9")
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
        import FiniteElementContainers as FEC
        import FiniteElementContainers.AppTools as AT
        using Exodus
        using FiniteElementContainers

        function app_main(ARGS::Vector{String})
            app = AT.App(\"$(name)\")
            sim = AT.setup(app, ARGS)
            println(sim.log_file.io, "Setup complete")
        end

        function @main(ARGS::Vector{String})
            app_main(ARGS)
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
        rm(build_path; force = true, recursive = true)

        img = ImageRecipe(
            output_type    = "--output-exe",
            file           = "\$src_path/src/$name.jl",
            trim_mode      = "$trim_str",
            add_ccallables = false,
            verbose        = false,
        )

        link = LinkRecipe(
            image_recipe = img,
            outname      = "\$build_path/$(lowercase(name))"
        )

        bun = BundleRecipe(
            link_recipe = link,
            #output_dir  = build_path # or `nothing` to skip bundling
            output_dir  = nothing
        )
        compile_products(img)
        link_products(link)
        bundle_products(bun)
        """
        print(io, src)
    end
end

function run_app(
    args::Vector{String};
    exe_name::Union{Nothing, String} = nothing,
    path::String = pwd()
)
    # first figure out exe name
    if exe_name === nothing
        data = TOML.parsefile(joinpath(path, "Project.toml"))
        exe_name = lowercase(data["name"])
    end
    run_cmds = pushfirst!(args, joinpath(path, "build", "bin", exe_name))
    run(Cmd(run_cmds))
end

end # module AppTools
