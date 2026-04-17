"""
All of these tools are meant to be type safe
from the perspective of working with JuliaC
trim mode for small binaries
"""
module AppTools

using TOML

# Some helper message
function _file_doesnt_exist_message(file::String)
    println(Core.stdout, "File $file doesn't exist")
end

function _help_message()
    usage = """
    TypeSafeParser usage:
    exe --input-file <name of input file.toml>

    or

    exe -i <name of input file.toml>
    """
    println(Core.stdout, usage)
end

struct CLIArgs
    input_file::String
end

# helper funcs for developers to use in apps
"""
For current asusmptions about input file see _help_message
"""
function parse_cli_args()
    println(Core.stdout, "Parsing command line arguments")
    if length(ARGS) != 2
        @assert false _help_message()
    end
    @assert ARGS[1] == "--input-file" || ARGS[1] == "-i" _help_message()
    input_file = ARGS[2]
    @assert isfile(input_file) _file_doesnt_exist_message(input_file)
    return CLIArgs(input_file)
end

struct MeshSettings
    file_path::String
    file_type::String
end

function _parse_mesh_settings(data)
    mesh_settings = data["mesh"]::Dict{String, Any}
    # meshes = Dict{String, MeshSettings}()
    # for (k, v) in mesh_settings
        # mesh_name = k::String
        # temp = v::Dict{String, Any}
    file_path = mesh_settings["file_path"]::String
    file_type = lowercase(mesh_settings["file_type"]::String)
        # meshes[mesh_name] = MeshSettings(file_path, file_type)
    # end
    return MeshSettings(file_path, file_type)
    # return meshes
end

struct InputSettings
    mesh::MeshSettings
end

function parse_input_file(cli_args::CLIArgs)
    println(Core.stdout, "Parsing input file from $(cli_args.input_file)")
    data = TOML.parsefile(cli_args.input_file)
    mesh = _parse_mesh_settings(data)
    return InputSettings(mesh)
end

#########################################################################
# Setup methods once things are parse
#########################################################################
# TODO need to think how to pull in stuff from FEC into hur
# function setup_mesh(input_settings::InputSettings)
#     mesh_settings = input_settings.meshes
#     for (k, v) in pairs(mesh_settings)
#         file_path = v.file_path
#         file_type = v.file_type
#         if lowercase(file_type) == "exodus"

#         end
#     end
# end

end # module AppTools