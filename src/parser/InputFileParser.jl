module InputFileParser

include("SimpleYAML.jl")

using .SimpleYAML
using TOML

const TOML_INPUT = 1
const YAML_INPUT = 2

mutable struct Parser
    data::Dict{String, Any}
    input_file::String
    input_style::Int
end

function Parser(input_file::String)
    _, ext = splitext(input_file)
    if ext == ".toml"
        data = TOML.parsefile(input_file)
        input_style = TOML_INPUT
    elseif ext == ".yaml"
        data = SimpleYAML.parsefile(input_file)
        data = SimpleYAML.to_dict(data)
        input_style = YAML_INPUT
    else
        @assert false
    end
    return Parser(data, input_file, input_style)
end

function Base.getindex(parser::Parser, key::String)
    return parser.data[key]
end

function Base.haskey(parser::Parser, key::String)
    return haskey(parser.data, key)
end

function get_nested_block(parser::Parser, key::String)
    val = parser.data[key]::Dict{String, Any}
    return val
end

function get_string_array(parser::Dict, key::String, input_style::Int)::Vector{String}
    if input_style == TOML_INPUT
        return parser[key]::Vector{String}
    elseif input_style == YAML_INPUT
        strs = parser[key]::Vector{Any}
        vals = String[]
        for str in strs
            val = str::String
            push!(vals, val)
        end
        return vals
    end
end

function get_string_array(parser::Parser, key::String)::Vector{String}
    if parser.input_style == TOML_INPUT
        return parser[key]::Vector{String}
    elseif parser.input_style == YAML_INPUT
        strs = parser[key]::Vector{Any}
        vals = String[]
        for str in strs
            val = str::String
            push!(vals, val)
        end
        return vals
    end
end

# juliac unsafe but easier
function get_value(parser::Parser, key::String)
    return parser[key]
end

# juliac safe but need to know type
function get_value(parser::Parser, key::String, type::Type{T}) where T
    val = parser[key]::type
    return val
end

end # module InputFileParser
