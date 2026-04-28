abstract type AbstractPostProcessor end

# TODO add a csv file for global stuff

# shoudl change the whole struct to allow for 
# for set up all at once or sequentially
#
# need to store var names, and also allocate some memory
# here in the form of a Dict{String, Any} that can
# map simple keys like "displ" => H1Field for instance
# then this struct can handle all host side memory
# and give a Pair of key => DeviceArray we can copy
# into the host side and then write
#
struct PostProcessor{O}
  output_file_name::String
  field_output_db::O
  nodal_names::Vector{String}
  element_names::Vector{String}
  global_names::Vector{String}

  function PostProcessor(f, db::O, nn, en, gn) where O
    new{O}(f, db, nn, en, gn)
  end
end

function PostProcessor(
  mesh, output_file::String, vars...;
  extra_nodal_names::Vector{String} = String[]
)
  copy_mesh(mesh, output_file)

  if length(vars) < 1
    @warn "no variables provided to post-processor"
  end

  ext = splitext(output_file)
  if occursin(".e", output_file) || occursin(".exo", output_file)
    pp = PostProcessor(ExodusMesh, output_file, vars...; extra_nodal_names = extra_nodal_names)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end

  return pp
end

Base.close(pp::PostProcessor) = Base.close(pp.field_output_db)

# TODO finish me...
# also start using me in the exodus pp
function add_function!(pp::PostProcessor, u)
  if isa(u.fspace.coords, H1Field)
    append!(pp.nodal_names, names(u))
  # elseif isa(u.fspace.coords, L2Field)
  #   append!(pp.element_names, names(u))
  # below will be deprecated soon
  # elseif isa(u.fspace.coords, NamedTuple)
  #   max_q_points = map(num_quadrature_points, values(u.fspace.ref_fes)) |> maximum
  #   temp_names = Symbol[]
  #   for name in names(var)
  #     for q in 1:max_q_points
  #       # push!(temp_names, Symbol(String(name) * "_$q"))
  #       push!(temp_names, "$(name)_$q")
  #     end
  #   end
  else
    @assert false "Unsupported variable type"
  end
end

function finalize_setup!(pp::PostProcessor)
  if length(pp.nodal_names) > 0
    write_names(pp.field_output_db, NodalVariable, pp.nodal_names)
  end

  if length(pp.element_names) > 0
    write_names(pp.field_output_db, ElementVariable, pp.element_names)
  end
end

function write_field end
function write_times end
