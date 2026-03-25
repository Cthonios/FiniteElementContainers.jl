abstract type AbstractPostProcessor end

# TODO add a csv file for global stuff
struct PostProcessor{O}
  field_output_db::O
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
  if occursin(".g", output_file) || occursin(".e", output_file) || occursin(".exo", output_file)
    pp = PostProcessor(ExodusMesh, output_file, vars...; extra_nodal_names=extra_nodal_names)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end

  return pp
end

Base.close(pp::PostProcessor) = Base.close(pp.field_output_db)

function write_field end
function write_times end
