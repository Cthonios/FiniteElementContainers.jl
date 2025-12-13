abstract type AbstractPostProcessor end

# TODO add a csv file for global stuff
struct PostProcessor{O}
  field_output_db::O
end

function PostProcessor(mesh, output_file::String, vars...)
  copy_mesh(mesh.mesh_obj.file_name, output_file)
  db_type = typeof(mesh.mesh_obj)#.name.name
  # field_output_db = FileMesh(output_file)
  # return PostProcessor(field_output_db)

  if length(vars) < 1
    @warn "no variables provided to post-processor"
  end

  ext = splitext(output_file)
  if occursin(".g", output_file) || occursin(".e", output_file) || occursin(".exo", output_file)
    pp = PostProcessor(ExodusMesh, output_file, vars...)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end

  return pp
end

Base.close(pp::PostProcessor) = Base.close(pp.field_output_db)

# function Base.write(pp::PostProcessor, n::Int, field::AbstractField)

# end

function write_field end
function write_times end
