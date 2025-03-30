abstract type AbstractPostProcessor <: FEMContainer end

# TODO add a csv file for global stuff
struct PostProcessor{O}
  field_output_db::O
end

function PostProcessor(mesh::AbstractMesh, output_file::String, vars...)
  copy_mesh(mesh.mesh_obj.file_name, output_file)
  db_type = typeof(mesh.mesh_obj)#.name.name
  # field_output_db = FileMesh(output_file)
  # return PostProcessor(field_output_db)

  if length(vars) < 1
    @warn "no variables provided to post-processor"
  end

  ext = splitext(output_file)
  if ext[2] == ".g" || ext[2] == ".e" || ext[2] == ".exo"
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
