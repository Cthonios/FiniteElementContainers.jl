# TODO add hessians as optional
struct FunctionSpace{N, D, Rtype, L1}
	q_offset::Int
	X_ξs::Vector{SVector{D, Rtype}}
	Ns::Vector{SVector{N, Rtype}}
	∇N_Xs::Vector{SMatrix{N, D, Rtype, L1}}
	JxWs::Vector{Rtype}
end

Base.getindex(f::FunctionSpace, q::Int, e::Int) = 
(; 
	:X_ξ  => f.X_ξs[f.q_offset * (e - 1) + q], 
	:N    => f.Ns[q],
	:∇N_X => f.∇N_Xs[f.q_offset * (e - 1) + q],
	:JxW  => f.JxWs[f.q_offset * (e - 1) + q]
)
Base.length(f::FunctionSpace) = length(f.X_ξs)
Base.ndims(f::FunctionSpace) = 1
Base.size(f::FunctionSpace) = length(f)
Base.eachindex(f::FunctionSpace) = Base.OneTo(length(f))
Base.iterate(f::FunctionSpace, state = 1) = state > length(f) ? nothing : (getindex(f))

function FunctionSpace(
	mesh::Mesh{Rtype, I, B}, 
	block::Block{I, B},
	re::ReferenceFE{Itype, N, D, Rtype, L1, L2}
) where {Rtype, I, B, Itype, N, D, L1, L2}

	coords    = mesh.coords
	el_coords = @views reinterpret(SMatrix{D, N, Rtype, L1}, vec(coords[:, block.conn]))
	q_offset  = length(re.interpolants)
	X_ξs      = Matrix{SVector{D, Rtype}}(undef, length(re.interpolants), block.num_elem)
	Ns        = re.interpolants.N
	∇N_Xs = Matrix{SMatrix{N, D, Rtype, L1}}(undef, length(re.interpolants), block.num_elem)
	JxWs  = Matrix{Rtype}(undef, length(re.interpolants), block.num_elem)
	setup_quadrature_point_coordinates!(X_ξs, el_coords, re)
	setup_shape_function_gradients_and_JxWs!(∇N_Xs, JxWs, el_coords, re)
	return FunctionSpace{N, D, Rtype, L1}(q_offset, X_ξs |> vec, Ns, ∇N_Xs |> vec, JxWs |> vec)
end

function FunctionSpace(
	mesh::Mesh{Rtype, I, B}, 
	block_id::Int, 
	q_degree::Int
) where {Rtype, I, B}
	# unpack stuff from mesh
  block  = filter(x -> x.id == block_id, mesh.blocks)[1]

  # make reference finite element for this block
  el_type = element_types[uppercase(block.elem_type)](q_degree)
  re      = ReferenceFE(el_type, I, Rtype)
	return FunctionSpace(mesh, block, re)
end
