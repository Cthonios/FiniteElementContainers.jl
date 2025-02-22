struct NewDofManager{T, NH1, NHcurl, NHdiv, IDs <: AbstractArray{T, 1}, Vars}
  H1_bc_dofs::IDs
  H1_unknown_dofs::IDs
  Hcurl_bc_dofs::IDs
  Hcurl_unknown_dofs::IDs
  Hdiv_bc_dofs::IDs
  Hdiv_unknown_dofs::IDs
  vars::Vars
end

# below not correct if fspaces differ
# right now this assumes everything is nodal as well
# this won't work for H(div) or H(curl) spaces
# we probably need to organize things by 
# node, edge, face, cell
# or
# cell, face, edge, node
# some differences in 2D vs. 3D as well
function NewDofManager(vars...)
  n_H1_dofs = 0
  n_Hcurl_dofs = 0
  n_Hdiv_dofs = 0
  for var in vars
    if isa(var.fspace.fspace_type, H1)
      n_H1_dofs = n_H1_dofs + length(var)
    elseif isa(var.fspace.fspace_type, Hcurl)
      n_Hcurl_dofs = n_Hcurl_dofs + length(var)
    elseif isa(var.fspace.fspace_type, Hdiv)
      n_Hdiv_dofs = n_Hdiv_dofs + length(var)
    # TODO what do to with L2 here?
    end
  end

  if length(vars) > 1
    @warn "Using multiple variables require they share FunctionSpaces currently. Checking this is satisfied..."
    for var in vars
      @assert typeof(var.fspace) == typeof(vars[1].fspace)
      @assert all(getfield(var.fspace, f) == getfield(vars[1].fspace, f) for f in fieldnames(typeof(vars[1].fspace)))
    end
  end

  # TODO fix this
  fspace = vars[1].fspace
  H1_unknown_dofs = 1:size(fspace.coords, 2) * n_H1_dofs |> collect
  H1_bc_dofs = typeof(H1_unknown_dofs)(undef, 0)

  # TODO
  # fill out Hcurl properly
  # fill out Hdiv properly
  # fill out L2Element properly
  # fill out L2Quadrature properly
  Hcurl_unknown_dofs = zeros(Int, 0)
  Hcurl_bc_dofs = zeros(Int, 0)
  Hdiv_unknown_dofs = zeros(Int, 0)
  Hdiv_bc_dofs = zeros(Int, 0)
  # L2 doesn't need to be here most likely...

  return NewDofManager{eltype(H1_unknown_dofs), n_H1_dofs, n_Hcurl_dofs, n_Hdiv_dofs, typeof(H1_unknown_dofs), typeof(vars)}(
    # H1_unknown_dofs, H1_bc_dofs, vars
    H1_bc_dofs, H1_unknown_dofs, 
    Hcurl_bc_dofs, Hcurl_unknown_dofs,
    Hdiv_bc_dofs, Hdiv_unknown_dofs,
    vars
  )
end

Base.eltype(::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} = T

Base.length(dof::NewDofManager) = length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs) +
                                  length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs) +
                                  length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs)

KA.get_backend(dof::NewDofManager) = KA.get_backend(dof.H1_unknown_dofs)

function create_field(dof::NewDofManager, ::Type{H1})
  backend = KA.get_backend(dof.H1_bc_dofs)
  NF, NN = num_dofs_per_node(dof), num_nodes(dof)
  field = KA.zeros(backend, Float64, NF, NN)
  return NodalField(field)
end

function create_unknowns(dof::NewDofManager)
  n_unknowns = length(dof.H1_unknown_dofs) + 
               length(dof.Hcurl_unknown_dofs) + 
               length(dof.Hdiv_unknown_dofs)

  # return typeof(dof.vars[1].fspace.coords.vals)
  return KA.zeros(KA.get_backend(dof), Float64, n_unknowns)
end

num_dofs_per_edge(::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} = NHcurl
num_dofs_per_face(::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} = NHdiv
num_dofs_per_node(::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} = NH1

function num_edges(dof::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} 
  if NHcurl == 0
    return 0
  else
    return (length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs)) ÷ NHcurl
  end
end

function num_faces(dof::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} 
  if NHdiv == 0
    return 0
  else
    return (length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs)) ÷ NHdiv
  end
end

function num_nodes(dof::NewDofManager{T, NH1, NHcurl, NHdiv, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, IDs, Vars} 
  if NH1 == 0
    return 0
  else
    return (length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs)) ÷ NH1
  end
end

num_unknowns(dof::NewDofManager) = length(dof.H1_unknown_dofs) + length(dof.Hcurl_unknown_dofs) + length(dof.Hdiv_unknown_dofs)

# TODO need to update to include H(div)/H(curl) spaces
function update_dofs!(dof::NewDofManager, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1}
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  resize!(dof.H1_bc_dofs, length(dirichlet_dofs))
  resize!(dof.H1_unknown_dofs, ND * NN)

  # checking dirichlet dofs make sense for this dof manager
  # for d in dirichlet_dofs
  #   @assert d >= 1 && d <= ND * NN
  # end
  AK.foreachindex(dirichlet_dofs) do i
    d = dirichlet_dofs[i]
    @assert d >= 1 && d <= ND * NN
  end

  dof.H1_bc_dofs .= dirichlet_dofs
  dof.H1_unknown_dofs .= 1:ND * NN
  deleteat!(dof.H1_unknown_dofs, dof.H1_bc_dofs)

  # checking things are re-sized appropriately at the end
  @assert length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs) == ND * NN
  return nothing
end

function update_field_bcs!(U::NodalField, dof::NewDofManager, Ubc::T) where T <: AbstractArray{<:Number, 1}
  AK.foreachindex(dof.H1_bc_dofs) do n
    U[dof.H1_bc_dofs[n]] = Ubc[n]
  end
  return nothing
end

function update_field_unknowns!(U::NodalField, dof::NewDofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
  AK.foreachindex(dof.H1_unknown_dofs) do n
    U[dof.H1_unknown_dofs[n]] = Uu[n]
  end
  return nothing
end

function update_field!(U::NodalField, dof::NewDofManager, Uu::T, Ubc::T) where T <: AbstractArray{<:Number, 1}
  update_field_bcs!(U, dof, Ubc)
  update_field_unknowns!(U, dof, Uu)
  return nothing
end

Base.show(io::IO, dof::NewDofManager) = 
print(io, "DofManager\n", 
          "  Number of nodes         = $(num_nodes(dof))\n",
          "  Number of dofs per node = $(num_dofs_per_node(dof))\n",
          "  Number of H1 dofs       = $(num_nodes(dof) * num_dofs_per_node(dof))\n",
          "  Number of edges         = $(num_edges(dof))\n",
          "  Number of dofs per edge = $(num_dofs_per_edge(dof))\n",
          "  Number of H(curl) dofs  = $(num_edges(dof) * num_dofs_per_edge(dof))\n",
          "  Number of faces         = $(num_faces(dof))\n",
          "  Number of dofs per edge = $(num_dofs_per_face(dof))\n",
          "  Number of H(div) dofs   = $(num_faces(dof) * num_dofs_per_face(dof))\n",
          "  Number of total dofs    = $(length(dof))\n",
          "  Storage type            = $(typeof(dof.H1_unknown_dofs))")
