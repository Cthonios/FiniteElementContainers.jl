# TODO
# need to add L2 element and L2 quadrature dofs. They don't need to deliniate between
# bcs or unknowns
#
# deliniate between different var types e.g. H1, Hdiv, etc.
#
# TODO
# Should we create a small DofManager for single functions spaces
# and one for mixed function spaces for an easier interface? Yes.
# what would be the container though? A namedtuple? Or just a general
# one?
struct NewDofManager{
  T, IDs <: AbstractArray{T, 1}, 
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}
  H1_bc_dofs::IDs
  H1_unknown_dofs::IDs
  Hcurl_bc_dofs::IDs
  Hcurl_unknown_dofs::IDs
  Hdiv_bc_dofs::IDs
  Hdiv_unknown_dofs::IDs
  L2_element_dofs::IDs
  L2_quadrature_dofs::IDs
  # TODO make bins of vars
  H1_vars::H1Vars
  Hcurl_vars::HcurlVars
  Hdiv_vars::HdivVars
  L2_element_vars::L2EVars
  L2_quadrature_vars::L2QVars
end

# below not correct if fspaces differ
# right now this assumes everything is nodal as well
# this won't work for H(div) or H(curl) spaces
# we probably need to organize things by 
# node, edge, face, cell
# or
# cell, face, edge, node
# some differences in 2D vs. 3D as well
# TODO change this so it reads the fspace type off of some method
function NewDofManager(vars...)

  H1_vars = ()
  Hcurl_vars = ()
  Hdiv_vars = ()
  L2_element_vars = ()
  L2_quadrature_vars = ()
  for var in vars
    if isa(var.fspace.fspace_type, H1)
      H1_vars = (H1_vars..., var)
    elseif isa(var.fspace.fspace_type, Hcurl)
      Hcurl_vars = (Hcurl_vars..., var)
    elseif isa(var.fspace.fspace_type, Hdiv)
      Hdiv_vars = (Hdiv_vars..., var)
    elseif isa(var.fspace.fspace_type, L2Element)
      L2_element_vars = (L2_element_vars..., var)
    elseif isa(var.fspace.fspace_type, L2Quadrature)
      L2_quadrature_vars = (L2_quadrature_vars..., var)
    else
      @assert false "Bad variable type $(typeof(var))"
    end
  end

  # maybe there's a cleaner way?
  if length(H1_vars) > 0
    n_H1_dofs = sum(length.(H1_vars))
  else
    n_H1_dofs = 0
  end

  if length(Hcurl_vars) > 0
    # n_Hcurl_dofs = mapreduce(x -> length(x), sum, Hcurl_vars)
    n_Hcurl_dofs = sum(length.(Hcurl_vars))
  else
    n_Hcurl_dofs = 0
  end

  if length(Hdiv_vars) > 0
    # n_Hdiv_dofs = mapreduce(x -> length(x), sum, Hdiv_dofs)
    n_Hdiv_dofs = sum(length.(Hdiv_vars))
  else
    n_Hdiv_dofs = 0
  end

  if length(L2_element_vars) > 0
    # n_L2_element_dofs = mapreduce(x -> length(x), sum, L2_element_vars)
    n_L2_element_dofs = sum(length.(L2_element_vars))
  else
    n_L2_element_dofs = 0
  end

  if length(L2_quadrature_vars) > 0
    # n_L2_quadrature_dofs = mapreduce(x -> length(x), sum, L2_quadrature_vars)
    n_L2_quadrature_dofs = sum(length.(L2_quadrature_vars))
  else
    n_L2_quadrature_dofs = 0
  end

  # hack for now
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
  L2_element_dofs = zeros(Int, 0)
  L2_quadrature_dofs = zeros(Int, 0)

  return NewDofManager{
    eltype(H1_unknown_dofs), typeof(H1_unknown_dofs),
    # n_H1_dofs, n_Hcurl_dofs, n_Hdiv_dofs, n_L2_element_dofs, n_L2_quadrature_dofs,
    # typeof(H1_unknown_dofs), typeof(vars)
    typeof(H1_vars), typeof(Hcurl_vars), typeof(Hdiv_vars), typeof(L2_element_vars), typeof(L2_quadrature_vars)
  }(
    # H1_unknown_dofs, H1_bc_dofs, vars
    H1_bc_dofs, H1_unknown_dofs, 
    Hcurl_bc_dofs, Hcurl_unknown_dofs,
    Hdiv_bc_dofs, Hdiv_unknown_dofs,
    L2_element_dofs, L2_quadrature_dofs,
    # vars
    H1_vars, Hcurl_vars, Hdiv_vars, L2_element_vars, L2_quadrature_vars
  )
end

Base.eltype(::NewDofManager{T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars}) where {T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars} = T

Base.length(dof::NewDofManager) = length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs) +
                                  length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs) +
                                  length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs) +
                                  length(dof.L2_element_dofs) + 
                                  length(dof.L2_quadrature_dofs)

KA.get_backend(dof::NewDofManager) = KA.get_backend(dof.H1_unknown_dofs)

function create_field(dof::NewDofManager, ::Type{H1})
  backend = KA.get_backend(dof.H1_bc_dofs)
  NF, NN = num_dofs_per_node(dof), num_nodes(dof)
  field = KA.zeros(backend, Float64, NF, NN)
  syms = mapreduce(x -> names(x), sum, dof.H1_vars)
  # nt = NamedTuple{syms}(1:length(syms))
  return H1Field(field, syms)
end

function create_unknowns(dof::NewDofManager)
  n_unknowns = length(dof.H1_unknown_dofs) + 
               length(dof.Hcurl_unknown_dofs) + 
               length(dof.Hdiv_unknown_dofs) +
               length(dof.L2_element_dofs) + 
               length(dof.L2_quadrature_dofs)

  # return typeof(dof.vars[1].fspace.coords.vals)
  return KA.zeros(KA.get_backend(dof), Float64, n_unknowns)
end

# num_dofs_per_edge(::NewDofManager{T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars}) where {T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars} = NHcurl
num_dofs_per_edge(dof::NewDofManager) = length(dof.Hcurl_vars)
# num_dofs_per_face(::NewDofManager{T, NH1, NHcurl, NHdiv, NL2E, NL2Q, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, NL2E, NL2Q, IDs, Vars} = NHdiv
num_dofs_per_face(dof::NewDofManager) = length(dof.Hdiv_vars)
# num_dofs_per_node(::NewDofManager{T, NH1, NHcurl, NHdiv, NL2E, NL2Q, IDs, Vars}) where {T, NH1, NHcurl, NHdiv, NL2E, NL2Q, IDs, Vars} = NH1
num_dofs_per_node(dof::NewDofManager) = length(dof.H1_vars)

function num_edges(dof::NewDofManager)
  if length(dof.Hcurl_vars) == 0
    return 0
  else
    return (length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs)) ÷ num_dofs_per_edge(dof)
  end
end

function num_faces(dof::NewDofManager)
  if length(dof.Hdiv_vars) == 0
    return 0
  else
    return (length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs)) ÷ num_dofs_per_face(dof)
  end
end

function num_nodes(dof::NewDofManager)
  if length(dof.H1_vars) == 0
    return 0
  else
    return (length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs)) ÷ num_dofs_per_node(dof)
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

function update_field_bcs!(U::H1Field, dof::NewDofManager, Ubc::T) where T <: AbstractArray{<:Number, 1}
  AK.foreachindex(dof.H1_bc_dofs) do n
    U[dof.H1_bc_dofs[n]] = Ubc[n]
  end
  return nothing
end

function update_field_unknowns!(U::H1Field, dof::NewDofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
  AK.foreachindex(dof.H1_unknown_dofs) do n
    U[dof.H1_unknown_dofs[n]] = Uu[n]
  end
  return nothing
end

function update_field!(U::H1Field, dof::NewDofManager, Uu::T, Ubc::T) where T <: AbstractArray{<:Number, 1}
  update_field_bcs!(U, dof, Ubc)
  update_field_unknowns!(U, dof, Uu)
  return nothing
end

Base.show(io::IO, dof::NewDofManager) = 
print(io, "DofManager\n", 
          "  Number of nodes              = $(num_nodes(dof))\n",
          "  Number of dofs per node      = $(num_dofs_per_node(dof))\n",
          "  Number of H1 dofs            = $(num_nodes(dof) * num_dofs_per_node(dof))\n",
          "  Number of edges              = $(num_edges(dof))\n",
          "  Number of dofs per edge      = $(num_dofs_per_edge(dof))\n",
          "  Number of H(curl) dofs       = $(num_edges(dof) * num_dofs_per_edge(dof))\n",
          "  Number of faces              = $(num_faces(dof))\n",
          "  Number of dofs per edge      = $(num_dofs_per_face(dof))\n",
          "  Number of H(div) dofs        = $(num_faces(dof) * num_dofs_per_face(dof))\n",
          "  Number of L2 element dofs    = $(length(dof.L2_element_dofs))\n",
          "  Number of L2 quadrature dofs = $(length(dof.L2_quadrature_dofs))\n",
          "  Number of total dofs         = $(length(dof))\n",
          "  Storage type                 = $(typeof(dof.H1_unknown_dofs))")
