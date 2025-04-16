# TODO
# Should we create a small DofManager for single functions spaces
# and one for mixed function spaces for an easier interface? Maybe.
# what would be the container though? A namedtuple? Or just a general
# one?
# TODO need to make NL2EDofs and NL2QDofs namedtuples Otherwise
# all blocks need have the same number of vars there
# which will not be the case for properties and state variables
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct DofManager{
  T, IDs <: AbstractArray{T, 1}, 
  # H1Syms, HcurlSyms, HdivSyms, L2ESyms, L2QSyms,
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}
  H1_bc_dofs::IDs
  H1_unknown_dofs::IDs
  Hcurl_bc_dofs::IDs
  Hcurl_unknown_dofs::IDs
  Hdiv_bc_dofs::IDs
  Hdiv_unknown_dofs::IDs
  # we can save space by having the below two fields be ranges
  L2_element_dofs::IDs
  L2_quadrature_dofs::IDs
  # bins of vars
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
"""
$(TYPEDSIGNATURES)
"""
function DofManager(vars...)
  H1_vars = _filter_field_type(vars, H1Field)
  # Hcurl_vars = _filter_field_type(vars, HcurlField)
  Hcurl_vars = NamedTuple() # TODO
  # Hdiv_vars = _filter_field_type(vars, HdivField)
  Hdiv_vars = NamedTuple() # TODO
  L2_element_vars = _filter_field_type(vars, L2ElementField)
  L2_quadrature_vars = _filter_field_type(vars, L2QuadratureField)
  # get number of dofs
  n_H1_dofs = _n_dofs_from_vars(H1_vars)
  n_Hcurl_dofs = _n_dofs_from_vars(Hcurl_vars)
  n_Hdiv_dofs = _n_dofs_from_vars(Hdiv_vars)
  n_L2_element_dofs = _n_dofs_from_vars(L2_element_vars)
  n_L2_quadrature_dofs = _n_dofs_from_vars(L2_quadrature_vars)

  # hack for now
  # TODO remove this warning
  # if length(vars) > 1
  #   @warn "Using multiple variables require they share FunctionSpaces currently. Checking this is satisfied..."
  #   for var in vars
  #     @assert typeof(var.fspace) == typeof(vars[1].fspace)
  #     @assert all(getfield(var.fspace, f) == getfield(vars[1].fspace, f) for f in fieldnames(typeof(vars[1].fspace)))
  #   end
  # end

  # TODO fix this
  # setting up H1 vars
  if length(H1_vars) > 0
    H1_unknown_dofs = 1:size(H1_vars[1].fspace.coords, 2) * n_H1_dofs |> collect
    H1_bc_dofs = typeof(H1_unknown_dofs)(undef, 0)
  else
    H1_unknown_dofs = zeros(Int, 0)
    H1_bc_dofs = zeros(Int, 0)
  end

  # TODO need coords everywhere basically, that's apparently the interace
  # if length(L2_element_vars) > 0
  #   L2_element_dofs = 1:size(L2_element_vars[1].fspace.coords, 2) * size(L2_element_vars[1].fspace.coords, 2) * n_L2_quadrature_dofs |> collect
  #   @show L2_element_dofs
  # end

  if length(L2_quadrature_vars) > 0
    # L2_quadrature_dofs = 1:size(L2_quadrature_vars[1].fspace.coords, 2) * size(L2_quadrature_vars[1].fspace.coords, 3) * n_L2_quadrature_dofs |> collect
    n_total_elements = 0
    for var in L2_quadrature_vars
      n_total_elements = n_total_elements + sum(map(x -> size(x, 3), values(var.fspace.coords)))
    end
    L2_quadrature_dofs = 1:n_L2_quadrature_dofs * n_total_elements |> collect
  else
    L2_quadrature_dofs = zeros(Int, 0)
  end

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
  # L2_quadrature_dofs = zeros(Int, 0)

  return DofManager{
    eltype(H1_unknown_dofs), typeof(H1_unknown_dofs),
    n_H1_dofs, n_Hcurl_dofs, n_Hdiv_dofs, n_L2_element_dofs, n_L2_quadrature_dofs,
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

function _dof_manager_sym_name(u::ScalarFunction)
  return names(u)[1]
end

function _dof_manager_sym_name(u::SymmetricTensorFunction)
  return Symbol(split(String(names(u)[1]), ['_'])[1])
end

function _dof_manager_sym_name(u::StateFunction)
  return Symbol(split(String(names(u)[1]), ['_'])[1])
end

function _dof_manager_sym_name(u::TensorFunction)
  return Symbol(split(String(names(u)[1]), ['_'])[1])
end

function _dof_manager_sym_name(u::VectorFunction)
  return Symbol(split(String(names(u)[1]), ['_'])[1])
end

function _dof_manager_vars(dof::DofManager, ::Type{<:H1Field})
  return dof.H1_vars
end

function _filter_field_type(vars, T)
  if T == L2QuadratureField
    # vars = map(y -> isa(y, T), values(x.fspace.coords))
    # vars = map(x -> , vars)
    temp_vars = filter(x -> isa(x.fspace.coords, NamedTuple), vars)
    var_syms = Symbol[]
    var_vals = [] # TODO type this array
    for var in temp_vars
      if isa(values(var.fspace.coords)[1], L2QuadratureField)
        push!(var_syms, names(var)[1])
        push!(var_vals, var)
      end
    end
    # @show vars
    # vars = NamedTuple{tuple(var_syms...)}(tuple(var_vals...))
    vars = tuple(var_vals...)
  else
    vars = filter(x -> isa(x.fspace.coords, T), vars)
  end
  syms = map(_dof_manager_sym_name, vars)
  return NamedTuple{syms}(vars)
end

function _n_dofs_from_vars(vars)
  if length(vars) > 0
    n_dofs = sum(length.(values(vars)))
  else
    n_dofs = 0
  end
  return n_dofs
end

function _vector_to_scalars(u::ScalarFunction)
  syms = names(u)
  return ScalarFunction.((u.fspace,), syms)
end

function _vector_to_scalars(u::VectorFunction)
  syms = names(u)
  return ScalarFunction.((u.fspace,), syms)
end

# Base.eltype(::DofManager{T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars}) where {T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars} = T

"""
$(TYPEDSIGNATURES)
"""
Base.eltype(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = T

"""
$(TYPEDSIGNATURES)
"""
Base.length(dof::DofManager) = length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs) +
                               length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs) +
                               length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs) +
                               length(dof.L2_element_dofs) + 
                               length(dof.L2_quadrature_dofs)

KA.get_backend(dof::DofManager) = KA.get_backend(dof.H1_unknown_dofs)

"""
$(TYPEDSIGNATURES)
"""
function create_bcs(dof::DofManager, ::Type{H1Field})
  backend = KA.get_backend(dof.H1_bc_dofs)
  return KA.zeros(backend, Float64, length(dof.H1_bc_dofs))
end

# this one needs some TLC
function create_bcs(dof::DofManager, ::Type{H1Field}, dirichlet_bcs, time)
  backend = KA.get_backend(dof.H1_bc_dofs)

  X = dof.H1_vars[1].fspace.coords
  # ND = num_dimensions(dof.H1_vars[1].fspace)
  ND = size(X, 1) # TODO do this in a type stable way
  # ND = num_fields(X)
  # gather all dirichlet dofs
  dirichlet_dofs = vcat(map(x -> x.bookkeeping.dofs, dirichlet_bcs)...)
  dirichlet_vals = KA.zeros(backend, Float64, 0)

  dirichlet_perm = sortperm(dirichlet_dofs)
  dirichlet_dofs = dirichlet_dofs[dirichlet_perm]
  dof_unique = unique(i -> dirichlet_dofs[i], eachindex(dirichlet_dofs))
  dirichlet_dofs = dirichlet_dofs[dof_unique]

  # dirichlet_vals = vcat(map(x -> x.func.(X[:, x.bookkeeping.nodes], (time,)), dirichlet_bcs)...)

  for bc in dirichlet_bcs
    # for node in bc.bookkeeping.nodes
    AK.foreachindex(bc.bookkeeping.nodes) do n
      node = bc.bookkeeping.nodes[n]
      X_temp = SVector{ND, Float64}(@views X[:, node])
      # @show bc.func(X_temp, time)
      push!(dirichlet_vals, bc.func(X_temp, time))
    end
  end
  # @show dirichlet_vals
  # dirichlet_dofs
  # need to get the coordinates and time
  # for bc in dirichlet_bcs

  # end
  dirichlet_vals[dirichlet_perm][dof_unique]
end


"""
$(TYPEDSIGNATURES)
"""
function create_field(dof::DofManager, ::Type{H1Field})
  backend = KA.get_backend(dof.H1_bc_dofs)
  NF, NN = num_dofs_per_node(dof), num_nodes(dof)
  field = KA.zeros(backend, Float64, NF, NN)

  syms = ()
  for var in dof.H1_vars
    syms = (syms..., names(var)...)
  end
  return H1Field(field, syms)
end

"""
$(TYPEDSIGNATURES)
"""
function create_field(dof::DofManager, ::Type{H1Field}, syms)
  backend = KA.get_backend(dof.H1_bc_dofs)
  NF, NN = num_dofs_per_node(dof), num_nodes(dof)
  field = KA.zeros(backend, Float64, NF, NN)
  @assert NF == length(syms)
  return H1Field(field, syms)
end

"""
$(TYPEDSIGNATURES)
"""
function create_field(dof::DofManager, ::Type{L2QuadratureField})
  backend = KA.get_backend(dof.L2_quadrature_dofs)
  fspace = dof.L2_quadrature_vars[1].fspace
  # NS = 
  for ref_fe in fspace.ref_fes
    @show ref_fe
    NQ = num_quadrature_points(ref_fe)

  end
end

"""
$(TYPEDSIGNATURES)
"""
function create_unknowns(dof::DofManager)
  n_unknowns = length(dof.H1_unknown_dofs) + 
               length(dof.Hcurl_unknown_dofs) + 
               length(dof.Hdiv_unknown_dofs) +
               length(dof.L2_element_dofs) + 
               length(dof.L2_quadrature_dofs)

  # return typeof(dof.vars[1].fspace.coords.vals)
  return KA.zeros(KA.get_backend(dof), Float64, n_unknowns)
end

function create_unknowns(dof::DofManager, ::Type{<:H1Field})
  n_unknowns = length(dof.H1_unknown_dofs)
  return KA.zeros(KA.get_backend(dof), Float64, n_unknowns)
end

"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_edge(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = NHcurlDofs

"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_element(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = NL2EDofs

"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_face(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = NHdivDofs

"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_node(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = NH1Dofs

"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_quadrature_point(::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
} = NL2QDofs

"""
$(TYPEDSIGNATURES)
"""
function num_edges(dof::DofManager)
  # if length(dof.Hcurl_vars) == 0
  if length(dof.Hcurl_vars) == 0
    return 0
  else
    return (length(dof.Hcurl_bc_dofs) + length(dof.Hcurl_unknown_dofs)) ÷ num_dofs_per_edge(dof)
  end
end

"""
$(TYPEDSIGNATURES)
"""
function num_faces(dof::DofManager)
  # if length(dof.Hdiv_vars) == 0
  if length(dof.Hdiv_vars) == 0
    return 0
  else
    return (length(dof.Hdiv_bc_dofs) + length(dof.Hdiv_unknown_dofs)) ÷ num_dofs_per_face(dof)
  end
end

function num_nodes(dof::DofManager)
  # if length(dof.H1_vars) == 0
  if length(dof.H1_vars) == 0
    return 0
  else
    return (length(dof.H1_bc_dofs) + length(dof.H1_unknown_dofs)) ÷ num_dofs_per_node(dof)
  end
end

"""
$(TYPEDSIGNATURES)
"""
num_unknowns(dof::DofManager) = length(dof.H1_unknown_dofs) + length(dof.Hcurl_unknown_dofs) + length(dof.Hdiv_unknown_dofs)

# TODO need to update to include H(div)/H(curl) spaces
"""
$(TYPEDSIGNATURES)
Currently not GPU compatable.

This is only an issue if dofs that correspond to Dirichlet BCs
will change often. Otherwise, setup can be achieved on the CPU
and transferred to the GPU.

TODO this method need to look at the dirichlet dofs
to see what type of variable is there. That way the 
appropriate function space book keepers can be updated.
Currently this only works with H1 spaces.
"""
function update_dofs!(dof::DofManager, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1}
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  Base.resize!(dof.H1_bc_dofs, length(dirichlet_dofs))
  Base.resize!(dof.H1_unknown_dofs, ND * NN)

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

KA.@kernel function _update_field_bcs_kernel!(U::H1Field, dof::DofManager, Ubc::T) where T <: AbstractArray{<:Number, 1}
  N = KA.@index(Global)
  @inbounds U[dof.H1_bc_dofs[N]] = Ubc[N]
end

function _update_field_bcs!(U::H1Field, dof::DofManager, Ubc::T, backend::KA.Backend) where T <: AbstractArray{<:Number, 1}
  kernel! = _update_field_bcs_kernel!(backend)
  kernel!(U, dof, Ubc, ndrange = length(Ubc))
  return nothing
end

function _update_field_bcs!(U::H1Field, dof::DofManager, Ubc::T, ::KA.CPU) where T <: AbstractArray{<:Number, 1}
  U[dof.H1_bc_dofs] .= Ubc
  return nothing
end

"""
$(TYPEDSIGNATURES)
Does a simple copy on CPUs. On GPUs it uses a ```KernelAbstractions``` kernel
"""
function update_field_bcs!(U::H1Field, dof::DofManager, Ubc::T) where T <: AbstractArray{<:Number, 1}
  _update_field_bcs!(U, dof, Ubc, KA.get_backend(dof))
  return nothing
end

# # CPU only for now
# function update_field_bcs!(U::H1Field, dof::DofManager, dbc::DirichletBC, t)
#   X_global = dof.H1_vars[1].fspace.coords
#   for (n, node) in enumerate(dbc.bookkeeping.nodes)
#     X = @views X_global[:, node]
#     dbc.vals[n] = dbc.func(X, t)
#   end
#   for (n, dof) in enumerate(dbc.bookkeeping.dofs)
#     U[dof] = dbc.vals[n]
#   end
#   return nothing
# end

# CPU only for now, implementations in bcs folders
function update_field_bcs!(U::H1Field, dof::DofManager, dbcs, t)
  for bc in dbcs
    update_field_bcs!(U, dof, bc, t)
  end
end

KA.@kernel function _update_field_unknowns_kernel!(U::H1Field, dof::DofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
  N = KA.@index(Global)
  @inbounds U[dof.H1_unknown_dofs[N]] = Uu[N]
end

function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, backend::KA.Backend) where T <: AbstractArray{<:Number, 1}
  kernel! = _update_field_unknowns_kernel!(backend)
  kernel!(U, dof, Uu, ndrange = length(Uu))
  return nothing
end

# Need a seperate CPU method since CPU is basically busted in KA
function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, ::KA.CPU) where T <: AbstractArray{<:Number, 1}
  U[dof.H1_unknown_dofs] .= Uu
  return nothing
end

"""
$(TYPEDSIGNATURES)
Does a simple copy on CPUs. On GPUs it uses a ```KernelAbstractions``` kernel
"""
function update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
  _update_field_unknowns!(U, dof, Uu, KA.get_backend(dof))
  return nothing
end

KA.@kernel function _update_field_unknowns_kernel_plus!(U::H1Field, dof::DofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
  N = KA.@index(Global)
  @inbounds U[dof.H1_unknown_dofs[N]] += Uu[N]
end

function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, ::typeof(+), backend::KA.Backend) where T <: AbstractArray{<:Number, 1}
  kernel! = _update_field_unknowns_kernel_plus!(backend)
  kernel!(U, dof, Uu, ndrange = length(Uu))
  return nothing
end

function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, ::typeof(+), ::KA.CPU) where T <: AbstractArray{<:Number, 1}
  for (n, d) in enumerate(dof.H1_unknown_dofs)
    U[d] += Uu[n]
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
Does a simple addition on CPUs. On GPUs it uses a ```KernelAbstractions``` kernel
"""
function update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, ::typeof(+)) where T <: AbstractArray{<:Number, 1}
  _update_field_unknowns!(U, dof, Uu, +, KA.get_backend(dof))
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_field!(U::H1Field, dof::DofManager, Uu::T, Ubc::T) where T <: AbstractArray{<:Number, 1}
  update_field_bcs!(U, dof, Ubc)
  update_field_unknowns!(U, dof, Uu)
  return nothing
end

Base.show(io::IO, dof::DofManager) = 
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
