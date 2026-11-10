module AdaptExt

import Adapt
import Adapt: adapt
import FiniteElementContainers as FEC
import KernelAbstractions as KA

# assemblers
function Adapt.adapt_structure(to, asm::FEC.MatrixFreeAssembler)
    return FEC.MatrixFreeAssembler(
        adapt(to, asm.dof),
        adapt(to, asm.vector_pattern),
        adapt(to, asm.constraint_storage),
        adapt(to, asm.residual_storage),
        adapt(to, asm.residual_unknowns),
        adapt(to, asm.stiffness_action_storage),
        adapt(to, asm.stiffness_action_unknowns)
    )
end

function Adapt.adapt_structure(to, asm::FEC.SparseMatrixAssembler)
  return FEC.SparseMatrixAssembler{
    FEC._is_condensed(asm.dof),
    FEC._sparse_matrix_type(asm),
    FEC._use_inplace_methods(asm),
    FEC._use_sparse_vector(asm),
  }(
    adapt(to, asm.dof),
    adapt(to, asm.matrix_pattern),
    adapt(to, asm.vector_pattern),
    adapt(to, asm.constraint_storage),
    adapt(to, asm.mass_storage),
    adapt(to, asm.residual_storage),
    adapt(to, asm.residual_unknowns),
    adapt(to, asm.scalar_quadrature_storage),
    adapt(to, asm.stiffness_storage),
    adapt(to, asm.stiffness_action_storage),
    adapt(to, asm.stiffness_action_unknowns)
  )
end

function Adapt.adapt_structure(to, asm::FEC.SparseMatrixPattern)
  return FEC.SparseMatrixPattern(
    adapt(to, asm.Is),
    adapt(to, asm.Js),
    adapt(to, asm.unknown_dofs),
    asm.block_start_indices, asm.max_entries,
    adapt(to, asm.klasttouch),
    adapt(to, asm.csrrowptr),
    adapt(to, asm.csrcolval),
    adapt(to, asm.csrnzval),
    adapt(to, asm.csccolptr),
    adapt(to, asm.cscrowval),
    adapt(to, asm.cscnzval_mass), adapt(to, asm.cscnzval_stiffness),
    adapt(to, asm.permutation)
  )
end

function Adapt.adapt_structure(to, pattern::FEC.SparseVectorPattern)
  return FEC.SparseVectorPattern(
    adapt(to, pattern.Is),
    adapt(to, pattern.permutation),
    adapt(to, pattern.unknown_dofs),
    pattern.block_start_indices,
    pattern.max_entries
  )
end

# boundary conditions
function Adapt.adapt_structure(to, bc::FEC.DirichletBCContainer)
  dofs = adapt(to, bc.dofs)
  nodes = adapt(to, bc.nodes)
  vals = adapt(to, bc.vals)
  vals_dot = adapt(to, bc.vals_dot)
  vals_dot_dot = adapt(to, bc.vals_dot_dot)
  return FEC.DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
end

function Adapt.adapt_structure(to, bcs::FEC.DirichletBCs{BCF, IV, RV}) where {BCF, IV, RV}
  # NOTE
  # below logic is needed due to improper
  # adapt mapping for an empty array in julia 1.10/1.11
  # where Vector{T}(undef, 0) gets mappend to Vector{Any}
  if length(bcs.bc_lengths) > 0
    bc_cache = adapt(to, bcs.bc_cache)
  else
    temp_int = adapt(to, zeros(Int, 0))
    temp_floats = adapt(to, zeros(Float64, 0))
    bc_cache = FEC.DirichletBCContainer(
      temp_int, copy(temp_int),
      temp_floats, copy(temp_floats), copy(temp_floats)
    )
  end
  return FEC.DirichletBCs{BCF, typeof(bc_cache.dofs), typeof(bc_cache.vals)}(
    bc_cache,
    bcs.bc_funcs,
    bcs.bc_lengths
  )
end

function Adapt.adapt_structure(to, bc::FEC.NeumannBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  vals = adapt(to, bc.vals)
  return FEC.NeumannBCContainer(el_conns, elements, sides, vals)
end

function Adapt.adapt_structure(to, bcs::FEC.NeumannBCs)
  # NOTE
  # below logic is needed due to improper
  # adapt mapping for an empty array in julia 1.10/1.11
  # where Vector{T}(undef, 0) gets mappend to Vector{Any}
  if length(bcs.bc_caches) > 0
    bc_caches = map(x -> adapt(to, x), bcs.bc_caches)
  else
    temp_int = adapt(to, zeros(Int, 0))
    temp_vals = adapt(to, FEC._vals_type(bcs.bc_caches)(undef, 0, 0))
    bc_caches = FEC.NeumannBCContainer{Int, typeof(temp_int), typeof(temp_vals)}[]
  end
  return FEC.NeumannBCs(
    bcs.block_id_to_bc,
    bc_caches,
    bcs.bc_funcs,
    bcs.block_ids,
    bcs.block_names
  )
end

function Adapt.adapt_structure(to, bc::FEC.PeriodicBCContainer)
    return FEC.PeriodicBCContainer(
        adapt(to, bc.side_a_dofs),
        adapt(to, bc.side_a_nodes),
        adapt(to, bc.side_b_dofs),
        adapt(to, bc.side_b_nodes),
        adapt(to, bc.vals)
    )
end

function Adapt.adapt_structure(to, pbcs::FEC.PeriodicBCs)
    # NOTE
    # below logic is needed due to improper
    # adapt mapping for an empty array in julia 1.10/1.11
    # where Vector{T}(undef, 0) gets mappend to Vector{Any}
    if length(pbcs) > 0
        bc_caches = map(x -> adapt(to, x), pbcs.bc_caches)
    else
        temp_int = adapt(to, zeros(Int, 0))
        temp_floats = adapt(to, zeros(Float64, 0))
        bc_cache = FEC.PeriodicBCContainer(
            copy(temp_int), copy(temp_int),
            copy(temp_int), copy(temp_int),
            temp_floats
        )
        bc_caches = Vector{typeof(bc_cache)}(undef, 0)
    end
    return FEC.PeriodicBCs(
        bc_caches,
        pbcs.bc_funcs
    )
end

function Adapt.adapt_structure(to, bc::FEC.PointSourceBCContainer)
  dofs = adapt(to, bc.dofs)
  nodes = adapt(to, bc.nodes)
  vals = adapt(to, bc.vals)
  return FEC.PointSourceBCContainer(dofs, nodes, vals)
end

function Adapt.adapt_structure(to, bcs::FEC.PointSourceBCs{BCF, IV, RV}) where {BCF, IV, RV}
  # NOTE
  # below logic is needed due to improper
  # adapt mapping for an empty array in julia 1.10/1.11
  # where Vector{T}(undef, 0) gets mappend to Vector{Any}
  if length(bcs.bc_lengths) > 0
    bc_cache = adapt(to, bcs.bc_cache)
  else
    temp_int = adapt(to, zeros(Int, 0))
    temp_floats = adapt(to, zeros(Float64, 0))
    bc_cache = FEC.PointSourceBCContainer(temp_int, copy(temp_int), temp_floats)
  end
  return FEC.PointSourceBCs{BCF, typeof(bc_cache.dofs), typeof(bc_cache.vals)}(
    bc_cache,
    bcs.bc_funcs,
    bcs.bc_lengths
  )
end

function Adapt.adapt_structure(to, bc::FEC.RobinBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  vals = adapt(to, bc.vals)
  dvalsdu = adapt(to, bc.dvalsdu)
  return FEC.RobinBCContainer(el_conns, elements, sides, vals, dvalsdu)
end

function Adapt.adapt_structure(to, bcs::FEC.RobinBCs)
  # NOTE
  # below logic is needed due to improper
  # adapt mapping for an empty array in julia 1.10/1.11
  # where Vector{T}(undef, 0) gets mappend to Vector{Any}
  if length(bcs.bc_caches) > 0
    bc_caches = map(x -> adapt(to, x), bcs.bc_caches)
  else
    temp_int = adapt(to, zeros(Int, 0))
    temp_vals = adapt(to, FEC._vals_type(bcs.bc_caches)(undef, 0, 0))
    temp_dvals = adapt(to, FEC._dvals_type(bcs.bc_caches)(undef, 0, 0))
    bc_caches = FEC.RobinBCContainer{Int, typeof(temp_int), typeof(temp_vals), typeof(temp_dvals)}[]
  end
  return FEC.RobinBCs(
    bcs.block_id_to_bc,
    bc_caches,
    bcs.bc_funcs,
    bcs.block_ids,
    bcs.block_names
  )
end

function Adapt.adapt_structure(to, source::FEC.SourceContainer)
  FEC.SourceContainer(
    adapt(to, source.vals),
    source.is_constant,
    source.initialized,
  )
end

function Adapt.adapt(to, sources::FEC.Sources{SourceFuncs, RM}) where {SourceFuncs, RM}
  if length(sources.source_caches) > 0
    caches = map(x -> adapt(to, x), sources.source_caches)
  else
    temp_floats = adapt(to, FEC._vals_type(sources.source_caches)(undef, 0, 0))
    caches = FEC.SourceContainer{typeof(temp_floats)}[]
  end

  return FEC.Sources{SourceFuncs, FEC._vals_type(caches)}(
    sources.block_id_to_source,
    sources.source_block_ids,
    sources.source_block_names,
    caches,
    sources.source_funcs
  )
end

# dof manager
function Adapt.adapt_structure(to, dof::FEC.DofManager)
    return FEC.DofManager{FEC._is_condensed(dof)}(
        adapt(to, dof.dirichlet_dofs),
        adapt(to, dof.dof_to_unknown),
        adapt(to, dof.periodic_side_a_dofs),
        adapt(to, dof.periodic_side_b_dofs),
        adapt(to, dof.periodic_side_b_to_side_a_unknown),
        adapt(to, dof.unknown_dofs),
        adapt(to, dof.var)
    )
end

# expressions

########################################################
# Adapt
#
# Both flat expression functions subtype `Function` so they can be called,
# but they are plain isbits data — no captured arrays, nothing to move to a
# device.  Adapt's generic `Function` method assumes any callable is a
# closure and infers one type parameter per captured field: for
# `ScalarExpressionFunction{T}` that is 1 type parameter against 3 fields,
# so it computes `num_static_params = -2` and throws
# `ArgumentError: tuple length should be >= 0, got -2`.  That fires the
# moment one of these is passed to a GPU kernel as a bare argument rather
# than reached through an enclosing closure.  Adapting them is the
# identity.
########################################################

Adapt.adapt_structure(to, f::FEC.Expressions.ScalarExpressionFunction) = f
Adapt.adapt_structure(to, f::FEC.Expressions.VectorExpressionFunction) = f

# fields
function Adapt.adapt_structure(to, conn::FEC.Connectivity{T, D}) where {T, D}
    return FEC.Connectivity(
        adapt(to, conn.data),
        conn.nblocks,
        conn.nepes,
        conn.nelems,
        conn.offsets
    )
end

function Adapt.adapt_structure(to, conn::FEC.Connectivity_v2)
    return FEC.Connectivity_v2(
        adapt(to, conn.data),
        conn.nblocks,
        adapt(to, conn.nepes),
        adapt(to, conn.nelems),
        adapt(to, conn.offsets)
    )
end

function Adapt.adapt_structure(to, field::FEC.H1Field{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return FEC.H1Field{T, typeof(data), NF}(data)
end

function Adapt.adapt_structure(to, field::FEC.HcurlField{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return FEC.HcurlField{T, typeof(data), NF}(data)
end

function Adapt.adapt_structure(to, field::FEC.HdivField{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return FEC.HdivField{T, typeof(data), NF}(data)
end

function Adapt.adapt_structure(to, field::FEC.L2Field{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return FEC.L2Field{T, typeof(data), NF}(
        data,
        field.nblocks,
        field.nepes,
        field.nelems,
        field.offsets
    )
end

function Adapt.adapt_structure(to, field::FEC.PropertyField)
    data = adapt(to, field.data)
    isblockconstant = adapt(to, field.isblockconstant)
    return FEC.PropertyField{eltype(field), typeof(data), typeof(isblockconstant)}(
        data,
        isblockconstant,
        field.nblocks,
        adapt(to, field.nepes),
        adapt(to, field.nelems),
        adapt(to, field.offsets)
    )
end

function Adapt.adapt_structure(to, field::FEC.StateVariableField{T, D, I}) where {T, D, I}
    data = adapt(to, field.data)
    nfields = adapt(to, field.nfields)
    return FEC.StateVariableField{T, typeof(data), typeof(nfields)}(
        data,
        field.nblocks,
        nfields,
        adapt(to, field.nepes),
        adapt(to, field.nelems),
        adapt(to, field.offsets)
    )
end

# functions
function Adapt.adapt_structure(to, f::FEC.ScalarFunction)
  fspace = adapt(to, f.fspace)
  return FEC.ScalarFunction{typeof(fspace)}(fspace, f.names)
end

function Adapt.adapt_structure(to, f::FEC.VectorFunction)
  return FEC.VectorFunction(adapt(to, f.fspace), f.names)
end

function Adapt.adapt_structure(to, f::FEC.TensorFunction)
  return FEC.TensorFunction(adapt(to, f.fspace), f.names)
end

function Adapt.adapt_structure(to, f::FEC.SymmetricTensorFunction)
  return FEC.SymmetricTensorFunction(adapt(to, f.fspace), f.names)
end

function Adapt.adapt_structure(to, f::FEC.GeneralFunction)
  return FEC.GeneralFunction(adapt(to, f.fspace), f.names)
end

# function spaces
function Adapt.adapt_structure(to, fspace::FEC.FunctionSpace)
  return FEC.FunctionSpace{FEC._is_juliac_safe(fspace), FEC._field_type(fspace)}(
    fspace.block_names, fspace.block_to_ref_fe_id,
    adapt(to, fspace.coords),
    adapt(to, fspace.elem_conns), 
    map(x -> adapt(to, x), fspace.elem_id_maps),
    adapt(to, fspace.elem_to_facets),
    adapt(to, fspace.facet_orientation),
    adapt(to, fspace.node_id_map),
    map(x -> adapt(to, x), fspace.ref_fes)
  )
end

# initial conditions
function Adapt.adapt_structure(to, ic::FEC.InitialConditionContainer)
    return FEC.InitialConditionContainer(
        adapt(to, ic.dofs),
        adapt(to, ic.locations),
        adapt(to, ic.vals)
    )
end

function Adapt.adapt_structure(to, ics::FEC.InitialConditions)
    # NOTE
    # below logic is needed due to improper
    # adapt mapping for an empty array in julia 1.10/1.11
    # where Vector{T}(undef, 0) gets mappend to Vector{Any}
    if length(ics.ic_caches) > 0
        ic_caches = map(x -> adapt(to, x), ics.ic_caches)
    else
        temp_int = adapt(to, zeros(Int, 0))
        temp_floats = adapt(to, zeros(Float64, 0))
        ic_caches = FEC.InitialConditionContainer{typeof(temp_int), typeof(temp_floats)}[]
    end

    return FEC.InitialConditions(ic_caches, ics.ic_funcs)
end

# integrals
function Adapt.adapt_storage(to, int::FEC.AbstractIntegral)
    return eval(typeof(int).name.name)(
        adapt(to, int.assembler),
        adapt(to, int.cache),
        int.integrand
    )
end

# parameters
function Adapt.adapt_structure(to, p::FEC.Parameters)
  return FEC.Parameters(
    adapt(to, p.ics),
    adapt(to, p.dirichlet_bcs),
    adapt(to, p.neumann_bcs),
    adapt(to, p.periodic_bcs),
    adapt(to, p.point_source_bcs),
    adapt(to, p.robin_bcs),
    adapt(to, p.sources),
    adapt(to, p.times),
    adapt(to, p.physics),
    adapt(to, p.properties),
    adapt(to, p.state_old),
    adapt(to, p.state_new),
    adapt(to, p.coords),
    adapt(to, p.field),
    adapt(to, p.field_old),
    # scratch fields
    adapt(to, p.hvp_scratch_field)
  )
end

# misc
FEC.to_backend(::KA.CPU, x) = adapt(Array, x)

end # module
