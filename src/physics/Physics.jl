abstract type AbstractPhysics{NF, NP, NS} end

num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

# this won't work
# function create_physics(name, vars...)
#   # @show field_vars
#   # @show prop_vars
#   H1_vars = _filter_field_type(vars, H1Field)
#   # TODO setup Hdiv and Hcurl logic as well
#   L2_element_vars = _filter_field_type(vars, L2ElementField)
#   L2_quadrature_vars = _filter_field_type(vars, L2QuadratureField)
  
#   H1_var_names = ()
#   for var in H1_vars
#     H1_var_names = (H1_var_names..., names(var)...)
#   end
#   H1_var_names = NamedTuple{H1_var_names}(1:length(H1_var_names))

#   L2_element_var_names = ()
#   for var in L2_element_vars
#     L2_element_var_names = (L2_element_var_names..., names(var)...)
#   end
#   L2_element_var_names = NamedTuple{L2_element_var_names}(1:length(L2_element_var_names))

#   L2_quadrature_var_names = ()
#   for var in L2_quadrature_vars
#     L2_quadrature_var_names = (L2_quadrature_var_names..., names(var)...)
#   end
#   L2_quadrature_var_names = NamedTuple{L2_quadrature_var_names}(1:length(L2_quadrature_var_names))
  
# end

# function initialize_state(::AbstractPhysics{NF, NP, 0}) where {NF, NP}
#   return 
# end

# physics like methods
function damping end
function energy end
function mass end
function residual end
function stiffness end

# optimization like methods
function gradient end
function hessian end
function value end
