```@meta
CurrentModule = FiniteElementContainers
```

# Function spaces
```@docs
FunctionSpace
```

# Implementations
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["./function_spaces/NonAllocatedFunctionSpace.jl"]
Order = [:type, :function, :constant, :macro]
```

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["./function_spaces/VectorizedPreAllocatedFunctionSpace.jl"]
Order = [:type, :function, :constant, :macro]
```

## Useful methods
```@docs
dof_connectivity
element_level_fields
element_level_fields_reinterpret
reference_element
quadrature_level_field_values
quadrature_level_field_gradients
volume
```
