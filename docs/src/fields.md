```@meta
CurrentModule = FiniteElementContainers
```

# Fields
Fields serve as loose wrappers around ```AbstractArray``` subtypes such that the size of array slices are known at compile time. Although this introduces a type-instability, the idea is to do this at the top most level (mainly at setup time of a FEM simulation). By introducing this type instability, we can gain information about the field type that is used in methods downstream to construct ```StaticArray```s of ```view```s of field types.

All fields are subtypes of the abstract type ```AbstractField```
```@repl
using FiniteElementContainers
FiniteElementContainers.AbstractField
```

## Example - H1Field a.k.a. NodalField
We can set up a ```H1Field``` in one of two ways. The simplest constructor form can be used as follows
```@repl h1field
using FiniteElementContainers
field = H1Field(rand(2, 10))
```
This is stored in a vectorized way as can be seen above
```@repl h1field
field.data
```
Fields can be indexed like regular arrays, e.g.
```@repl h1field
field[1, 1]
```
```@repl h1field
field[1, :]
```
etc.

## Abstract type
The base type for fields is the ```AbstractField``` abstract type. 
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/Fields.jl"]
Order = [:type]
```
Any new field added to ```FiniteElementContainers``` should be a subtype of this type.

## Methods for ```AbstractField```
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/Fields.jl"]
Order = [:function]
```

## Implementations
The existing direct subtypes of ```AbstractField``` are the following

### Connectivity
The connectivity type is a simple alias for ```L2ElementField``` defined below
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/Connectivity.jl"]
Order = [:type, :function]
```

### H1 field
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/H1Field.jl"]
Order = [:type, :function]
```

### L2Element field
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/L2ElementField.jl"]
Order = [:type, :function]
```

### L2Quadrature field
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/L2QuadratureField.jl"]
Order = [:type, :function]
```

There are plans to add ```HcurlField``` and ```HdivField``` types as well
