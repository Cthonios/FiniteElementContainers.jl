```@meta
CurrentModule = FiniteElementContainers
```

# Fields
Fields serve as loose wrappers around ```AbstractArray``` subtypes such that the size of the array is known at compile time. Although this introduces a type-instability, the idea is to do this at the top most level (mainly at setup time of a FEM simulation). By introducing this type instability, we can gain information about the field type that is used in methods downstream to construct ```StaticArray```s of ```view```s of field types.

## Example - NodalField
We can set up a ```NodalField``` in one of two ways. The simplest constructor form can be used as follows
```julia
julia> vals = rand(2, 10)
2×10 Matrix{Float64}:
 0.671652  0.163963  0.538689  0.480536  0.833398  0.551275  0.790613  0.609717  0.383385  0.0387093
 0.074336  0.963916  0.658381  0.902902  0.642238  0.617257  0.566368  0.71399   0.493144  0.153415

julia> field = NodalField{2, 10, Matrix}(vals)
2×10 FiniteElementContainers.SimpleNodalField{Float64, 2, 2, 10, Matrix{Float64}}:
 0.671652  0.163963  0.538689  0.480536  0.833398  0.551275  0.790613  0.609717  0.383385  0.0387093
 0.074336  0.963916  0.658381  0.902902  0.642238  0.617257  0.566368  0.71399   0.493144  0.153415

julia> field.vals
2×10 Matrix{Float64}:
 0.671652  0.163963  0.538689  0.480536  0.833398  0.551275  0.790613  0.609717  0.383385  0.0387093
 0.074336  0.963916  0.658381  0.902902  0.642238  0.617257  0.566368  0.71399   0.493144  0.153415

```

We could also store this in vectorized format as follows

```julia
julia> vals = rand(2, 10)
2×10 Matrix{Float64}:
 0.80535   0.730575  0.712725  0.474454  0.0892281  0.156759  0.0425675  0.864044  0.538667  0.377565
 0.348685  0.852966  0.531284  0.34784   0.133556   0.483717  0.280693   0.155209  0.827217  0.532938

julia> field = NodalField{2, 10, Vector}(vals)
2×10 FiniteElementContainers.VectorizedNodalField{Float64, 2, 2, 10, Vector{Float64}}:
 0.80535   0.730575  0.712725  0.474454  0.0892281  0.156759  0.0425675  0.864044  0.538667  0.377565
 0.348685  0.852966  0.531284  0.34784   0.133556   0.483717  0.280693   0.155209  0.827217  0.532938

julia> field.vals
20-element Vector{Float64}:
 0.805349766666621
 0.34868545083892455
 0.7305749598014636
 0.8529660539980414
 0.7127252744440836
 0.5312840325763362
 0.47445360786490387
 0.34784023927079855
 0.08922808334086696
 ⋮
 0.04256750367757933
 0.28069295436131725
 0.864043693939776
 0.15520937344242647
 0.5386666785752343
 0.8272174353577112
 0.3775645360156227
 0.532938439960596

```

## Implementation
The base type for fields is the ```AbstractField``` abstract type. 

```@docs
AbstractField
```

Any new field added to ```FiniteElementContainers``` should be a subtype of this type.

The existing direct subtypes of ```AbstractField``` are the following

```@docs
ElementField
NodalField
QuadratureField
```

## Types
There's several different implementations currently for different field types.

## Methods on ```AbstractField```
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/Fields.jl"]
Order = [:function]
```

## Internal constructors for ```ElementField```s
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/SimpleElementField.jl", "fields/VectorizedElementField.jl"]
Order = [:type]
```

## Internal constructors for ```NodalField```s
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/SimpleNodalField.jl", "fields/VectorizedNodalField.jl"]
Order = [:type]
```

## Internal constructors for ```QuadratureField```s
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["fields/SimpleQuadratureField.jl", "fields/VectorizedQuadratureField.jl"]
Order = [:type]
```