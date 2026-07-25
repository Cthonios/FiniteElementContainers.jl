# Parameters

`Parameters` bundles everything a physics evaluation needs that is not the
solution field itself: boundary and initial conditions, sources, the time
stepper, and the per-block `physics` and `properties`.

## Per-block physics and properties

`physics` and `properties` may be given either as a single object shared by the
whole mesh

```julia
p = create_parameters(mesh, asm, physics, props)
```

or as a `NamedTuple` with one entry per element block, keyed by block name

```julia
p = create_parameters(
    mesh, asm,
    (steel = steel_physics, foam = foam_physics),
    (steel = steel_props,   foam = foam_props)
)
```

In the single-object form the object is replicated across every block. In the
`NamedTuple` form the keys must be exactly the mesh's block names: a block with
no entry, or an entry naming no block, raises a `BlockMismatchError` rather than
running with some block silently taking another block's material.

The entries are reordered to match the block order of the function space (see
[Block order](@ref)), so the order the `NamedTuple` is written in does not
matter — only the names do. Downstream, `p.physics` and `p.properties` are
always keyed by block name and ordered by block index, which is the pairing
`foreach_block` and the assembly kernels rely on.

## API

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Parameters.jl"]
Order = [:type, :function]
```
