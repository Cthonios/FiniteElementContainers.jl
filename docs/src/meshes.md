```@meta
CurrentModule = FiniteElementContainers
```

# Meshes
Meshes in ```FiniteElementContainers``` leverage a very abstract interface. No
single mesh format is directly supported in the main src code but rather different
mesh types are relegated to package extensions. Currently, only an ```Exodus``` package
extension is supported but others could be readily supported.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Meshes.jl"]
Order = [:type, :function]
```