```@meta
CurrentModule = FiniteElementContainers
```

# Meshes
Meshes in ```FiniteElementContainers``` leverage a very abstract interface. Currently, only an ```Exodus``` interface is directly supported within the main package but others could be readily supported through package extensions which we are planning on.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Meshes.jl"]
Order = [:type, :function]
```

# Structured Meshes
Simple structured meshes on rectangles or parallepipeds can be create through ```StructuredMesh``` mesh type.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["StructuredMesh.jl"]
Order = [:type, :function]
```

# Unstructured Meshes
Unstructured meshes (e.g. those read from a file created by a mesher) can be created with the following mesh type

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["UnstructuredMesh.jl"]
Order = [:type, :function]
```

# Exodus interface API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Exodus.jl"]
Order = [:type, :function]
```