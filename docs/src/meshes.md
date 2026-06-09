# Meshes
Meshes in ```FiniteElementContainers``` leverage a very abstract interface. Currently, only an ```Exodus``` interface is directly supported within the main package but others could be readily supported through package extensions which we are planning on.

# Mesh Infrastructure

The mesh system provides a common interface for reading, storing, manipulating, and writing finite element meshes independently of the underlying file format.

Rather than exposing Exodus, Gmsh, or Abaqus APIs directly, the library converts all mesh data into a common `AbstractMesh` representation that can be consumed by function spaces, assemblers, physics models, and post-processing routines.

The overall design can be viewed as

```text
           Exodus (.g/.exo)
          /
Mesh File -----> FileMesh ------> UnstructuredMesh ------> FEM Application
          \
           Gmsh (.msh/.geo)

                        |
                        +--> FunctionSpace
                        +--> Boundary Conditions
                        +--> Initial Conditions
                        +--> Physics
                        +--> Assembly
                        +--> PostProcessor
```

The goal is that application code never depends on the mesh file format being used.

---

# Mesh Types

The mesh system consists of three layers.

## File Meshes

A `FileMesh` represents an open mesh file.

```julia
meshfile = FileMesh(ExodusMesh(), "mesh.g")
```

A `FileMesh` is intentionally lightweight and provides only an interface for reading data from disk.

The following methods define the minimal mesh reader interface:

```julia
element_blocks(mesh)
nodal_coordinates_and_ids(mesh)
num_dimensions(mesh)
```

Optional helper methods include

```julia
nodesets(mesh)
sidesets(mesh)
node_id_map(mesh)
```

New mesh file formats can be added simply by implementing these methods.

For example,

```julia
struct MyMeshFormat <: AbstractMeshFileType
end

function element_blocks(mesh::FileMesh{...,MyMeshFormat})
    ...
end
```

without modifying the rest of the FEM library.

---

## UnstructuredMesh

Most FEM algorithms operate on an `UnstructuredMesh`.

This object owns all mesh connectivity and geometric information required during analysis.

```julia
mesh = UnstructuredMesh("mesh.g")
```

Internally this stores

* nodal coordinates
* element connectivity
* element blocks
* element types
* node ID maps
* element ID maps
* nodesets
* sidesets
* optional edge connectivity
* optional face connectivity

allowing the remainder of the code base to be completely independent of Exodus or Gmsh.

For example,

```julia
mesh.nodal_coords

mesh.element_conns["block_1"]

mesh.nodeset_nodes["left"]

mesh.sideset_elems["traction"]
```

provide direct access to commonly used mesh data structures.

---

# Reading Meshes

The simplest way to construct a mesh is

```julia
mesh = UnstructuredMesh("mesh.g")
```

The constructor automatically dispatches based on file extension.

```text
.g
.e
.exo       -> Exodus reader

.msh
.geo       -> Gmsh reader

.inp
.i         -> Abaqus reader (planned)
```

Application code therefore remains independent of mesh format.

```julia
mesh = UnstructuredMesh(input_file)
```

works regardless of whether the mesh originated from Exodus or Gmsh.

---

# Mesh Summary

Meshes provide a convenient summary through Julia's display system.

```julia
println(mesh)
```

produces output similar to

```text
UnstructuredMesh:

  Number of dimensions = 2
  Number of nodes      = 1245

  Element Blocks:
      block_1
          Element type       = QUAD4
          Number of elements = 1024

  Node sets:
      left
      right

  Side sets:
      traction
```

which is often useful when debugging new applications.

---

# Element Blocks

Elements are grouped into **blocks**.

Each block has

* a name
* an element type
* a connectivity matrix
* an element ID map

For example,

```julia
conn = mesh.element_conns["solid"]

etype = mesh.element_types["solid"]
```

returns the connectivity and element type associated with the block named `"solid"`.

This organization makes it straightforward to assign different material models or physics objects to different regions of a mesh.

---

# Nodesets and Sidesets

Boundary conditions are typically applied through named mesh entities.

Node sets contain collections of nodes,

```julia
mesh.nodeset_nodes["fixed"]
```

while side sets contain element faces or edges,

```julia
mesh.sideset_elems["traction"]
mesh.sideset_sides["traction"]
```

allowing boundary conditions to be specified symbolically rather than through explicit node lists.

For example,

```julia
DirichletBC(
    "u",
    x -> 0.0;
    nodeset_name="fixed"
)
```

can automatically locate the appropriate degrees of freedom.

---

# Creating Edge and Face Connectivity

Many finite element methods require explicit edge or face connectivity.

These may be requested during mesh construction

```julia
mesh = UnstructuredMesh(
    "mesh.g",
    create_edges=true
)
```

which constructs a globally unique edge numbering from the element connectivity.

Similarly,

```julia
create_faces=true
```

can be used for three-dimensional meshes.

These connectivity objects are useful for

* H(curl) elements
* H(div) elements
* discontinuous Galerkin methods
* mortar methods
* adaptive refinement

without requiring repeated reconstruction of edge topology.

---

# Higher Order Mesh Generation

Linear meshes may be automatically upgraded to higher-order Lagrange meshes.

For example,

```julia
mesh2 = UnstructuredMesh(
    "mesh.g",
    p_order=2
)
```

creates a quadratic mesh by inserting mid-edge nodes and interior nodes into each element.

Internally this process

1. identifies unique mesh edges,
2. inserts edge nodes,
3. inserts element interior nodes,
4. updates element connectivity.

This provides a convenient mechanism for generating higher-order discretizations from existing linear meshes.

---

# Rigid Body Modes

The mesh infrastructure can automatically compute rigid body modes.

```julia
R = rigid_body_modes(mesh)
```

For two-dimensional meshes this returns

* x translation
* y translation
* rotation

while three-dimensional meshes return

* Tx
* Ty
* Tz
* Rx
* Ry
* Rz

These modes are useful for

* nullspace construction,
* multigrid methods,
* constrained optimization,
* singular stiffness matrices,
* elasticity verification tests.

---

# Writing Meshes

Meshes may be written back to disk using

```julia
write_to_file(mesh, "output.g")
```

which currently outputs Exodus meshes.

This allows generated meshes or modified connectivity to be saved for visualization or reuse.

---

# Copying Meshes

Existing Exodus meshes may be duplicated without reading all mesh data into memory.

```julia
copy_mesh(mesh, "new_mesh.g")
```

This is particularly useful for creating output databases prior to writing simulation results.

---

# Post Processing

The mesh infrastructure integrates directly with the post-processing system.

A postprocessor may be created from an existing mesh using

```julia
pp = PostProcessor(
    ExodusMesh,
    mesh,
    "results.g"
)
```

or from an existing Exodus file

```julia
pp = PostProcessor(
    ExodusMesh,
    "results.g",
    displacement,
    pressure
)
```

which automatically registers nodal and element variables.

Fields can then be written directly

```julia
write_field(
    pp,
    step,
    names(U),
    U
)
```

or

```julia
write_field(
    pp,
    step,
    "solid",
    "stress",
    stress
)
```

allowing visualization in ParaView, VisIt, or other Exodus-compatible tools.

---

# Extending the Mesh System

Supporting a new mesh format requires only implementing the minimal `FileMesh` interface.

```julia
element_blocks(mesh)

nodal_coordinates_and_ids(mesh)

num_dimensions(mesh)
```

along with optional helpers such as

```julia
nodesets(mesh)

sidesets(mesh)

node_id_map(mesh)
```

Once these methods are provided, the entire FEM infrastructure—including function spaces, assembly, physics objects, boundary conditions, and post-processing—works automatically without modification.

This separation between **mesh I/O** and **finite element algorithms** allows new mesh formats to be integrated with minimal effort while keeping the core FEM implementation completely file-format independent.


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

# Gmsh interface API
```@autodocs
Modules = [gmsh_ext]
Order = [:module, :type, :function]
```