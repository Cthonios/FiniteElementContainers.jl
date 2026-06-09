# MPI Parallelism (`PartitionedArraysExt`)

## Overview

`PartitionedArraysExt` provides distributed-memory parallelism for **FiniteElementContainers.jl** using **MPI** through the `PartitionedArrays.jl` ecosystem.

Unlike traditional FEM frameworks that require separate serial and parallel implementations of assembly and data structures, `PartitionedArraysExt` reuses the existing serial implementation by:

* partitioning the mesh,
* constructing local serial finite element objects on each MPI rank,
* assembling local element contributions independently,
* and using `PartitionedArrays.jl` to automatically communicate ghost values and assemble global distributed vectors and matrices.

The overall workflow is

```text
                Global Mesh
                     │
             Exodus decomposition
                     │
         ┌───────────┴────────────┐
         │                        │
      Rank 0                  Rank 1
         │                        │
 Local UnstructuredMesh    Local UnstructuredMesh
         │                        │
   Local FunctionSpace      Local FunctionSpace
         │                        │
     Local DofManager        Local DofManager
         │                        │
  Local Element Assembly  Local Element Assembly
         └───────────┬────────────┘
                     │
      PartitionedArrays communication
                     │
        Distributed Matrix / Vector
                     │
              Krylov Linear Solver
```

The key design principle is that **element integration remains completely serial** while only the global algebraic objects are distributed.

---

# Mesh Distribution

## `distribute_mesh`

```julia
distribute_mesh(file_name, n_ranks, ranks)
```

Partitions an Exodus mesh into `n_ranks` subdomains using the Exodus `decomp` utility.

```julia
distribute_mesh("mesh.g", 8, ranks)
```

produces

```text
mesh.g.8.000
mesh.g.8.001
...
mesh.g.8.007
```

Each MPI process subsequently reads only its local mesh file.

Internally

```julia
map_main(ranks) do rank
    decomp(file_name, n_ranks)
end
```

ensures that decomposition occurs only once before all ranks synchronize via

```julia
PartitionedArrays.barrier(ranks)
```

---

# Parallel Mesh

## `PUnstructuredMesh`

```julia
struct PUnstructuredMesh
    mesh
    num_ranks
    ranks
end
```

A `PUnstructuredMesh` is simply a distributed collection of ordinary serial `UnstructuredMesh` objects.

Each MPI rank owns one local mesh.

```text
Global mesh

      +----------------------+
      |                      |
      |                      |
      |                      |
      +----------------------+

            decomposed

      +----------+----------+
      | Rank 0   | Rank 1   |
      |          |          |
      +----------+----------+
```

Construction is simply

```julia
mesh = UnstructuredMesh("mesh.g", 4, ranks)
```

which internally loads

```text
mesh.g.4.000
mesh.g.4.001
mesh.g.4.002
mesh.g.4.003
```

on the respective MPI processes.

---

# Parallel Function Spaces

## `PFunctionSpace`

```julia
PFunctionSpace(mesh, field_type, interp_type)
```

constructs a serial `FunctionSpace` independently on every MPI process.

Internally

```julia
map(mesh.mesh) do local_mesh
    FunctionSpace(local_mesh, field_type, interp_type)
end
```

creates

```text
Rank 0
--------

serial FunctionSpace

Rank 1
--------

serial FunctionSpace

...
```

No parallel logic is needed during finite element interpolation.

---

# Global Numberings

One of the major responsibilities of the extension is maintaining the relationship between

* global field DOFs,
* global unknowns,
* local owned values,
* ghost values.

This is accomplished through several partition structures.

---

# Field Partition

## `FieldPartition`

`FieldPartition` represents the distribution of the **complete finite element field**, including ghost values.

```text
Global DOFs

1 2 3 4 5 6 7 8

Rank 0

1 2 3 4 5

Rank 1

4 5 6 7 8

      ghosts
```

Each partition stores

```text
global numbering
owner rank
local numbering
```

using `LocalIndices` from `PartitionedArrays.jl`.

---

# Solution Partition

## `SolutionPartition`

`SolutionPartition` stores only **active unknowns**.

Dirichlet DOFs are removed before partitioning.

For example

```text
Field DOFs

1 2 3 4 5 6 7

Dirichlet

2 6

Unknowns

1 3 4 5 7
```

becomes

```text
field id → unknown id

1 → 1
2 → -1
3 → 2
4 → 3
5 → 4
6 → -1
7 → 5
```

where `-1` indicates an eliminated Dirichlet degree of freedom.

This mapping is created by

```julia
_create_field_to_unknown(...)
```

and is central to condensation of boundary conditions.

---

# `PDofManager`

The distributed degree-of-freedom manager stores

* `field_partition`
* `solution_partition`
* `field_to_solution`
* `solution_to_field`
* `local_dof_managers`

Its purpose is to bridge

```text
serial FEM numbering
```

and

```text
distributed Krylov numbering
```

Each MPI process still owns a completely ordinary serial `DofManager`.

The distributed wrapper only manages communication and indexing.

---

# Updating Unknowns

After solving the linear system, unknown vectors must be copied back into the full finite element field.

This is accomplished by

```julia
update_field_unknowns!(U, dof, Uu)
```

The algorithm is

```text
for each owned field DOF

        global field id

              │

              ▼

field_to_solution lookup

              │

              ▼

global unknown id

              │

              ▼

local unknown id

              │

              ▼

copy value into field
```

Internally

```julia
solution_id = dof.field_to_solution[field_id]

if solution_id > 0
    U_local[i] = Uu_local[
        solution_gtl[solution_id]
    ]
end
```

which skips Dirichlet nodes automatically.

---

# Parallel Sparse Patterns

Finite element assembly first produces element-level arrays.

These must be scattered into distributed sparse matrices.

The module constructs explicit insertion patterns.

## `PSparseMatrixPattern`

stores `(Is, Js)` corresponding to global matrix row and column indices.

For every element

```text
      local stiffness

      x x x
      x x x
      x x x

            ↓

global sparse entries

(I1,J1)
(I1,J2)
(I1,J3)

...
```

Dirichlet rows and columns are removed during construction.

---

## `PSparseVectorPattern`

Similarly,

`PSparseVectorPattern`

stores insertion locations for residual vectors.

```text
element residual

r1
r2
r3

      ↓

global vector

I1
I2
I3
```

again omitting constrained DOFs.

---

# Parallel Assembly

The parallel assembler is

```julia
PSparseMatrixAssembler
```

which contains

* local assemblers
* matrix pattern
* vector pattern

The local assemblers are simply ordinary serial `SparseMatrixAssembler`s.

Assembly proceeds independently on every MPI process:

```julia
map(
    asm.local_assemblers,
    partition(u),
    p.local_parameters
) do local_asm, local_u, local_p

    assemble_stiffness!(
        local_asm,
        func,
        local_u,
        local_p
    )
end
```

No MPI communication occurs during element integration.

Communication occurs only when the distributed sparse matrix is constructed.

---

# Distributed Sparse Matrix Construction

Once local assembly is complete,

```julia
stiffness(asm)
```

collects local storage arrays

```julia
vals = map(asm.local_assemblers) do local_asm
    local_asm.stiffness_storage
end
```

and constructs a distributed sparse matrix

```julia
psparse(pattern, vals)
```

using the insertion patterns previously computed.

The resulting object is a `PartitionedArrays.PSparseMatrix`.

Likewise,

```julia
residual(asm)
```

returns a distributed `PVector`.

---

# Distributed Parameters

## `PParameters`

`PParameters` stores one parameter object per MPI rank.

```text
Rank 0

Material
Quadrature
History

Rank 1

Material
Quadrature
History
```

Each local assembler therefore operates entirely independently.

---

# Krylov Integration

The extension provides a specialization

```julia
CgWorkspace(
    A::PSparseMatrix,
    b::PVector
)
```

allowing **Krylov.jl** iterative solvers to work directly on distributed matrices and vectors.

This avoids converting to serial matrices and enables scalable MPI linear solves while retaining the standard Krylov API.

---

# Parallel Postprocessing

`PPostProcessor` wraps multiple serial postprocessors.

```text
Rank 0

output.e.4.000

Rank 1

output.e.4.001

...

Rank N

output.e.4.N
```

Writing fields is simply

```julia
write_field(pp, step, names, u)
```

which maps over all local postprocessors.

Optionally,

```julia
close(pp)
```

can invoke

```text
epu
```

to merge all distributed Exodus files into a single output database:

```text
output.e.4.000
output.e.4.001
output.e.4.002
output.e.4.003

        │
        ▼

      epu

        │
        ▼

output.e
```

This produces a standard Exodus file that can be visualized directly in ParaView.

---

# Design Philosophy

The central idea behind `PartitionedArraysExt` is to **reuse the serial finite element implementation unchanged**. Every MPI process performs ordinary serial finite element assembly on its local mesh partition. `PartitionedArrays.jl` is responsible for managing ownership, ghost values, communication, and distributed sparse algebra, while `FiniteElementContainers.jl` continues to operate on familiar serial data structures at the element level. This separation keeps the finite element kernels simple while allowing the package to scale to distributed-memory systems with minimal duplication of code.

# API
```@autodocs
Modules = [parrays_ext]
Order = [:module, :type, :function]
```