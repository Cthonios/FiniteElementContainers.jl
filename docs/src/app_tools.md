# AppTools
The submodule ``AppTools`` contains useful tools to help
researchers generate new applications, parse command line
arguments, parse input files, and various other routine
operations common to finite element programs.

# AppTools

`AppTools` is a collection of utilities for building **command-line finite element applications** on top of **FiniteElementContainers.jl**.

Rather than writing boilerplate code for command line parsing, input file processing, mesh loading, and simulation setup, `AppTools` provides a standardized interface that automatically constructs these objects from a simple configuration file.

The goal is to allow developers to focus on implementing new physics while reusing a common application infrastructure.

---

# Overview

Most FEM applications follow the same high-level workflow:

1. Parse command line arguments.
2. Open a log file.
3. Read an input deck.
4. Load the mesh.
5. Construct analytic functions.
6. Construct initial conditions.
7. Construct boundary conditions.
8. Build a simulation object.
9. Execute the solver.

`AppTools` automates Steps 1–8, leaving only the physics implementation to the application developer.

---

# Main Components

## App

`App` represents a finite element command line application.

It owns the command line parser and serves as the main entry point for application setup.

```julia
app = App{2}("HeatSolver")
sim = setup(app, ARGS)
```

After calling `setup`, a fully initialized `Simulation` object is returned.

---

## CLI Argument Parser

The framework contains a lightweight command line parser supporting

* required arguments
* optional arguments
* default values
* short names
* automatic help messages

For example

```bash
mysolver \
    --input-file input.toml \
    --log-file output.log
```

Additional application-specific arguments can be registered:

```julia
add_cli_arg!(
    app,
    "--num-steps";
    short_name="-n",
    default="100"
)
```

---

## Log Files

Every application writes a structured log file documenting

* parsed command line arguments
* input file contents
* parsed functions
* boundary conditions
* mesh information
* simulation setup

This provides a reproducible record of every simulation.

---

# Input Files

Applications are configured through TOML input files.

The parser currently supports

* mesh definitions
* analytic functions
* boundary conditions
* initial conditions

The raw TOML dictionary is also preserved for application-specific extensions.

---

## Mesh Settings

Meshes are specified using

```toml
[mesh]
file_path = "mesh.g"
file_type = "exodus"
```

Currently, Exodus meshes are supported.

The mesh loader automatically constructs an `UnstructuredMesh` containing

* nodal coordinates
* element connectivity
* element blocks
* node sets
* side sets
* global IDs

ready for finite element assembly.

---

# Function Definitions

Analytic functions may be declared once and reused throughout the input deck.

Example

```toml
[functions.temperature]
type = "scalar expression"
expression = "100*x + y"
variables = ["x","y"]
```

Vector-valued functions are also supported.

These functions are internally compiled into expression objects that may be evaluated during assembly.

---

# Boundary Conditions

Boundary conditions reference previously-defined functions.

Supported types include

* Dirichlet
* Neumann
* Robin
* Source terms

For example

```toml
[[boundary_conditions.dirichlet]]
variables = ["ux"]
function = "fixed"
side_sets = ["left"]
```

Boundary conditions may be attached to

* element blocks
* node sets
* side sets

depending on the application.

---

# Initial Conditions

Initial conditions are specified similarly to boundary conditions and reference existing analytic functions.

For example

```toml
[[initial_conditions]]
variables = ["temperature"]
function = "initial_temp"
blocks = ["block_1"]
```

---

# Simulation Object

After setup, all parsed information is collected into a `Simulation` object containing

* mesh
* Dirichlet BCs
* Neumann BCs
* Robin BCs
* source terms
* initial conditions
* log file

The solver implementation can immediately begin finite element assembly using these objects.

---

# Project Generation

`generate_app()` creates a new standalone FEM application.

```julia
generate_app("HeatSolver")
```

This automatically generates

```
HeatSolver/
    Project.toml
    build.jl
    src/
        HeatSolver.jl
```

including dependencies and a basic application skeleton.

Optional backends may be enabled:

```julia
generate_app(
    "MySolver",
    backends=["cpu","cuda","mpi"]
)
```

which automatically adds the necessary package dependencies.

---

# Building Applications

Generated applications may be compiled into standalone executables using

```julia
build_app()
```

which invokes `build.jl` and uses JuliaC to produce an executable.

---

# Running Applications

Compiled applications may be launched programmatically using

```julia
run_app([
    "--input-file",
    "input.toml",
    "--log-file",
    "run.log"
])
```

or directly from the command line.

---

# Philosophy

`AppTools` is designed to separate **physics implementation** from **application infrastructure**.

Solver developers should only need to implement

* constitutive models
* residual assembly
* tangent assembly
* time integration
* nonlinear solution algorithms

while all application setup, configuration parsing, logging, and mesh initialization are handled automatically by the framework.

This enables new finite element applications to be prototyped with minimal boilerplate while maintaining a consistent interface across multiple solvers.


```@autodocs
Modules = [FiniteElementContainers]
Pages = ["AppTools.jl"]
Order = [:module, :type, :function]
```