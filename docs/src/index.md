```@meta
CurrentModule = FiniteElementContainers
```
# FiniteElementContainers
```FiniteElementContainers.jl``` is a package whose main purpose is to help 
researchers develop new finite element method (FEM) applications
for both well known existing techniques and new novel strategies.

This package is specificially designed with the challenging aspects
of computational solid mechanics in mind where meshes deform,
there's path dependence, there's contact between bodies, there are potentially heterogeneous material properties, and other challenges.

If you're primarily interested in writing FEM applications for e.g.
the Poisson equation or heat equation, there's likely more efficient packages (e.g. ```Gridap``` or ```Ferrite```) out there for this purpose in terms of memory and computational efficiency.

However, if you need to solve problems with multiple material models, meshes where
there are mixed elements types, etc. this is likely the only julia package
at the time of writing this that supports such capabilities. 

Inspiration for the software design primarily comes from ```fenics``` and ```MOOSE```.
We've specifically designed the interface to get around all the shortcomings of ```fenics``` 
(e.g. boundary conditions are a pain, mixed element types a plain, different blocks are pain
etc.) 

Our goal is also to ensure all of our methods are next generation hardware capable. This
means not only supporting things on CPUs but also GPUs (and that doesn't just mean NVIDIA).