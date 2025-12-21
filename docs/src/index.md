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
the Poisson equation or heat equation, there's likely other FEM packages with less mental overhead out there for this purpose.

However, if you need to solve problems with multiple material models, meshes where
there are mixed elements types, etc. this is the package for you. 

Inspiration for the software design primarily comes from ```fenics``` and ```MOOSE```.
We've specifically designed the interface to get around all the shortcomings of ```fenics``` 
(e.g. boundary conditions are a pain, mixed element types are pain, different blocks are pain
etc.) 

Our goal is also to ensure all of our methods are next generation hardware capable. This
means not only supporting things on CPUs but also GPUs (and that doesn't just mean NVIDIA). The package is regularly tested against CUDA and RocM aware hardware to ensure all the types, methods, etc. work on CPUs and GPUs. Additionally, we test against Linux, MacOS, and Windows so in theory this should run on any personal computer out of the box. High performance computers are another story.