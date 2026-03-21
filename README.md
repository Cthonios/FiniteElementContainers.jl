# FiniteElementContainers 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cthonios.github.io/FiniteElementContainers.jl/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cthonios.github.io/FiniteElementContainers.jl/dev/) 
[![Build Status](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI.yml?query=branch%3Amain) 
[![CUDA](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_CUDA.yml/badge.svg?branch=main)](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_CUDA.yml?query=branch%3Amain) 
[![ROCm](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_ROCM.yml/badge.svg?branch=main)](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_ROCM.yml?query=branch%3Amain) 
[![Coverage](https://codecov.io/gh/Cthonios/FiniteElementContainers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Cthonios/FiniteElementContainers.jl)

This package is meant to serve as a the minimal tools necessary to build new finite element method based applications for researchers working in challenging domains with e.g. large deformation, path dependence, contact, etc. All runtime intensive containers are written with the unique julia GPU infrastructure in mind. [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) is used in the few places where we have written custom kernels in tandem with [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl) to eliminate race conditions in assembly operations.

The goal is to be platform independent, provide CPU/GPU implementations, and leverage [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl) for easy implementation of new element types/formulations with out too much needed shim code. 

A semi-agnostic yet exodusII centric mesh interface is used via [Exodus.jl](https://github.com/cmhamel/Exodus.jl). This is the only mesh format directly supported within the main package module. For other mesh formats, package extensions are used for different mesh formats. There is support for gmsh files and abaqus files but these are not completely supported.

The long term goals are to enable mixed finite element space multiphysics problems but currently only H1 spaces are fully supported. Stay tuned for more details here.

# Installation
From the julia package manager one can do the following
```julia
pkg> add FiniteElementContainers
```

# Tutorials
A set of tutorials can be found at the following [link](https://cthonios.github.io/FiniteElementContainers.jl/dev/)

# Applications built on FiniteElementContainers.jl
* [Carina.jl](https://github.com/Cthonios/Carina.jl)
* [Cthonios.jl](https://github.com/Cthonios/Cthonios.jl)

# CPU execution
If you're only interested in using the package on CPU backends, you do not need to do anything special.

# GPU execution
If you would like to use a specific GPU backend, you'll need to add appropriate packages to your environment such as [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) for AMD devices or [CUDA.jl](https://github.com/juliagpu/cuda.jl) for Nvidia based devices. We have currently only tested against AMD and Nvidia devices although ~90% of the package "should" work on Intel devices. Metal is probably a different story.

We have currently tested the package on the following list of devices

* AMD Radeon RX 7900 XT
* AMD Radeon RX 7600 (Navi 33)
* NVIDIA Tesla V100-PCI-32GB

If you have had success with this package on other devices, please open a PR with a change to the README.

# MPI execution
This is not completely supported yet. This is actively being worked on with [PartitionedArrays.jl](https://github.com/PartitionedArrays/PartitionedArrays.jl) as the distributed linear algebra backend.
