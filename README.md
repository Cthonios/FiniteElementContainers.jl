# FiniteElementContainers 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cthonios.github.io/FiniteElementContainers.jl/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cthonios.github.io/FiniteElementContainers.jl/dev/) 
[![Build Status](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI.yml?query=branch%3Amain) 
[![ROCm](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_ROCM.yml/badge.svg?branch=main)](https://github.com/Cthonios/FiniteElementContainers.jl/actions/workflows/CI_ROCM.yml?query=branch%3Amain) 
[![Coverage](https://codecov.io/gh/Cthonios/FiniteElementContainers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Cthonios/FiniteElementContainers.jl)

This package is meant to serve as a light weight and allocation free set of containers for carrying out finite element or finite element-like calculations.

This package is meant to serve as a the minimal tools necessary to build new finite element method based applications for researchers working in challenging domains with e.g. large deformation, path dependence, contact, etc. All runtime intensive containers are written with the unique julia GPU infrastructure in mind. [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) is used in the few places where we have written custom kernels in tandem with [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl) to eliminate race conditions in assembly operations.

The goal is to be platform independent, provide CPU/GPU implementations, and leverage [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl) for easy implementation of new element types/formulations with out too much needed shim code. 

A semi-agnostic yet exodusII centric mesh interface is used. No single mesh format is directly supported within the main package module. Instead, package extensions are used for different mesh formats. Currently only exodusII files are supported for IO through [Exodus.jl](https://github.com/cmhamel/Exodus.jl).

The long term goals are to enable mixed finite element space multiphysics problems but currently only H1 spaces are fully supported. Stay tuned for more details here.

