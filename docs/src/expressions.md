# Expressions
This package supports type-stable expression functions that can be parsed from strings at runtime. This enables arbitrary boundary condition support in compiled binaries. This feature heavily leverages the package [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl)

## ScalarExpressionFunction
```@repl
import FiniteElementContainers.Expressions: ScalarExpressionFunction
using StaticArrays
expr = "5.0 * exp(-2 * t) * cos(2 * pi * t)"
var_names = ["x", "y", "z", "t"]
func = ScalarExpressionFunction{Float64}(expr, var_names)
X = SVector{3, Float64}(1., 2., 3.)
t = 5.0
val = func(X, t)
```

## VectorExpressionFunction
```@repl
import FiniteElementContainers.Expressions: VectorExpressionFunction
using StaticArrays
exprs = ["0", "5.0 * exp(-2 * t) * cos(2 * pi * t)", "0"]
var_names = ["x", "y", "z", "t"]
func = VectorExpressionFunction{3, Float64}(exprs, var_names)
X = SVector{3, Float64}(1., 2., 3.)
t = 5.0
val = func(X, t)
```

# API
```@autodocs
Modules = [FiniteElementContainers.Expressions]
Pages = ["Expressions.jl"]
Order = [:type, :function]
```