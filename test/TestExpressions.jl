using TestItemRunner
using TestItems

@testitem "Expression - constants" begin
    import FiniteElementContainers.Expressions: ExpressionFunction

    string = "5.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.0

    string = "-5.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ -5.0

    string = "5.0 + 3.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 8.0

    string = "7.0 - 3.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 4.0

    string = "2.0 * 3.5"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 7.0

    string = "7.0 / 2.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 3.5

    string = "5.0 + (4.0 * 2.0) / 4.0 + 2.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 9.0

    string = "5.0^2.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 25.0
end

@testitem "Expression - test_different_floats" begin
    import FiniteElementContainers.Expressions: ExpressionFunction
    import FiniteElementContainers.Expressions: InvalidScientificNotationError
    string = "5.64e-12"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.64e-12

    string = "-4.68e-68"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ -4.68e-68

    string = "53e-"
    @test_throws InvalidScientificNotationError ExpressionFunction{Float64}(string, String[])
    # @test val ≈ -4.68e-68
end

@testitem "Expression - test_simple_variable_expressions" begin
    import FiniteElementContainers.Expressions: ExpressionFunction

    string = "x"
    func = ExpressionFunction{Float64}(string, String["x"])
    val = func(Float64[53.0])
    @test val ≈ 53.0

    string = "x"
    func = ExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 53.0

    string = "y"
    func = ExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 40.0
end

@testitem "Expression - test_variable_arithmetic" begin
    import FiniteElementContainers.Expressions: ExpressionFunction

    string = "-x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ -5.0

    string = "x + x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 10.0

    string = "x - x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 0.0

    string = "x * x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 25.0

    string = "x / x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 1.0

    string = "x ^ x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 5.0^5.0

    string = "y + (x * x + 5) * y"
    func = ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([5.0, 2.0])
    @test val ≈ 62.0

    string = "x * y * exp(x + y)"
    func = ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([1.0, 2.0])
    @test val ≈ 1.0 * 2.0 * exp(1.0 + 2.0)
end

@testitem "Expression - builtin functions" begin
    import FiniteElementContainers.Expressions: ExpressionFunction

    funcs = [cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh]
    for func in funcs
        string = "$(String(Symbol(func)))(x)"
        expr_func = ExpressionFunction{Float64}(string, ["x"])
        x = rand(Float64)
        val = expr_func(x)
        @test val ≈ func(x)
    end
end

@testitem "Expression - special method calls" begin
    import FiniteElementContainers.Expressions: ExpressionFunction
    import StaticArrays: SVector

    string = "x + y + z * t"
    func = ExpressionFunction{Float64}(string, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 15.0
    val = func(X, t)
end
