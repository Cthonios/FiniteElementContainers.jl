using TestItemRunner
using TestItems

@testitem "ScalarExpressionFunction - constants" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction

    string = "5.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.0

    string = "-5.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ -5.0

    string = "5.0 + 3.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 8.0

    string = "7.0 - 3.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 4.0

    string = "2.0 * 3.5"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 7.0

    string = "7.0 / 2.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 3.5

    string = "5.0 + (4.0 * 2.0) / 4.0 + 2.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 9.0

    string = "5.0^2.0"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 25.0
end

@testitem "ScalarExpressionFunction - test_different_floats" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction
    import FiniteElementContainers.Expressions: InvalidScientificNotationError
    string = "5.64e-12"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.64e-12

    string = "-4.68e-68"
    func = ScalarExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ -4.68e-68

    string = "53e-"
    @test_throws InvalidScientificNotationError ScalarExpressionFunction{Float64}(string, String[])
    # @test val ≈ -4.68e-68
end

@testitem "ScalarExpressionFunction - test_simple_variable_expressions" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction

    string = "x"
    func = ScalarExpressionFunction{Float64}(string, String["x"])
    val = func(Float64[53.0])
    @test val ≈ 53.0

    string = "x"
    func = ScalarExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 53.0

    string = "y"
    func = ScalarExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 40.0
end

@testitem "ScalarExpressionFunction - test_variable_arithmetic" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction

    string = "-x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ -5.0

    string = "x + x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 10.0

    string = "x - x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 0.0

    string = "x * x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 25.0

    string = "x / x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 1.0

    string = "x ^ x"
    func = ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 5.0^5.0

    string = "y + (x * x + 5) * y"
    func = ScalarExpressionFunction{Float64}(string, ["x", "y"])
    val = func([5.0, 2.0])
    @test val ≈ 62.0

    string = "x * y * exp(x + y)"
    func = ScalarExpressionFunction{Float64}(string, ["x", "y"])
    val = func([1.0, 2.0])
    @test val ≈ 1.0 * 2.0 * exp(1.0 + 2.0)
end

@testitem "ScalarExpressionFunction - builtin functions" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction

    funcs = [cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh]
    for func in funcs
        string = "$(String(Symbol(func)))(x)"
        expr_func = ScalarExpressionFunction{Float64}(string, ["x"])
        x = rand(Float64)
        val = expr_func(x)
        @test val ≈ func(x)
    end
end

@testitem "ScalarExpressionFunction - special method calls" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction
    import StaticArrays: SVector

    string = "x + y + z * t"
    func = ScalarExpressionFunction{Float64}(string, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 15.0
    val = func(X, t)
    @test val ≈ 48.0
end

@testitem "VectorExpressionFunction - constants" begin
    import FiniteElementContainers.Expressions: VectorExpressionFunction
    strings = ["0.0", "-5.0", "0.0"]
    func = VectorExpressionFunction{3, Float64}(strings, String[])
    val = func(Float64[])
    @test val ≈ [0.0, -5.0, 0.0]

    strings = ["10.0e-7", "-5.0e-3", "50.0e3"]
    func = VectorExpressionFunction{3, Float64}(strings, String[])
    val = func(Float64[])
    @test val ≈ [10.0e-7, -5.0e-3, 50e3]
end

@testitem "VectorExpressionFunction - time only" begin
    import FiniteElementContainers.Expressions: VectorExpressionFunction
    strings = ["1.0 * t", "-2.0 * t", "3.0 * t^2"]
    func = VectorExpressionFunction{3, Float64}(strings, ["t"])
    val = func(5.0)
    @test val ≈ [5.0, -10.0, 75.0]
end

@testitem "VectorExpressionFunction - all" begin
    import FiniteElementContainers.Expressions: VectorExpressionFunction
    import StaticArrays: SVector
    strings = ["x * t", "-2.0 * x * y * t^2", "5.0 * x^2 * z * exp(t)"]
    func = VectorExpressionFunction{3, Float64}(strings, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 5.0
    val = func(X, t)
    @test val[1] ≈ 5.0
    @test val[2] ≈ -100.0
    @test val[3] ≈ 15.0 * exp(5.0)
end

@testitem "ScalarExpressionFunction - is isbits (GPU + juliac safe)" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, FlatNode
    @test isbitstype(FlatNode{Float64})
    @test isbitstype(ScalarExpressionFunction{Float64})
    f = ScalarExpressionFunction{Float64}("a*exp(-(t-tc)^2 / (2*tau^2))",
                                          ["a", "tc", "tau", "t"])
    @test isbits(f)
end

@testitem "ScalarExpressionFunction - parser precedence: unary minus vs ^" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction
    # Standard math: `-t^2` parses as `-(t^2)`, not `(-t)^2`.
    f = ScalarExpressionFunction{Float64}("-t^2", ["t"])
    @test f([3.0]) ≈ -9.0
    @test f([-3.0]) ≈ -9.0

    g = ScalarExpressionFunction{Float64}("-2^2", String[])
    @test g(Float64[]) ≈ -4.0

    h = ScalarExpressionFunction{Float64}("2^-2", String[])
    @test h(Float64[]) ≈ 0.25
end

@testitem "differentiate - constants and variables" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate
    f = ScalarExpressionFunction{Float64}("3.14", ["x"])
    fp = differentiate(f, 1)
    @test fp([1.7]) ≈ 0.0
    @test fp([-9.3]) ≈ 0.0

    g = ScalarExpressionFunction{Float64}("x", ["x", "y"])
    @test differentiate(g, 1)([1.0, 2.0]) ≈ 1.0
    @test differentiate(g, 2)([1.0, 2.0]) ≈ 0.0
end

@testitem "differentiate - each unary operator" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate
    cases = [
        ("cos(x)",  x -> -sin(x)),
        ("cosh(x)", x ->  sinh(x)),
        ("exp(x)",  x ->  exp(x)),
        ("log(x)",  x ->  1.0 / x),
        ("sin(x)",  x ->  cos(x)),
        ("sinh(x)", x ->  cosh(x)),
        ("sqrt(x)", x ->  1.0 / (2 * sqrt(x))),
        ("tan(x)",  x ->  1.0 / cos(x)^2),
        ("tanh(x)", x ->  1.0 / cosh(x)^2),
    ]
    for (expr, dexpr) in cases
        f  = ScalarExpressionFunction{Float64}(expr, ["x"])
        fp = differentiate(f, 1)
        for x in (0.3, 0.7, 1.4, 2.6)
            @test fp([x]) ≈ dexpr(x) rtol=1e-12
        end
    end

    # Unary minus
    f  = ScalarExpressionFunction{Float64}("-x", ["x"])
    @test differentiate(f, 1)([5.0]) ≈ -1.0
end

@testitem "differentiate - binary operators" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate

    f = ScalarExpressionFunction{Float64}("x + y", ["x", "y"])
    @test differentiate(f, 1)([2.0, 3.0]) ≈ 1.0
    @test differentiate(f, 2)([2.0, 3.0]) ≈ 1.0

    f = ScalarExpressionFunction{Float64}("x - y", ["x", "y"])
    @test differentiate(f, 1)([2.0, 3.0]) ≈ 1.0
    @test differentiate(f, 2)([2.0, 3.0]) ≈ -1.0

    f = ScalarExpressionFunction{Float64}("x * y", ["x", "y"])
    @test differentiate(f, 1)([2.0, 3.0]) ≈ 3.0
    @test differentiate(f, 2)([2.0, 3.0]) ≈ 2.0

    f = ScalarExpressionFunction{Float64}("x / y", ["x", "y"])
    @test differentiate(f, 1)([2.0, 3.0]) ≈ 1/3
    @test differentiate(f, 2)([2.0, 3.0]) ≈ -2/9

    # Power: constant exponent
    f = ScalarExpressionFunction{Float64}("x^3", ["x"])
    @test differentiate(f, 1)([2.0]) ≈ 12.0

    # Power: constant base
    f = ScalarExpressionFunction{Float64}("2^x", ["x"])
    @test differentiate(f, 1)([3.0]) ≈ 8.0 * log(2.0)

    # Power: general
    f = ScalarExpressionFunction{Float64}("x^x", ["x"])
    @test differentiate(f, 1)([2.0]) ≈ 2.0^2.0 * (log(2.0) + 1.0)
end

@testitem "differentiate - Gaussian pulse to 2nd derivative" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate
    # g(t) = a * exp(-(t - tc)^2 / (2 τ^2))
    g   = ScalarExpressionFunction{Float64}(
        "a * exp(-(t - tc)^2 / (2 * tau^2))", ["a", "tc", "tau", "t"])
    gp  = differentiate(g, 4)
    gpp = differentiate(gp, 4)
    a, tc, τ = 1.0e-3, 2.5e-4, 5.0e-5
    for t in (0.0, 1.0e-4, 2.5e-4, 4.0e-4, 5.0e-4)
        η  = t - tc
        g_  = a * exp(-η^2 / (2 * τ^2))
        gp_ = -(η / τ^2) * g_
        gpp_= (η^2 / τ^4 - 1 / τ^2) * g_
        @test g([a, tc, τ, t])   ≈ g_   rtol=1e-12
        @test gp([a, tc, τ, t])  ≈ gp_  rtol=1e-12
        @test gpp([a, tc, τ, t]) ≈ gpp_ rtol=1e-9
    end
end

@testitem "differentiate - spatial (traveling-wave IC)" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate
    # u₀(z) = a * exp(-z^2 / (2 s^2));  ∂u/∂z = -(z/s²) u
    u0  = ScalarExpressionFunction{Float64}(
        "a * exp(-z^2 / (2 * s^2))", ["a", "s", "z"])
    duz = differentiate(u0, 3)
    a, s = 0.01, 0.02
    for z in (-0.04, -0.01, 0.0, 0.01, 0.04)
        u_  = a * exp(-z^2 / (2 * s^2))
        du_ = -(z / s^2) * u_
        @test u0([a, s, z])  ≈ u_  rtol=1e-12
        @test duz([a, s, z]) ≈ du_ rtol=1e-12
    end
end

@testitem "differentiate - var-name overload + error on unknown" begin
    import FiniteElementContainers.Expressions: ScalarExpressionFunction, differentiate
    f = ScalarExpressionFunction{Float64}("x + 2*t", ["x", "t"])
    fp = differentiate(f, ["x", "t"], "t")
    @test fp([0.5, 0.3]) ≈ 2.0
    @test_throws AssertionError differentiate(f, ["x", "t"], "y")
    @test_throws AssertionError differentiate(f, 5)
end
